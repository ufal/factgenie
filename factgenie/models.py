#!/usr/bin/env python3

import traceback
from openai import OpenAI
import json

import os
import logging
import requests
import copy

from ast import literal_eval
from factgenie.campaigns import CampaignMode

logger = logging.getLogger(__name__)

DIR_PATH = os.path.dirname(__file__)
LLM_ANNOTATION_DIR = os.path.join(DIR_PATH, "annotations")
LLM_GENERATION_DIR = os.path.join(DIR_PATH, "outputs")


class ModelFactory:
    """Register any new model here."""

    @staticmethod
    def model_classes():
        return {
            CampaignMode.LLM_EVAL: {
                "openai_metric": OpenAIMetric,
                "ollama_metric": OllamaMetric,
            },
            CampaignMode.LLM_GEN: {
                "openai_gen": OpenAIGen,
                "ollama_gen": OllamaGen,
                "tgwebui_gen": TextGenerationWebuiGen,
            },
        }

    @staticmethod
    def from_config(config, mode):
        metric_type = config["type"]
        classes = ModelFactory.model_classes()[mode]

        if metric_type not in classes:
            raise ValueError(f"Model type {metric_type} is not implemented.")

        return classes[metric_type](config)


class Model:
    def __init__(self, config):
        self.validate_config(config)
        self.config = config
        self.parse_model_args()

        if "extra_args" in config:
            # the key in the model output that contains the annotations
            self.annotation_key = config["extra_args"].get("annotation_key", "annotations")

    def get_annotator_id(self):
        return "llm-" + self.config["type"] + "-" + self.config["model"]

    def get_config(self):
        return self.config

    def parse_model_args(self):
        if "model_args" not in self.config:
            return

        for arg in self.config["model_args"]:
            try:
                self.config["model_args"][arg] = literal_eval(self.config["model_args"][arg])
            except:
                pass

    def validate_config(self, config):
        for field in self.get_required_fields():
            assert field in config, f"Field `{field}` is missing in the config. Keys: {config.keys()}"

        for field, field_type in self.get_required_fields().items():
            assert isinstance(
                config[field], field_type
            ), f"Field `{field}` must be of type {field_type}, got {config[field]=}"

        for field, field_type in self.get_optional_fields().items():
            if field in config:
                assert isinstance(
                    config[field], field_type
                ), f"Field `{field}` must be of type {field_type}, got {config[field]=}"
            else:
                # set the default value for the data type
                config[field] = field_type()

        # warn if there are any extra fields
        for field in config:
            if field not in self.get_required_fields() and field not in self.get_optional_fields():
                logger.warning(f"Field `{field}` is not recognized in the config.")


class LLMMetric(Model):
    def get_required_fields(self):
        return {
            "type": str,
            "annotation_span_categories": list,
            "prompt_template": str,
            "model": str,
            "model_args": dict,
        }

    def get_optional_fields(self):
        return {
            "system_msg": str,
            "start_with": str,
            "api_url": str,
            "extra_args": dict,
        }

    def postprocess_annotations(self, text, model_json):
        annotation_list = []
        current_pos = 0

        if self.annotation_key not in model_json:
            logger.error(f"Cannot find the key `{self.annotation_key}` in {model_json=}")
            return annotation_list

        for annotation in model_json[self.annotation_key]:
            # find the `start` index of the error in the text
            start_pos = text.lower().find(annotation["text"].lower(), current_pos)

            if start_pos == -1:
                logger.warning(f"Cannot find {annotation=} in text {text}, skipping")
                continue

            annotation["start"] = start_pos
            annotation_list.append(copy.deepcopy(annotation))

            current_pos = start_pos + len(annotation["text"])

        return annotation_list

    def preprocess_data_for_prompt(self, data):
        """Override this method to change the format how the data is presented in the prompt. See self.prompt() method for usage."""
        return data

    def prompt(self, data, text):
        assert isinstance(text, str) and len(text) > 0, f"Text must be a non-empty string, got {text=}"
        data_for_prompt = self.preprocess_data_for_prompt(data)

        prompt_template = self.config["prompt_template"]

        # we used to require replacing any curly braces with double braces
        # to support existing prompts, we replace any double braces with single braces
        # this should not do much harm, as the prompts usually do contain double braces (but remove this in the future?)
        prompt_template = prompt_template.replace("{{", "{").replace("}}", "}")

        return prompt_template.replace("{data}", str(data_for_prompt)).replace("{text}", text)

    def annotate_example(self, data, text):
        """
        Annotate the given text with the model.

        Args:
            data: the data to be used in the prompt
            text: the text to be annotated

        Returns:
            A dictionary: {
                "prompt": the prompt used for the annotation,
                "annotations": a list of annotations
            }
        """
        raise NotImplementedError("Override this method in the subclass to call the LLM API")


class OpenAIMetric(LLMMetric):
    def __init__(self, config):
        super().__init__(config)
        self.client = OpenAI()

    def get_required_fields(self):
        return {
            "type": str,
            "annotation_span_categories": list,
            "prompt_template": str,
            "model": str,
        }

    def get_optional_fields(self):
        return {
            "system_msg": str,
            "model_args": dict,
            "api_url": str,  # TODO we receive it from the UI, but can be removed
            "extra_args": dict,  # TODO we receive it from the UI, but can be removed
        }

    def annotate_example(self, data, text):
        try:
            prompt = self.prompt(data, text)

            logger.debug(f"Calling OpenAI API with prompt: {prompt}")
            response = self.client.chat.completions.create(
                model=self.config["model"],
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self.config["system_msg"]},
                    {"role": "user", "content": prompt},
                ],
                **self.config.get("model_args", {}),
            )
            annotation_str = response.choices[0].message.content
            j = json.loads(annotation_str)
            logger.info(j)

            return {"prompt": prompt, "annotations": self.postprocess_annotations(text=text, model_json=j)}
        except Exception as e:
            traceback.print_exc()
            logger.error(e)
            raise e


class OllamaMetric(LLMMetric):
    def __init__(self, config):
        super().__init__(config)

        self.set_api_endpoint()

    def set_api_endpoint(self):
        # make sure the API URL ends with the `generate` endpoint
        self.config["api_url"] = self.config["api_url"].rstrip("/")
        if not self.config["api_url"].endswith("/generate"):
            self.config["api_url"] += "/generate/"

    def postprocess_output(self, output):
        output = output.strip()
        j = json.loads(output)

        if self.config["model"].startswith("llama3"):
            # the model often tends to produce a nested list
            annotations = j[self.annotation_key]
            if isinstance(annotations, list) and len(annotations) >= 1 and isinstance(annotations[0], list):
                j[self.annotation_key] = j[self.annotation_key][0]

        return j

    def annotate_example(self, data, text):
        prompt = self.prompt(data=data, text=text)
        request_d = {
            "model": self.config["model"],
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": self.config.get("model_args", {}),
        }
        msg = f"Ollama API {self.config['api_url']} with args:\n\t{request_d}"
        response, annotation_str, j = None, None, None
        try:
            logger.debug(f"Calling {msg}")
            response = requests.post(self.config["api_url"], json=request_d)
            if response.status_code != 200:
                raise ValueError(f"Received status code {response.status_code} from the API. Response: {response.text}")

            try:
                response_json = response.json()
            except json.JSONDecodeError:
                logger.warning(f"Received non-JSON response: {response.text}")
                return []

            annotation_str = response_json["response"]

            j = self.postprocess_output(annotation_str)
            logger.info(j)
            return {
                "prompt": prompt,
                "annotations": self.postprocess_annotations(text=text, model_json=j),
            }
        except (ConnectionError, requests.exceptions.ConnectionError) as e:
            # notifiy the user that the API is down
            logger.error(f"Connection error: {e}")
            raise e
        except Exception as e:
            # ignore occasional problems not to interrupt the annotation process
            logger.error(f"Received\n\t{response=}\n\t{annotation_str=}\n\t{j=}\nError:{e}")
            traceback.print_exc()
            return {}


class LLMGen(Model):
    def get_required_fields(self):
        return {
            "type": str,
            "prompt_template": str,
            "model": str,
            "model_args": dict,
        }

    def get_optional_fields(self):
        return {
            "system_msg": str,
            "api_url": str,
            "extra_args": dict,
            "start_with": str,
        }

    def postprocess_output(self, output):
        if self.config.get("extra_args", {}).get("remove_suffix", ""):
            suffix = self.config["extra_args"]["remove_suffix"]

            if output.endswith(suffix):
                output = output[: -len(suffix)]

        output = output.strip()
        return output

    def prompt(self, data):
        prompt_template = self.config["prompt_template"]
        data = self.preprocess_data_for_prompt(data)

        # we used to require replacing any curly braces with double braces
        # to support existing prompts, we replace any double braces with single braces
        # this should not do much harm, as the prompts usually do contain double braces (but remove this in the future?)
        prompt_template = prompt_template.replace("{{", "{").replace("}}", "}")

        return prompt_template.replace("{data}", str(data))

    def preprocess_data_for_prompt(self, data):
        """Override this method to change the format how the data is presented in the prompt. See self.prompt() method for usage."""
        return data

    def generate_output(self, data):
        """
        Generate the output with the model.

        Args:
            data: the data to be used in the prompt

        Returns:
            A dictionary: {
                "prompt": the prompt used for the generation,
                "output": the generated output
            }
        """
        raise NotImplementedError("Override this method in the subclass to call the LLM API")


class OpenAIGen(LLMGen):
    def __init__(self, config):
        super().__init__(config)
        self.client = OpenAI()

    def get_required_fields(self):
        return {
            "type": str,
            "prompt_template": str,
            "model": str,
        }

    def get_optional_fields(self):
        return {
            "system_msg": str,
            "model_args": dict,
            "api_url": str,  # TODO we receive it from the UI, but can be removed
            "extra_args": dict,  # TODO we receive it from the UI, but can be removed
            "start_with": str,
        }

    def generate_output(self, data):
        try:
            prompt = self.prompt(data)

            messages = [
                {"role": "system", "content": self.config["system_msg"]},
                {"role": "user", "content": prompt},
            ]

            if self.config.get("start_with"):
                messages.append({"role": "assistant", "content": self.config["start_with"]})

            logger.debug(f"Calling OpenAI API with prompt: {prompt}")
            response = self.client.chat.completions.create(
                model=self.config["model"],
                messages=messages,
                **self.config.get("model_args", {}),
            )
            output = response.choices[0].message.content
            output = self.postprocess_output(output)
            logger.info(output)

            return {"prompt": prompt, "output": output}

        except Exception as e:
            traceback.print_exc()
            logger.error(e)
            raise e


class TextGenerationWebuiGen(LLMGen):
    def __init__(self, config):
        super().__init__(config)

    def get_required_fields(self):
        return {"type": str, "prompt_template": str, "model": str, "api_url": str, "extra_args": dict}

    def get_optional_fields(self):
        return {
            "system_msg": str,
            "model_args": dict,
            "start_with": str,
        }

    def generate_output(self, data):
        try:
            prompt = self.prompt(data)
            api_url = self.config["api_url"]
            api_key = self.config["extra_args"]["api_key"]

            messages = [
                {"role": "user", "content": prompt},
            ]

            if self.config.get("start_with"):
                messages.append({"role": "assistant", "content": self.config["start_with"]})

            model_args = self.config.get("model_args", {})
            logger.debug(f"Calling Text Generation Webui API with prompt: {prompt}")
            response = requests.post(
                api_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json={"model": self.config["model"], "messages": messages, **model_args},
            )

            output = response.choices[0].message.content
            output = self.postprocess_output(output)
            logger.info(output)

            return {"prompt": prompt, "output": output}
        except Exception as e:
            traceback.print_exc()
            logger.error(e)
            raise e


class OllamaGen(LLMGen):
    def __init__(self, config):
        super().__init__(config)

        self.set_api_endpoint()

    def set_api_endpoint(self):
        # make sure the API URL ends with the `chat` endpoint
        self.config["api_url"] = self.config["api_url"].rstrip("/")
        if not self.config["api_url"].endswith("/chat"):
            self.config["api_url"] += "/chat/"

    def generate_output(self, data):
        try:
            prompt = self.prompt(data=data)

            messages = [
                {"role": "system", "content": self.config["system_msg"]},
                {"role": "user", "content": prompt},
            ]
            if self.config.get("start_with"):
                messages.append({"role": "assistant", "content": self.config["start_with"]})

            request_d = {
                "model": self.config["model"],
                "messages": messages,
                "stream": False,
                "options": self.config.get("model_args", {}),
            }
            msg = f"Ollama API {self.config['api_url']} with args:\n\t{request_d}"
            response, output = None, None

            logger.debug(f"Calling {msg}")

            response = requests.post(self.config["api_url"], json=request_d)
            response_json = response.json()

            if "error" in response_json:
                raise ValueError(f"Received error from the API: {response_json['error']}")

            output = response_json["message"]["content"]
            output = self.postprocess_output(output)
            logger.info(output)
            return {"prompt": prompt, "output": output}
        except (ConnectionError, requests.exceptions.ConnectionError) as e:
            # notifiy the user that the API is down
            logger.error(f"Connection error: {e}")
            raise e
        except Exception as e:
            # ignore occasional problems not to interrupt the annotation process
            logger.error(f"Received\n\t{response=}\n\t{output=}\nError:{e}")
            traceback.print_exc()
            raise e
