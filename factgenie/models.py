#!/usr/bin/env python3

from abc import abstractmethod
import traceback
from typing import Optional
from openai import OpenAI
from textwrap import dedent
import json

import os
import logging
from pydantic import BaseModel, Field, ValidationError
import requests
import copy

from ast import literal_eval
from factgenie.campaign import CampaignMode

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
                "vllm_metric": VLLMMetric,
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


class SpanAnnotation(BaseModel):
    text: str = Field(description="The text which is annotated.")
    # Do not name it type since it is a reserved keyword in JSON schema
    annotation_type: int = Field(
        description="Index to the list of span annotation types defined for the annotation campaign."
    )
    reason: str = Field(description="The reason for the annotation.")


class OutputAnnotations(BaseModel):
    annotations: list[SpanAnnotation] = Field(description="The list of annotations.")


class Model:
    def __init__(self, config):
        self.validate_config(config)
        self.config = config
        self.parse_model_args()

    @property
    def new_connection_error_advice_docstring(self):
        return """Please check the LLM engine documentation. The call to the LLM API server failed."""

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

    def parse_annotations(self, text, annotations_json):
        try:
            annotations_obj = OutputAnnotations.parse_raw(annotations_json)
            annotations = annotations_obj.annotations
        except ValidationError as e:
            logger.error(f"LLM response in not in the expected format: {e}\n\t{annotations_json=}")

        annotation_list = []
        current_pos = 0
        for annotation in annotations:
            # find the `start` index of the error in the text
            start_pos = text.lower().find(annotation.text.lower(), current_pos)

            if start_pos == -1:
                logger.warning(f"Cannot find {annotation=} in text {text}, skipping")
                continue

            annotation_d = annotation.dict()
            # For backward compatibility let's use shorter "type"
            # We do not use the name "type" in JSON schema for error types because it has much broader sense in the schema (e.g. string or integer)
            annotation_d["type"] = annotation.annotation_type
            del annotation_d["annotation_type"]
            # logging where the annotion starts to disambiguate errors on the same string in different places
            annotation_d["start"] = start_pos
            annotation_list.append(annotation_d)

            current_pos = start_pos + len(annotation.text)  # does not allow for overlapping annotations

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


class OpenAIClientMetric(LLMMetric):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.client = OpenAI(**kwargs)

        config_schema = config.get("extra_args", {}).get("schema", {})
        pydantic_schema = OutputAnnotations.model_json_schema()
        if config_schema:
            self._schema = config_schema
            logger.warning(
                f"We expect parsing according to \n{pydantic_schema=}\n but got anoter schema from config\n{config_schema=}"
                "\nAdapt parsing accordingly!"
            )
        else:
            self._schema = pydantic_schema

        # Required for  OpenAI API but make sense in general too
        # TODO make it more pydantic / Python friendly
        self._schema["additionalProperties"] = False
        self._schema["$defs"]["Annotation"]["additionalProperties"] = False

        logger.warning(f"The schema is set to\n{self._schema}.\n\tCheck that your prompt is compatible!!! ")
        # access the later used config keys early to log them once and test if they are present
        logger.info(f"Using {config['model']=} with {config['system_msg']=}")

    @property
    def schema(self):
        return self._schema

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

    @abstractmethod
    def _prepare_chat_completions_create_args(self):
        raise NotImplementedError("Override this method in the subclass to prepare the arguments for the OpenAI API")

    def annotate_example(self, data, text):
        try:
            prompt = self.prompt(data, text)

            logger.debug(f"Calling OpenAI API with prompt: {prompt}")

            model = self.config["model"]

            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.config["system_msg"]},
                    {"role": "user", "content": prompt},
                ],
                **self._prepare_chat_completions_create_args(),
            )
            annotation_str = response.choices[0].message.content
            logger.info(annotation_str)

            return {
                "prompt": prompt,
                "annotations": self.parse_annotations(text=text, annotations_json=annotation_str),
            }
        except Exception as e:
            traceback.print_exc()
            logger.error(e)
            raise e


class VLLMMetric(OpenAIClientMetric):
    def __init__(self, config, **kwargs):
        base_url = config["api_url"]  # Mandatory for VLLM
        api_key = config.get("api_key", None)  # Optional authentication for VLLM

        super().__init__(config, base_url=base_url, api_key=api_key, **kwargs)

    def _prepare_chat_completions_create_args(self):
        guided_json = self.schema
        # # works well with vllm https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters
        config_args = {"extra_body": {"guided_json": guided_json}}
        return config_args


class OpenAIMetric(OpenAIClientMetric):
    def _prepare_chat_completions_create_args(self):
        model = self.config["model"]

        model_supported = any(model.startswith(prefix) for prefix in ["gpt-4o", "gpt-4o-mini"])
        if not model_supported:
            logger.warning(
                f"Model {model} does not support structured output. It is probablye there will be SOME OF PARSING ERRORS"
            )
            response_format = {"type": "json_object"}
        else:
            # Details at https://platform.openai.com/docs/guides/structured-outputs?context=without_parse
            json_schema = dict(name="OutputNLGAnnotations", strict=True, schema=self.schema)
            response_format = {
                "type": "json_schema",
                "json_schema": json_schema,
            }

        config_args = self.config.get("model_args", {})
        if "response_format" in config_args and config_args["response_format"] != response_format:
            logger.warning(f"Not using the default {response_format=} but using {config_args['response_format']=}")
        else:
            config_args["response_format"] = response_format
        return config_args


class OllamaMetric(LLMMetric):
    def __init__(self, config):
        super().__init__(config)

        self.set_api_endpoint()

    @property
    def new_connection_error_advice_docstring(self):
        return """\
Please check the Ollama documentation:
    https://github.com/ollama/ollama?tab=readme-ov-file#generate-a-response
"""

    def set_api_endpoint(self):
        # make sure the API URL ends with the `generate` endpoint
        self.config["api_url"] = self.config["api_url"].rstrip("/")
        if not self.config["api_url"].endswith("/generate"):
            self.config["api_url"] += "/generate/"

    def postprocess_output(self, output):
        output = output.strip()
        j = json.loads(output)

        ANNOTATION_STR = "annotations"
        assert (
            ANNOTATION_STR in OutputAnnotations.model_json_schema()["properties"]
        ), f"Has the {OutputAnnotations=} schema changed?"

        # Required for OllamaMetric. You may want to switch to VLLMMetric which uses constrained decoding.
        # It is especially useful for weaker models which have problems decoding valid JSON on output.
        if self.config["model"].startswith("llama3"):
            # the model often tends to produce a nested list

            annotations = j[ANNOTATION_STR]
            if isinstance(annotations, list) and len(annotations) >= 1 and isinstance(annotations[0], list):
                j[ANNOTATION_STR] = j[ANNOTATION_STR][0]

        return json.dumps(j)

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
            annotation_postprocessed = self.postprocess_output(annotation_str)
            logger.info(annotation_postprocessed)
            return {
                "prompt": prompt,
                "annotations": self.parse_annotations(text=text, annotations_json=annotation_postprocessed),
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

    @property
    def new_connection_error_advice_docstring(self):
        return """\
Please check the Ollama documentation:
    https://github.com/ollama/ollama?tab=readme-ov-file#generate-a-response
"""

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
