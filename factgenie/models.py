#!/usr/bin/env python3

from abc import abstractmethod
import traceback

import os
import logging
from pydantic import BaseModel, Field, ValidationError
import json
import re
import random
import time
from ast import literal_eval
from factgenie.campaign import CampaignMode

# LiteLLM seems to be triggering deprecation warnings in Pydantic, so we suppress them
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import litellm

# also disable info logs from litellm
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Proxy").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)

# disable requests logging
logging.getLogger("httpx").setLevel(logging.ERROR)

logger = logging.getLogger("factgenie")

DIR_PATH = os.path.dirname(__file__)
LLM_ANNOTATION_DIR = os.path.join(DIR_PATH, "annotations")
LLM_GENERATION_DIR = os.path.join(DIR_PATH, "outputs")


class ModelFactory:
    """Register any new model here."""

    @staticmethod
    def model_classes():
        return {
            CampaignMode.LLM_EVAL: {
                "openai": OpenAIMetric,
                "ollama": OllamaMetric,
                "ollama_reasoning": OllamaReasoningMetric,
                "vllm": VLLMMetric,
                "anthropic": AnthropicMetric,
                "gemini": GeminiMetric,
                "vertexai": VertexAIMetric,
                "vertexai_reasoning": VertexAIReasoningMetric,
            },
            CampaignMode.LLM_GEN: {
                "openai": OpenAIGen,
                "ollama": OllamaGen,
                "vllm": VLLMGen,
                "anthropic": AnthropicGen,
                "gemini": GeminiGen,
                "vertexai": VertexAIGen,
            },
        }

    @staticmethod
    def from_config(config, mode):
        metric_type = config["type"]

        # suffixes are not needed
        if metric_type.endswith("_metric"):
            metric_type = metric_type[: -len("_metric")]
        elif metric_type.endswith("_gen"):
            metric_type = metric_type[: -len("_gen")]

        classes = ModelFactory.model_classes()[mode]

        if metric_type not in classes:
            raise ValueError(f"Model type {metric_type} is not implemented.")

        return classes[metric_type](config)


class SpanAnnotation(BaseModel):
    reason: str = Field(description="The reason for the annotation.")
    text: str = Field(description="The text which is annotated.")
    # Do not name it type since it is a reserved keyword in JSON schema
    annotation_type: int = Field(
        description="Index to the list of span annotation types defined for the annotation campaign."
    )


class SpanAnnotationNoReason(BaseModel):
    text: str = Field(description="The text which is annotated.")
    # Do not name it type since it is a reserved keyword in JSON schema
    annotation_type: int = Field(
        description="Index to the list of span annotation types defined for the annotation campaign."
    )


class OutputAnnotations(BaseModel):
    annotations: list[SpanAnnotation] = Field(description="The list of annotations.")


class OutputAnnotationsNoReason(BaseModel):
    annotations: list[SpanAnnotationNoReason] = Field(description="The list of annotations.")


class Model:
    def __init__(self, config):
        self.config = config
        self.parse_model_args()

    def _api_url(self):
        # by default we ignore the API URL
        # override for local services that actually require the API URL (such as Ollama)
        return None

    def _service_prefix(self):
        raise NotImplementedError(
            "Override this method in the subclass to call the appropriate API. See LiteLLM documentation: https://docs.litellm.ai/docs/providers."
        )

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
        }

    def get_optional_fields(self):
        return {
            "system_msg": str,
            "start_with": str,
            "annotation_overlap_allowed": bool,
            "model_args": dict,
            "api_url": str,
            "extra_args": dict,
        }

    def parse_annotations(self, text, annotations_json):
        extra_args = self.config.get("extra_args", {})

        if extra_args.get("no_reason"):
            out_cls = OutputAnnotationsNoReason
        else:
            out_cls = OutputAnnotations

        try:
            annotations_obj = out_cls.model_validate_json(annotations_json)
            annotations = annotations_obj.annotations
        except ValidationError as e:
            logger.error("Parsing error: ", json.loads(e.json())[0]["msg"])
            try:
                logger.error(f"Model response does not follow the schema.")
                parsed_json = json.loads(annotations_json)
                logger.error(f"Model response: {parsed_json}")
            except json.JSONDecodeError:
                logger.error(f"Model response is not a valid JSON.")
                logger.error(f"Model response: {annotations_json}")

            return []

        annotation_list = []

        logger.info(f"Response contains {len(annotations)} annotations.")

        for i, annotation in enumerate(annotations):
            annotated_span = annotation.text.lower().strip()

            if len(text) == 0:
                logger.warning(f"❌ Span EMPTY.")
                continue

            # find the `start` index of the error in the text
            start_pos = text.lower().find(annotated_span)

            overlap_allowed = self.config.get("annotation_overlap_allowed", False)

            if not overlap_allowed and start_pos != -1:
                # check if the annotation overlaps with any other annotation
                for other_annotation in annotation_list:
                    other_start = other_annotation["start"]
                    other_end = other_start + len(other_annotation["text"])

                    if start_pos < other_end and start_pos + len(annotated_span) > other_start:
                        logger.warning(
                            f"❌ Span OVERLAP: {annotated_span} ({start_pos}:{start_pos + len(annotated_span)}) overlaps with {other_annotation['text']} ({other_start}:{other_end})"
                        )
                        continue

            if start_pos == -1:
                logger.warning(f'❌ Span NOT FOUND: "{annotated_span}"')
                continue

            annotation_d = annotation.model_dump()
            # For backward compatibility let's use shorter "type"
            # We do not use the name "type" in JSON schema for error types because it has much broader sense in the schema (e.g. string or integer)
            annotation_d["type"] = annotation.annotation_type
            del annotation_d["annotation_type"]

            # Save the start position of the annotation
            annotation_d["start"] = start_pos

            try:
                annotation_type_str = self.config["annotation_span_categories"][annotation_d["type"]]["name"]
            except:
                logger.error(f"Annotation type {annotation_d['type']} not found in the annotation_span_categories.")
                continue

            if start_pos == 0 and start_pos + len(annotated_span) == 0:
                logger.warning(f"❌ Span EMPTY.")
                continue

            logger.info(
                f'[\033[32m\033[1m{annotation_type_str}\033[0m] "\033[32m{annotation.text}\033[0m" ({start_pos}:{start_pos + len(annotation.text)})'
            )

            annotation_list.append(annotation_d)

        return annotation_list

    def preprocess_data_for_prompt(self, data):
        """Override this method to change the format how the data is presented in the prompt. See self.prompt() method for usage."""
        return data

    def prompt(self, data, text):
        if not isinstance(text, str) or len(text) == 0:
            logger.warning(f"Text must be a non-empty string, got {text=}")
        data_for_prompt = self.preprocess_data_for_prompt(data)

        prompt_template = self.config["prompt_template"]

        return prompt_template.replace("{data}", str(data_for_prompt)).replace("{text}", text)

    def get_model_response(self, prompt, model_service):
        messages = []

        if self.config.get("system_msg"):
            messages.append({"role": "system", "content": self.config["system_msg"]})

        messages.append({"role": "user", "content": prompt})

        response = litellm.completion(
            model=model_service,
            messages=messages,
            response_format=OutputAnnotations,
            api_base=self._api_url(),
            **self.config.get("model_args", {}),
        )

        return response

    def validate_environment(self, model_service):
        response = litellm.validate_environment(model=model_service)

        if not response["keys_in_environment"]:
            raise ValueError(
                f"Required API variables not found for the model {model_service}. Please add the following keys to the system environment or factgenie config: {response['missing_keys']}"
            )

    def annotate_example(self, data, text):
        model = self.config["model"]
        model_service = self._service_prefix() + model

        self.validate_environment(model_service)

        # temporarily disable until this is properly merged: https://github.com/BerriAI/litellm/pull/7832

        # assert litellm.supports_response_schema(
        #     model_service
        # ), f"Model {model_service} does not support the JSON response schema."

        try:
            prompt = self.prompt(data, text)

            logger.debug(f"Prompt: {prompt}")

            logger.info("Annotated text:")
            logger.info(f"\033[34m{text}\033[0m")

            logger.info(f"Waiting for {model_service}.")
            start = time.time()
            response = self.get_model_response(prompt, model_service)
            logger.info(f"Received response in {time.time() - start:.2f} seconds.")

            logger.debug(f"Prompt tokens: {response.usage.prompt_tokens}")
            logger.debug(f"Response tokens: {response.usage.completion_tokens}")

            annotation_str = response.choices[0].message.content

            return {
                "prompt": prompt,
                "annotations": self.parse_annotations(text=text, annotations_json=annotation_str),
            }
        except Exception as e:
            traceback.print_exc()
            logger.error(e)
            raise e


class OpenAIMetric(LLMMetric):
    # https://docs.litellm.ai/docs/providers/openai
    def __init__(self, config, **kwargs):
        super().__init__(config)

    def _service_prefix(self):
        # OpenAI models do not seem to require a prefix
        return ""


class OllamaMetric(LLMMetric):
    # https://docs.litellm.ai/docs/providers/ollama
    def __init__(self, config):
        super().__init__(config)

    def validate_environment(self, model_service):
        # Ollama would require setting OLLAMA_API_BASE, but we set the API URL in the config
        pass

    def _service_prefix(self):
        # we want to call the `chat` endpoint: https://docs.litellm.ai/docs/providers/ollama#using-ollama-apichat
        return "ollama_chat/"

    def _api_url(self):
        # local server URL
        api_url = self.config.get("api_url", None)
        api_url = api_url.rstrip("/")

        if api_url.endswith("/generate") or api_url.endswith("/chat") or api_url.endswith("/api"):
            raise ValueError(f"The API URL {api_url} is not valid. Use only the base URL, e.g. http://localhost:11434.")

        return api_url


class ReasoningMetric(LLMMetric):
    """
    Specialized Ollama metric for CoT and reasoning models that use <think> tags for thinking traces.
    This class handles outputs where the model produces content with format:
    <think> ... thinking trace ... </think> JSON output
    """

    def __init__(self, config):
        super().__init__(config)

    def get_model_response(self, prompt, model_service):
        """
        Override to get unstructured response without Pydantic schema enforcement
        """
        messages = []

        if self.config.get("system_msg"):
            messages.append({"role": "system", "content": self.config["system_msg"]})

        messages.append({"role": "user", "content": prompt})

        # Use regular completion without response_format
        response = litellm.completion(
            model=model_service,
            messages=messages,
            api_base=self._api_url(),
            **self.config.get("model_args", {}),
        )

        return response

    def annotate_example(self, data, text):
        model = self.config["model"]
        model_service = self._service_prefix() + model

        self.validate_environment(model_service)

        try:
            prompt = self.prompt(data, text)

            logger.debug(f"Prompt: {prompt}")

            logger.info("Annotated text:")
            logger.info(f"\033[34m{text}\033[0m")

            logger.info(f"Waiting for {model_service}.")
            start = time.time()
            response = self.get_model_response(prompt, model_service)
            logger.info(f"Received response in {time.time() - start:.2f} seconds.")

            logger.debug(f"Prompt tokens: {response.usage.prompt_tokens}")
            logger.debug(f"Response tokens: {response.usage.completion_tokens}")

            raw_response = response.choices[0].message.content

            # Extract content after thinking trace
            annotation_str = self._extract_final_output(raw_response)
            logger.debug(f"Extracted output: {annotation_str}")

            return {
                "prompt": prompt,
                "annotations": self.parse_annotations(text=text, annotations_json=annotation_str),
            }
        except Exception as e:
            traceback.print_exc()
            logger.error(e)
            raise e

    def _extract_final_output(self, content):
        """
        Extract final JSON output from reasoning model response by:
        1. Removing <think>...</think> blocks
        2. Finding and extracting the JSON object

        If multiple <think> blocks exist, handles them all.
        """
        import re

        think_blocks = re.findall(r"<think>(.*?)</think>", content, flags=re.DOTALL)
        for block in think_blocks:
            logger.info(f"\033[90m[THINKING] {block.strip()}\033[0m")  # Grey color

        # Remove all <think>...</think> sections

        output = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

        # Try to find the JSON object by looking for matching curly braces
        # Find all potential JSON objects in the output by tracking balanced braces
        potential_jsons = []
        stack = []
        start_indices = []

        for i, char in enumerate(output):
            if char == "{":
                if not stack:  # If this is a new potential JSON object
                    start_indices.append(i)
                stack.append("{")
            elif char == "}" and stack:
                stack.pop()
                if not stack:  # If we've found a complete balanced JSON
                    start = start_indices.pop()
                    potential_jsons.append(output[start : i + 1])

        # Try to validate each potential JSON, prioritizing the last valid one
        valid_jsons = []
        for json_str in potential_jsons:
            try:
                json.loads(json_str)  # Validate JSON
                valid_jsons.append(json_str)
            except json.JSONDecodeError:
                continue

        if not valid_jsons:
            logger.error("Failed to find valid JSON object in response.")
            return ""

        # Return the last valid JSON (most likely to be the final output)
        return valid_jsons[-1]

    def parse_annotations(self, text, annotations_json):
        """
        Override parse_annotations to handle potentially invalid JSON
        """
        try:
            # First attempt to parse as JSON
            json_obj = json.loads(annotations_json)

            # Create proper structure for Pydantic validation
            if isinstance(json_obj, dict) and "annotations" in json_obj:
                annotations_json = json.dumps(json_obj)
            else:
                # If the JSON doesn't have the expected structure, wrap it
                annotations_json = json.dumps({"annotations": json_obj if isinstance(json_obj, list) else []})

            # Now use the parent class method to validate and process
            return super().parse_annotations(text, annotations_json)

        except json.JSONDecodeError:
            logger.error(f"Failed to parse response as JSON: {annotations_json}")
            return []
        except Exception as e:
            logger.error(f"Error parsing annotations: {str(e)}")
            return []


class OllamaReasoningMetric(OllamaMetric, ReasoningMetric):
    """
    Specialized Ollama metric for CoT and reasoning models that use <think> tags for thinking traces.
    This class handles outputs where the model produces content with format:
    <think> ... thinking trace ... </think> JSON output
    """

    def __init__(self, config):
        super().__init__(config)

    def annotate_example(self, data, text):
        return ReasoningMetric.annotate_example(self, data, text)

    def _extract_final_output(self, content):
        return ReasoningMetric._extract_final_output(self, content)

    def parse_annotations(self, text, annotations_json):
        return ReasoningMetric.parse_annotations(self, text, annotations_json)


class VLLMMetric(LLMMetric):
    # https://docs.litellm.ai/docs/providers/vllm
    def __init__(self, config):
        super().__init__(config)

    def _service_prefix(self):
        return "hosted_vllm/"

    def _api_url(self):
        # local server URL
        api_url = self.config.get("api_url", None)

        return api_url


class AnthropicMetric(LLMMetric):
    # https://docs.litellm.ai/docs/providers/anthropic
    def __init__(self, config):
        super().__init__(config)

    def _service_prefix(self):
        return "anthropic/"

    def _get_model_response_with_retries(self, prompt, model_service):
        """Handle rate limits and overload errors with exponential backoff and retry logic"""
        max_retries = 15
        initial_retry_delay = 2  # seconds

        messages = []
        if self.config.get("system_msg"):
            messages.append({"role": "system", "content": self.config["system_msg"]})

        messages.append({"role": "user", "content": prompt})

        for attempt in range(max_retries):
            try:
                # Use regular completion
                response = litellm.completion(
                    model=model_service,
                    messages=messages,
                    response_format=OutputAnnotations,
                    api_base=self._api_url(),
                    **self.config.get("model_args", {}),
                )
                return response

            except (litellm.exceptions.RateLimitError, litellm.exceptions.InternalServerError) as e:
                # Check if InternalServerError is specifically an "Overloaded" error
                is_overloaded = isinstance(e, litellm.exceptions.InternalServerError) and "Overloaded" in str(e)

                # Check if we've reached max retries
                if attempt == max_retries - 1:
                    error_type = "Rate limit" if isinstance(e, litellm.exceptions.RateLimitError) else "Server overload"
                    logger.error(f"{error_type} exceeded after {max_retries} attempts. Giving up.")
                    raise e

                # Only retry for rate limits or overloaded errors
                if not (isinstance(e, litellm.exceptions.RateLimitError) or is_overloaded):
                    logger.error(f"Non-retryable InternalServerError: {str(e)}")
                    raise e

                # Calculate exponential backoff with jitter
                retry_delay = initial_retry_delay * (2**attempt) + (random.uniform(0, 1))
                error_type = "Rate limit" if isinstance(e, litellm.exceptions.RateLimitError) else "Server overload"
                logger.warning(
                    f"{error_type} hit. Retrying in {retry_delay:.2f} seconds (attempt {attempt+1}/{max_retries})..."
                )
                time.sleep(retry_delay)

            except Exception as e:
                # For other exceptions, don't retry
                logger.error(f"Error calling Anthropic: {str(e)}")
                raise e

    def get_model_response(self, prompt, model_service):
        """Override to use retry mechanism for rate limits"""
        return self._get_model_response_with_retries(prompt, model_service)


class GeminiMetric(LLMMetric):
    # https://docs.litellm.ai/docs/providers/gemini
    def __init__(self, config):
        super().__init__(config)

    def _service_prefix(self):
        return "gemini/"


class VertexAIMetric(LLMMetric):
    # https://docs.litellm.ai/docs/providers/vertex
    def __init__(self, config):
        super().__init__(config)

        self.load_google_credentials()

    def load_google_credentials(self):
        json_file_path = os.environ.get("VERTEXAI_JSON_FULL_PATH")

        if not json_file_path:
            raise ValueError(
                "Please set VERTEXAI_JSON_FULL_PATH in your environment or in the config. For more details, see https://docs.litellm.ai/docs/providers/vertex"
            )

        # check if file exists
        if not os.path.exists(json_file_path):
            raise ValueError(f"The file {json_file_path} was not found.")

        # Load the JSON file
        with open(json_file_path, "r") as file:
            vertex_credentials = json.load(file)

        # Convert to JSON string
        self.vertex_credentials_json = json.dumps(vertex_credentials)

    def _service_prefix(self):
        return "vertex_ai/"

    def get_model_response(self, prompt, model_service):
        response = litellm.completion(
            model=model_service,
            messages=[
                {"role": "system", "content": self.config["system_msg"]},
                {"role": "user", "content": prompt},
            ],
            response_format=OutputAnnotations,
            api_base=self._api_url(),
            vertex_credentials=self.vertex_credentials_json,
            **self.config.get("model_args", {}),
        )

        return response


class VertexAIReasoningMetric(VertexAIMetric, ReasoningMetric):
    """
    Specialized VertexAI metric for CoT and reasoning models that use <think> tags for thinking traces.
    This class handles outputs where the model produces content with format:
    <think> ... thinking trace ... </think> JSON output
    """

    def __init__(self, config):
        super().__init__(config)

    def annotate_example(self, data, text):
        return ReasoningMetric.annotate_example(self, data, text)

    def _extract_final_output(self, content):
        return ReasoningMetric._extract_final_output(self, content)

    def parse_annotations(self, text, annotations_json):
        return ReasoningMetric.parse_annotations(self, text, annotations_json)

    def _get_model_response_with_retries(self, messages, model_service):
        max_retries = 15
        initial_retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                # Use regular completion without response_format
                response = litellm.completion(
                    model=model_service,
                    messages=messages,
                    vertex_credentials=self.vertex_credentials_json,
                    **self.config.get("model_args", {}),
                )

                return response

            except litellm.exceptions.RateLimitError as e:
                # Check if we've reached max retries
                if attempt == max_retries - 1:
                    logger.error(f"Rate limit exceeded after {max_retries} attempts. Giving up.")
                    raise e

                # Calculate exponential backoff with jitter
                retry_delay = initial_retry_delay * (2**attempt) + (random.uniform(0, 1))
                logger.warning(
                    f"Rate limit hit. Retrying in {retry_delay:.2f} seconds (attempt {attempt+1}/{max_retries})..."
                )
                time.sleep(retry_delay)

            except Exception as e:
                # For other exceptions, don't retry
                logger.error(f"Error calling LLM: {str(e)}")
                raise e

    def get_model_response(self, prompt, model_service):
        """
        Override to get unstructured response without Pydantic schema enforcement
        """
        messages = []

        if self.config.get("system_msg"):
            messages.append({"role": "system", "content": self.config["system_msg"]})

        messages.append({"role": "user", "content": prompt})

        response = self._get_model_response_with_retries(messages, model_service)
        return response


class LLMGen(Model):
    def get_required_fields(self):
        return {
            "type": str,
            "prompt_template": str,
            "model": str,
        }

    def get_optional_fields(self):
        return {
            "model_args": dict,
            "system_msg": str,
            "api_url": str,
            "extra_args": dict,
            "start_with": str,
        }

    def postprocess_output(self, output):
        extra_args = self.config.get("extra_args", {})

        # cut model generation at the stopping sequence
        if extra_args.get("stopping_sequence", False):
            stopping_sequence = extra_args["stopping_sequence"]

            # re-normalize double backslashes ("\\n" -> "\n")
            stopping_sequence = stopping_sequence.encode().decode("unicode_escape")

            if stopping_sequence in output:
                output = output[: output.index(stopping_sequence)]

        output = output.strip()

        # strip the suffix from the output
        if extra_args.get("remove_suffix", ""):
            suffix = extra_args["remove_suffix"]

            if output.endswith(suffix):
                output = output[: -len(suffix)]

        # remove any multiple spaces
        output = " ".join(output.split())

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

    def get_model_response(self, messages, model_service):
        response = litellm.completion(
            model=model_service,
            messages=messages,
            api_base=self._api_url(),
            **self.config.get("model_args", {}),
        )
        return response

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
        model = self.config["model"]
        model_service = self._service_prefix() + model

        try:
            prompt = self.prompt(data)

            messages = []

            if self.config.get("system_msg"):
                messages.append({"role": "system", "content": self.config["system_msg"]})

            messages.append(
                {"role": "user", "content": prompt},
            )

            if self.config.get("start_with"):
                messages.append({"role": "assistant", "content": self.config["start_with"]})

            response = self.get_model_response(messages, model_service)

            output = response.choices[0].message.content
            output = self.postprocess_output(output)
            logger.info(output)

            return {"prompt": prompt, "output": output}

        except Exception as e:
            traceback.print_exc()
            logger.error(e)
            raise e


class OpenAIGen(LLMGen):
    # https://docs.litellm.ai/docs/providers/openai
    def __init__(self, config, **kwargs):
        super().__init__(config)

    def _service_prefix(self):
        # OpenAI models do not seem to require a prefix: https://docs.litellm.ai/docs/providers/openai
        return ""


class OllamaGen(LLMGen):
    # https://docs.litellm.ai/docs/providers/ollama
    def __init__(self, config):
        super().__init__(config)

    def _service_prefix(self):
        # we want to call the `chat` endpoint: https://docs.litellm.ai/docs/providers/ollama#using-ollama-apichat
        return "ollama_chat/"

    def _api_url(self):
        # local server URL
        api_url = self.config.get("api_url", None)
        api_url = api_url.rstrip("/")

        if api_url.endswith("/generate") or api_url.endswith("/chat") or api_url.endswith("/api"):
            raise ValueError(f"The API URL {api_url} is not valid. Use only the base URL, e.g. http://localhost:11434.")

        return api_url


class VLLMGen(LLMGen):
    # https://docs.litellm.ai/docs/providers/vllm
    def __init__(self, config):
        super().__init__(config)

    def _service_prefix(self):
        return "hosted_vllm/"

    def _api_url(self):
        # local server URL
        api_url = self.config.get("api_url", None)

        return api_url


class AnthropicGen(LLMGen):
    # https://docs.litellm.ai/docs/providers/anthropic
    def __init__(self, config):
        super().__init__(config)

    def _service_prefix(self):
        return "anthropic/"


class GeminiGen(LLMGen):
    # https://docs.litellm.ai/docs/providers/gemini
    def __init__(self, config):
        super().__init__(config)

    def _service_prefix(self):
        return "gemini/"


class VertexAIGen(LLMGen):
    # https://docs.litellm.ai/docs/providers/vertex
    def __init__(self, config):
        super().__init__(config)

        self.load_google_credentials()

    def load_google_credentials(self):
        json_file_path = os.environ.get("VERTEXAI_JSON_FULL_PATH")

        if json_file_path:
            if not os.path.exists(json_file_path):
                raise ValueError(
                    "File not found in VERTEXAI_JSON_FULL_PATH. For more details, see https://docs.litellm.ai/docs/providers/vertex"
                )

            # Load the JSON file
            with open(json_file_path, "r") as file:
                vertex_credentials = json.load(file)

            # Convert to JSON string
            self.vertex_credentials_json = json.dumps(vertex_credentials)

    def get_model_response(self, messages, model_service):
        response = litellm.completion(
            model=model_service,
            messages=messages,
            api_base=self._api_url(),
            vertex_credentials=self.vertex_credentials_json,
            **self.config.get("model_args", {}),
        )
        return response

    def _service_prefix(self):
        return "vertex_ai/"
