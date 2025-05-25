#!/usr/bin/env python3

import abc
import json
import logging
import re
import time
import traceback

from pydantic import ValidationError

from factgenie.annotations import AnnotationModelFactory
from factgenie.api import ModelAPI
from factgenie.text_processing import template_replace

logger = logging.getLogger("factgenie")


class PromptingStrategy(abc.ABC):
    def __init__(self, config: dict):
        self.config = config
        self.extra_args = config.get("extra_args", {})
        self.prompt_strat_kwargs = {}

    def prompt(self, data, to_annotate: str = "{text}"):
        data = self.preprocess_data_for_prompt(data)
        keyword_dict = {"data": data, "text": to_annotate}

        prompt_template = self.config["prompt_template"]
        prompt_template = template_replace(prompt_template, keyword_dict)

        return prompt_template

    def construct_message(self, prompt):
        messages = []

        if self.config.get("system_msg"):
            messages.append({"role": "system", "content": self.config["system_msg"]})

        messages.append({"role": "user", "content": prompt})

        if self.config.get("start_with"):
            messages.append({"role": "assistant", "content": self.config["start_with"]})

        return messages

    def postprocess_output(self, output):
        # cut model generation at the stopping sequence
        if self.extra_args.get("stopping_sequence", False):
            stopping_sequence = self.extra_args["stopping_sequence"]

            # re-normalize double backslashes ("\\n" -> "\n")
            stopping_sequence = stopping_sequence.encode().decode("unicode_escape")

            if stopping_sequence in output:
                output = output[: output.index(stopping_sequence)]

        output = output.strip()

        # strip the suffix from the output
        if self.extra_args.get("remove_suffix", ""):
            suffix = self.extra_args["remove_suffix"]

            if output.endswith(suffix):
                output = output[: -len(suffix)]

        # remove any multiple spaces
        output = " ".join(output.split())

        output = output.strip()
        return output

    def get_model_response(self, api: ModelAPI, prompt):
        """Get model response with timing and logging."""
        messages = self.construct_message(prompt)

        start = time.time()
        response = api.get_model_response_with_retries(messages, prompt_strat_kwargs=self.prompt_strat_kwargs)
        logger.info(f"Received response in {time.time() - start:.2f} seconds.")

        logger.debug(f"Prompt tokens: {response.usage.prompt_tokens}")
        logger.debug(f"Response tokens: {response.usage.completion_tokens}")

        return response.choices[0].message.content

    def preprocess_data_for_prompt(self, data):
        """Override this method to change the format how the data is presented in the prompt. See self.prompt() method for usage."""
        return data

    @abc.abstractmethod
    def get_model_output(self, api: ModelAPI, data, text=None):
        """
        Abstract method that each subclass must implement to get output from the model.

        Args:
            api: The ModelAPI instance to use for calling the model
            data: The source data to be used in the prompt (if present)
            text: The text to be annotated (annotation tasks only)

        Returns:
            A dictionary with the prompt and either 'output' or 'annotations'
        """
        pass


class GenerationStrategy(PromptingStrategy):
    """Strategy for generating new text content from data."""

    def get_model_output(self, api: ModelAPI, data, text=None):
        """
        Generate output text with the model.

        Args:
            api: The ModelAPI instance
            data: The data to be used in the prompt
            text: Ignored in this strategy

        Returns:
            A dictionary: {
                "prompt": the prompt used for the generation,
                "output": the generated output
            }
        """
        try:
            prompt = self.prompt(data)

            raw_output = self.get_model_response(api, prompt)
            output = self.postprocess_output(raw_output)
            logger.info(output)

            return {"prompt": prompt, "output": output}

        except Exception as e:
            traceback.print_exc()
            logger.error(e)
            raise e


class AnnotationsStrategy(PromptingStrategy):
    """Base strategy for annotation tasks."""

    def __init__(self, config):
        super().__init__(config)

        if self.extra_args.get("with_reason") == False:
            with_reason = False
        else:
            # default
            with_reason = True

        self.output_validation_model = AnnotationModelFactory.get_output_model(with_reason)

    def parse_annotations(self, text: str, annotations_json: str):
        """
        Parse annotations from JSON and validate them.

        Args:
            text: The text to be annotated.
            annotations_json: A JSON string containing the annotations.

        Returns:
            A list of validated annotations.
        """

        annotation_span_categories = self.config["annotation_span_categories"]
        overlap_allowed = self.config.get("annotation_overlap_allowed", False)

        try:
            annotations_obj = self.output_validation_model.model_validate_json(annotations_json)
            annotations = annotations_obj.annotations
        except ValidationError as e:
            logger.exception("Parsing error: ")

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
                annotation_type_str = annotation_span_categories[annotation_d["type"]]["name"]
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


class StructuredOutputStrategy(AnnotationsStrategy):
    """Strategy for generating structured annotations with reasoning."""

    def __init__(self, config):
        super().__init__(config)

        # We force the output format with the `response_format` parameter
        self.prompt_strat_kwargs["response_format"] = self.output_validation_model

    def get_model_output(self, api: ModelAPI, data, text):
        """
        Annotate text with the model.

        Args:
            api: The ModelAPI instance
            data: The data from which the text was generated (optional)
            text: The text to annotate (required)

        Returns:
            A dictionary: {
                "prompt": the prompt used for the generation,
                "annotations": the annotations for the text
            }
        """
        assert isinstance(text, str) and len(text) > 0, f"Text must be a non-empty string, got {text=}"

        try:
            prompt = self.prompt(data, text)
            logger.debug(f"Prompt: {prompt}")

            logger.info("Annotated text:")
            logger.info(f"\033[34m{text}\033[0m")

            annotation_str = self.get_model_response(api, prompt)

            return {
                "prompt": prompt,
                "annotations": self.parse_annotations(text=text, annotations_json=annotation_str),
            }
        except Exception as e:
            traceback.print_exc()
            logger.error(e)
            raise e


class RawOutputStrategy(AnnotationsStrategy):
    """Strategy for generating structured annotations that need to be extracted from raw text."""

    def extract_json_from_raw(self, content):
        """
        Extract JSON object from raw model output, handling potential thinking traces.
        """
        # First, if the response contains <think> tags, log them and remove them
        think_blocks = re.findall(r"<think>(.*?)</think>", content, flags=re.DOTALL)
        for block in think_blocks:
            logger.info(f"\033[90m[THINKING] {block.strip()}\033[0m")  # Grey color

        # Remove all <think>...</think> sections
        output = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

        # Try to find the JSON object by looking for matching curly braces
        potential_jsons = []
        stack = []
        start_indices = []

        for i, char in enumerate(output):
            if char == "{":
                if not stack:  # If this is a new potential JSON object
                    start_indices.append(i)
                stack.append("{")
            elif (char == "}") and stack:
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
        final_json = valid_jsons[-1]

        return {
            "json_str": final_json,
            "thinking_trace": "\n".join(think_blocks).strip(),
        }

    def get_model_output(self, api: ModelAPI, data, text):
        """
        Annotate text with the model using raw JSON extraction.

        Args:
            api: The ModelAPI instance
            data: The data from which the text was generated (optional)
            text: The text to annotate (required)

        Returns:
            A dictionary: {
                "prompt": the prompt used for the generation,
                "annotations": the annotations for the text
                "thinking_trace": the thinking trace (if any)
            }
        """
        assert isinstance(text, str) and len(text) > 0, f"Text must be a non-empty string, got {text=}"

        try:
            prompt = self.prompt(data, text)
            logger.debug(f"Prompt: {prompt}")

            logger.info("Annotated text:")
            logger.info(f"\033[34m{text}\033[0m")

            raw_response = self.get_model_response(api, prompt)

            # Extract JSON from the raw output
            extracted = self.extract_json_from_raw(raw_response)

            ret = {
                "prompt": prompt,
                "annotations": self.parse_annotations(text=text, annotations_json=extracted["json_str"]),
            }
            if extracted.get("thinking_trace"):
                ret["thinking_trace"] = extracted["thinking_trace"]

            return ret
        except Exception as e:
            traceback.print_exc()
            logger.error(e)
            raise e
