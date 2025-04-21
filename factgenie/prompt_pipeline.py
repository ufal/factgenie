#!/usr/bin/env python3

import abc
import copy
import json
import logging
import re
import time
import traceback
from typing import Literal, Type
from litellm.types.utils import StandardBuiltInToolsParams
import yaml
from itertools import chain
from pydantic import BaseModel, ValidationError
from factgenie import annotations
from factgenie.annotations import AnnotationModelFactory
from factgenie.api import ModelAPI
from factgenie.prompting import PromptingStrategy, RawOutputStrategy
from factgenie.text_processing import find_all_template_keys, template_replace

logger = logging.getLogger("factgenie")


# Conventions:
#  - If the transform has an `input_field` argument, it should be the first argument.
#  - If the transform has an `output_field` argument, it should be the second argument.
class Transform:
    @abc.abstractmethod
    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        pass

    @property
    @abc.abstractmethod
    def requires_fields(self) -> list[str]:
        pass

    @property
    @abc.abstractmethod
    def outputs_fields(self) -> list[str]:
        pass

    @property
    @abc.abstractmethod
    def clears_other_fields(self) -> bool:
        pass


def derive_field(current: list[dict], api: ModelAPI, function, output_field: str):
    return [{**c, output_field: function(c, api)} for c in current]


class DeriveField(Transform):
    def __init__(self, function, output_field: str, expects: list[str] = []):
        """
        Args:
            function: A function taking a dictionary and returning any object.
            output_field: An output field to save the output as in each dictionary.
        """
        self.function = function
        self.output_field = output_field
        self.expects = expects

    @property
    def requires_fields(self) -> list[str]:
        return self.expects

    @property
    def outputs_fields(self) -> list[str]:
        return [self.output_field]

    @property
    def clears_other_fields(self) -> bool:
        return False

    def __call__(self, current: list[dict], api: ModelAPI):
        return derive_field(current, api, self.function, self.output_field)


CopyType = Literal["reference", "shallow", "deep"]

class Duplicate(Transform):
    def __init__(self, input_field: str, output_field: str, copy_type: CopyType = "reference"):
        """
        Args:
            copy_type: "reference" (default) should be enough, unless you expect in-place modifications to either field. In those cases, use "shallow" or "deep".
        """
        self.output_field = output_field
        self.input_field = input_field
        self.copy_type = copy_type

    @property
    def requires_fields(self) -> list[str]:
        return [self.input_field]

    @property
    def outputs_fields(self) -> list[str]:
        return [self.output_field]

    @property
    def clears_other_fields(self) -> bool:
        return False

    def duplicate(self, c: dict, api: ModelAPI):
        if self.copy_type == "reference":
            return c[self.input_field]
        elif self.copy_type == "shallow":
            return copy.copy(c[self.input_field])
        else: # deep
            return copy.deepcopy(c[self.input_field])
 
    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        return derive_field(current, api, self.duplicate, self.output_field)


class Filter(Transform):
    def __init__(self, input_fields: list[str], condition):
        """
        Args:
            input_fields: A list of input fields passed as parameters to the condition.
            condition: A function like `condition(input_1, input_2, ...) -> bool`

        Returns only the elements that pass the conditions.
        """
        self.input_fields = input_fields
        self.condition = condition

    @property
    def requires_fields(self) -> list[str]:
        return self.input_fields

    @property
    def outputs_fields(self) -> list[str]:
        return []

    @property
    def clears_other_fields(self) -> bool:
        return False

    def passes(self, c: dict, api: ModelAPI):
        function_inputs = [c[field] for field in self.input_fields]
        return self.condition(*function_inputs)
 
    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        return [c for c in current if self.passes(c, api)]


class SentenceSplit(Transform):
    def __init__(self, output_field: str):
        self.output_field = output_field

    @property
    def requires_fields(self) -> list[str]:
        return ["text"]

    @property
    def outputs_fields(self) -> list[str]:
        return [self.output_field]

    @property
    def clears_other_fields(self) -> bool:
        return False

    # I also tried this text splitter (https://github.com/mediacloud/sentence-splitter). It can properly recognize sentences. It needs the next sentence to either start with a capital letter or a 4-digit number (year). Unfortunately it has problems with markdown, which is a common output of LLMs.
    @classmethod
    def iter_sentences(cls, text: str):
        # This regex:
        #  - '.' and a negative lookahead
        #    - Can't be followed by another numer, comma, colon, or spaces* lowercase.
        #      This is needed for decimals (3.5) and abbreviations (e.g. this).
        #    - Can be followed by an optional \".
        #  - '?' or '!' followed by an optional \".
        #  - Extra chunking shouldn't hurt once I show it preceding context.
        punc_regex = "\\.(?![0-9]|,|:|\\s*[a-z])\"?|\\?\"?|!\"?"
        parts = [part for part in re.split(punc_regex, text) if len(part) > 2]
        for part in parts:
            yield part

    def __call__(self, current: list[dict], api: ModelAPI):
        # Go over ever sentence and join each sentence with the rest of the input dictionary.
        return [{**c, self.output_field: part}
                for c in current
                for part in
                self.iter_sentences(c["text"])]


class InterpretCode(Transform):
    def __init__(self, input_field: str, output_field: str):
        self.input_field = input_field
        self.output_field = output_field

    def interpret(self, c: dict, api: ModelAPI):
        code = c[self.input_field]
        output = f"<result of `{code}`"
        return output

    def __call__(self, current: list[dict], api: ModelAPI):
        return derive_field(current, api, self.interpret, self.output_field)


class PostprocessOutput(Transform):
    def __init__(self, input_field, output_field, remove_suffix="", stopping_sequence: str | None = None):
        self.remove_suffix = remove_suffix
        self.input_field = input_field
        self.output_field = output_field
        if type(stopping_sequence) is str:
            # re-normalize double backslashes ("\\n" -> "\n")
            self.stopping_sequence = stopping_sequence.encode().decode("unicode_escape")
        else:
            self.stopping_sequence = None

    @property
    def requires_fields(self) -> list[str]:
        return [self.input_field]

    @property
    def outputs_fields(self) -> list[str]:
        return [self.output_field]

    @property
    def clears_other_fields(self) -> bool:
        return False

    def postprocess(self, c: dict, api: ModelAPI):
        text = c[self.input_field]

        # cut model generation at the stopping sequence
        if self.stopping_sequence is not None:
            if self.stopping_sequence in text:
                text = text[: text.index(self.stopping_sequence)]

        text = text.strip()

        # strip the suffix from the text
        if self.remove_suffix is not None:
            if text.endswith(self.remove_suffix):
                text = text[: -len(self.remove_suffix)]

        # remove any multiple spaces
        text = " ".join(text.split())

        text = text.strip()
        return text

    def __call__(self, current: list[dict], api: ModelAPI):
        return derive_field(current, api, self.postprocess, self.output_field)


class ApplyTemplate(Transform):
    def __init__(self, prompt_template: str, output_field: str):
        self.prompt_template = prompt_template 
        self.output_field = output_field
        self.expects = find_all_template_keys(prompt_template)

    @property
    def requires_fields(self) -> list[str]:
        return self.expects

    @property
    def outputs_fields(self) -> list[str]:
        return [self.output_field]

    @property
    def clears_other_fields(self) -> bool:
        return False

    def apply_template(self, c: dict, api: ModelAPI):
        return template_replace(self.prompt_template, c)

    def __call__(self, current: list[dict], api: ModelAPI):
        return derive_field(current, api, self.apply_template, self.output_field)


class AskPrompt(Transform):
    def __init__(self, input_field: str, output_field: str, system_msg: None | str = None, start_with: str | None = None, completion_kwargs: dict = {}):
        """
        Assume we are trying to prompt-map the following data:
        ```
        [
            {
                "text": <the whole annotation>,
                "data": <corresponding data>,
                "part": <first sentence>,
                prompt_name: <the prompt>,
            },
        ]
        ```

        The output will look like this:
        ```
        [
            {
                "text": <the whole annotation>,
                "data": <corresponding data>,
                "part": <first sentence>,
                prompt_name: <the prompt>,
                output_name: <the output of the LLM>,
            },
        ]
        ```

        The `prompt_template` may refer to any input field, such as "{part}" or "{data}" from the example above.
        """
        self.input_field = input_field
        self.output_field = output_field
        self.system_msg = system_msg
        self.start_with = start_with
        self.completion_kwargs = completion_kwargs

    @property
    def requires_fields(self) -> list[str]:
        return [self.input_field]

    @property
    def outputs_fields(self) -> list[str]:
        return [self.output_field]

    @property
    def clears_other_fields(self) -> bool:
        return False

    def construct_message(self, prompt: str):
        messages = []

        if self.system_msg is not None:
            messages.append({"role": "system", "content": self.system_msg})

        messages.append({"role": "user", "content": prompt})

        if self.start_with is not None:
            messages.append({"role": "assistant", "content": self.start_with})

        return messages

    def get_model_response(self, c: dict, api: ModelAPI):
        """Get model response with timing and logging."""
        prompt = c[self.input_field]
        messages = self.construct_message(prompt)

        start = time.time()
        response = api.get_model_response_with_retries(messages, prompt_strat_kwargs=self.completion_kwargs)
        logger.info(f"Received response in {time.time() - start:.2f} seconds.")

        logger.debug(f"Prompt tokens: {response.usage.prompt_tokens}")
        logger.debug(f"Response tokens: {response.usage.completion_tokens}")

        return response.choices[0].message.content

    def __call__(self, current: list[dict], api: ModelAPI):
        return derive_field(current, api, self.get_model_response, self.output_field)


class ParseAnnotations(Transform):
    def __init__(self, input_field: str, output_field: str, annotation_span_categories: list[dict], annotation_overlap_allowed: bool, output_validation_model: Type[BaseModel]):
        self.input_field = input_field
        self.output_field = output_field
        self.annotation_span_categories = annotation_span_categories
        self.annotation_overlap_allowed = annotation_overlap_allowed
        self.output_validation_model = output_validation_model

    @property
    def requires_fields(self) -> list[str]:
        return [self.input_field, "text"]

    @property
    def outputs_fields(self) -> list[str]:
        return [self.output_field]

    @property
    def clears_other_fields(self) -> bool:
        return False

    def parse_annotations(self, c: dict, api: ModelAPI):
        """
        Parse annotations from JSON and validate them.

        Args:
            text: The text to be annotated.
            annotations_json: A JSON string containing the annotations.

        Returns:
            A list of validated annotations.
        """

        annotations_json = c[self.input_field]
        text = c["text"]

        try:
            annotations_obj = self.output_validation_model.model_validate_json(annotations_json)
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

            if not self.annotation_overlap_allowed and start_pos != -1:
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
                annotation_type_str = self.annotation_span_categories[annotation_d["type"]]["name"]
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

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        return derive_field(current, api, self.parse_annotations, self.output_field)


class SequentialStrategy(PromptingStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.transform_sequence = self.get_transform_sequence()

    @abc.abstractmethod
    def get_transform_sequence(self) -> list[Transform]:
        pass

    def verify_sequence(self):
        # TODO: Check that inputs and outputs match
        # How to deal with deleted fields?
        pass

    def get_model_output(self, api: ModelAPI, data, text=None):
        try:
            # Initial condition
            current = [{"data": data}]
            if text is not None:
                current[0]["text"] = text

            for step in self.transform_sequence:
                current = step(current, api)

            # TODO: Assert format (or at least the required fields).
            # Actually do that earlier.
            return current

        except Exception as e:
            traceback.print_exc()
            logger.error(e)
            raise e


class ExtractTag(Transform):
    # I thought about replacing `remove_from_input: bool` with `save_modified_input: str` to allow non-destructive modifications. I decided against it because it doesn't feel as intuitive. The same functionality can still be achieved with Duplicate. 
    def __init__(self, input_field: str, output_field: str | None, tag: str, join_occurances=True, remove_from_input=True, log_as: str | None = None):
        """
        Args:
            tag: The inside of the tag, i.e. "think" for "<think>...</think>" blocks.
            join_occurances: If true, all occurances will be joined by "\n". Otherwise, `output[output_field]` will be a list of strings.
            remove_from_input: If true, all "<{tag}>...</{tag}>" blocks will be removed from the input field.
            log_as: `log_as="THINKING"` will log occurances as gray "[THINKING]" blocks.
        """
        self.input_field = input_field
        self.output_field = output_field
        self.tag = tag
        self.join_occurances = join_occurances
        self.remove_from_input = remove_from_input
        self.log_as = log_as

    @property
    def requires_fields(self) -> list[str]:
        return [self.input_field]

    @property
    def outputs_fields(self) -> list[str]:
        output_fields = []

        if self.output_field is not None: 
            output_fields.append(self.output_field)

        if self.remove_from_input:
            output_fields.append(self.input_field)

        return output_fields

    @property
    def clears_other_fields(self) -> bool:
        return False

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        regex = f"<{self.tag}>(.*?)</{self.tag}>"
        output = []
        for c in current:
            # Log and extract "<{tag}>...</{tag}>" sections.
            input = c[self.input_field]
            tag_blocks = re.findall(regex, input, flags=re.DOTALL)

            if self.log_as is not None:
                for block in tag_blocks:
                    logger.info(f"\033[90m[{self.log_as}] {block.strip()}\033[0m")  # Grey color

            # Join tag occurances if requested.
            if self.join_occurances:
                tag_output = "\n".join(tag_blocks).strip()
            else:
                tag_output = tag_blocks

            # Create the updated record and append it to output.
            c_changes = {}

            if self.remove_from_input:
                modified_input = re.sub(regex, "", input, flags=re.DOTALL)
                c_changes[self.input_field] = modified_input

            if self.output_field is not None:
                c_changes[self.output_field] = tag_output

            output.append({**c, **c_changes})

        return output


class ExtractJson(Transform):
    def __init__(self, input_field: str, output_field: str):
        self.input_field = input_field
        self.output_field = output_field

    @property
    def requires_fields(self) -> list[str]:
        return [self.input_field]

    @property
    def outputs_fields(self) -> list[str]:
        return [self.output_field]

    @property
    def clears_other_fields(self) -> bool:
        return False

    def extract_json(self, c: dict, api: ModelAPI):
        text = c[self.input_field]

        # Try to find the JSON object by looking for matching curly braces
        potential_jsons = []
        stack = []
        start_indices = []

        for i, char in enumerate(text):
            if char == "{":
                if not stack:  # If this is a new potential JSON object
                    start_indices.append(i)
                stack.append("{")
            elif (char == "}") and stack:
                stack.pop()
                if not stack:  # If we've found a complete balanced JSON
                    start = start_indices.pop()
                    potential_jsons.append(text[start : i + 1])

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

        return final_json

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        return derive_field(current, api, self.extract_json, self.output_field)


class CoderStrategy(SequentialStrategy):
    PART = "part"
    GATHER_PROMPT = "gather_prompt"
    GATHER_RESPONSE =  "gather_response"
    CODE = "code"
    CODE_OUTPUT = "code_result"
    ANNOTATION_PROMPT = "annotation_prompt"
    ANNOTATION_TEXT = "annotation_text"
    ANNOTATION_RESPONSE = "annotation_response"
    ANNOTATIONS = "annotations"
    EXTRACTED = "extracted"

    def get_transform_sequence(self) -> list[Transform]:
        template_gather = "Hello {data[raw]} and {part}"
        template_annotate = "Annotating {part} using {code_result}"
        # TODO: Postprocess output? If args?
        # I could make the Modules return args they need. Possibly they could even be sub-dictionaries.
        annotation_span_categories = self.config["annotation_span_categories"]
        annotation_overlap_allowed = self.config.get("annotation_overlap_allowed", False)
        with_reason = self.extra_args.get("with_reason", True)
        output_validation_model = AnnotationModelFactory.get_output_model(with_reason)

        ignore_keywords = self.extra_args.get("ignore_keywords", None)
        # Returns true if the response doesn't match one of the ignore phrases.
        # 
        def filter_irrelevant(response: str):
            if ignore_keywords is None:
                return True
            else:
                # Either of the ignore keywords with up to 7 paddings on either side (e.g. for a ".")
                regex = f"^.{0,7}(?:{'|'.join(ignore_keywords)}).{0,7}$"
                return not re.match(regex, response, re.DOTALL)
        
        return [
            # 1. Split sentences
            SentenceSplit(self.PART),

            # 2. Ask for a code
            ApplyTemplate(template_gather, self.GATHER_PROMPT),
            AskPrompt(self.GATHER_PROMPT, self.GATHER_RESPONSE),
            Filter([self.GATHER_PROMPT], filter_irrelevant),
            # TODO: Filter(lambda x: x not in ["...", "..."])?

            # 3. Intepret code
            ExtractTag(self.GATHER_RESPONSE, self.CODE, tag="code", join_occurances=False, remove_from_input=False, log_as="CODE"),
            InterpretCode(self.CODE, self.CODE_OUTPUT),

            # 4. Ask to annotate code
            ApplyTemplate(template_annotate, self.ANNOTATION_PROMPT),
            AskPrompt(self.ANNOTATION_PROMPT, self.ANNOTATION_RESPONSE),

            ExtractTag(self.ANNOTATION_RESPONSE, "thinking_trace", tag="think", join_occurances=True, remove_from_input=True, log_as="THINKING"),
            ExtractJson(self.ANNOTATION_RESPONSE, self.EXTRACTED),
            ParseAnnotations(self.EXTRACTED, self.ANNOTATIONS, annotation_span_categories, annotation_overlap_allowed, output_validation_model),
            # TODO: ParseAnnotations(self.ANNOTATION_RESPONSE, self.ANNOTATIONS)

            # 5. Join answers
            # ...
        ]


class GenerationStrategy(SequentialStrategy):
    def get_transform_sequence(self) -> list[Transform]:
        system_msg = self.config.get("system_msg", None)
        starts_with = self.config.get("start_with", None)

        remove_suffix = self.extra_args.get("remove_suffix", None)
        stopping_sequence = self.extra_args.get("stopping_sequence", None)

        return [
            ApplyTemplate(self.config["prompt_template"], "prompt"),
            AskPrompt("prompt", "output", system_msg, starts_with),
            PostprocessOutput("output", "output", remove_suffix, stopping_sequence)
            # TODO: Log("output")? or just always log it? Or in pieces?
        ]


class StructuredAnnotationStrategy(SequentialStrategy):
    def get_transform_sequence(self) -> list[Transform]:
        system_msg = self.config.get("system_msg", None)
        starts_with = self.config.get("start_with", None)

        annotation_span_categories = self.config["annotation_span_categories"]
        annotation_overlap_allowed = self.config.get("annotation_overlap_allowed", False)
        with_reason = self.extra_args.get("with_reason", True)
        output_validation_model = AnnotationModelFactory.get_output_model(with_reason)

        return [
            ApplyTemplate(self.config["prompt_template"], "prompt"),
            AskPrompt("prompt", "annotations_raw", system_msg, starts_with),
            ParseAnnotations("annotations_raw", "annotations", annotation_span_categories, annotation_overlap_allowed, output_validation_model),
        ]


class RawOutputAnnotationStrategy(SequentialStrategy):
    def get_transform_sequence(self) -> list[Transform]:
        system_msg = self.config.get("system_msg", None)
        starts_with = self.config.get("start_with", None)

        annotation_span_categories = self.config["annotation_span_categories"]
        annotation_overlap_allowed = self.config.get("annotation_overlap_allowed", False)
        with_reason = self.extra_args.get("with_reason", True)
        output_validation_model = AnnotationModelFactory.get_output_model(with_reason)

        return [
            ApplyTemplate(self.config["prompt_template"], "prompt"),
            AskPrompt("prompt", "annotations_raw", system_msg, starts_with),
            ExtractTag("annotations_raw", "thinking_trace", tag="think", join_occurances=True, remove_from_input=True, log_as="THINKING"),
            ExtractJson("annotations_raw", "extracted"),
            ParseAnnotations("extracted", "annotations", annotation_span_categories, annotation_overlap_allowed, output_validation_model),
        ]


def idk(data, text: str, api: ModelAPI):
    gen = GenerationStrategy({})
    ann = RawOutputStrategy({})
    promptingProxy = PromptingProxy(gen, api)
    parse_raw = True

    template_gather = "Hello {data[raw]} and {part}"
    template_annotate = "Annotating {part} using {code_result}"

    steps = []
    steps.append(SentenceSplit(self.PART))
    steps.append(ApplyTemplate(template_gather, "gather_prompt"))
    steps.append(AskPrompt(api, "gather_prompt", "gather_response"))
    steps.append(InterpretCode("code", "code_result"))
    steps.append(ApplyTemplate(template_annotate, "annotation_text"))
    if parse_raw:
        steps.append(DeriveField(lambda c: ann.extract_json_from_raw(c["annotaiton_text"])["json_str"], "annotation_text"))
    steps.append(DeriveField(lambda c: ann.parse_annotations(c['text'], c['annotation_text']), "annotations"))

    current = {"data": data, "text": text}
