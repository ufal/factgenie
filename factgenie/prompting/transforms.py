#!/usr/bin/env python3

import abc
import copy
import functools
import json
import logging
import re
import time
import unittest
from typing import Any, Literal, Type

from pydantic import BaseModel, ValidationError

from factgenie.annotations import AnnotationModelFactory
from factgenie.colors import Ansi
from factgenie.prompting.model_apis import MockingAPI, ModelAPI
from factgenie.prompting.text_processing import (
    find_all_template_keys,
    iter_sentences,
    join_outer_lists,
    template_replace,
)

logger = logging.getLogger("factgenie")


# A few useful string constants.
join_string_short = " │ "
join_string_long = "\n\n" + "―" * 80 + "\n\n"


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
    def clears_other_fields(self) -> bool:
        return False

    @property
    def removes_fields(self) -> list[str]:
        return []


def derive_field(current: list[dict], api: ModelAPI, function, output_field: str):
    return [{**c, output_field: function(c, api)} for c in current]


class DeriveField(Transform):
    def __init__(self, input_fields: list[str], output_field: str, function):
        """
        Args:
            input_fields: A list of input fields that the function will need (minus the mandatory api).
            output_field: An output field to save the output as in each dictionary.
            function: A function taking as parameters the listed parameters + ModelAPI. E.g. `input_fields=["a", "b"]`, and `function(a, b, api) -> Any`
        """
        self.input_fields = input_fields
        self.output_field = output_field
        self.function = function

    @property
    def requires_fields(self) -> list[str]:
        return self.input_fields

    @property
    def outputs_fields(self) -> list[str]:
        return [self.output_field]

    def apply_function(self, c: dict, api: ModelAPI):
        params = [c[field] for field in self.input_fields]
        return self.function(*params, api)

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        return derive_field(current, api, self.apply_function, self.output_field)


class Duplicate(Transform):
    CopyType = Literal["reference", "shallow", "deep"]

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

    def duplicate(self, c: dict, api: ModelAPI):
        if self.copy_type == "reference":
            return c[self.input_field]
        elif self.copy_type == "shallow":
            return copy.copy(c[self.input_field])
        else:  # deep
            return copy.deepcopy(c[self.input_field])

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        return derive_field(current, api, self.duplicate, self.output_field)


class Filter(Transform):
    def __init__(self, input_fields: list[str], condition):
        """
        Args:
            input_fields: A list of input fields passed as parameters to the condition.
            condition: A function like `condition(input_1, input_2, ..., api: ModelAPI) -> bool`

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

    def passes(self, c: dict, api: ModelAPI):
        function_inputs = [c[field] for field in self.input_fields]
        return self.condition(*function_inputs, api)

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        return [c for c in current if self.passes(c, api)]


class Log(Transform):
    LogLevel = Literal["info", "debug", "none"]

    def __init__(
        self,
        text: str = "",
        field: str | None = None,
        log_level: LogLevel = "info",
        color: str | None = None,
        join_by: str = join_string_short,
    ):
        """
        Args:
            input_fields: A list of input fields passed as parameters to the condition.
            condition: A function like `condition(input_1, input_2, ..., api: ModelAPI) -> bool`

        Returns only the elements that pass the conditions.
        """
        self.text = text
        self.field = field
        self.log_level = log_level
        self.join_by = join_by

        # Default color is blue (when log_level is "info")
        if color is not None:
            self.color = color
        elif log_level == "info":
            self.color = Ansi.BLUE
        else:
            self.color = ""

    @property
    def requires_fields(self) -> list[str]:
        return [self.field] if self.field is not None else []

    @property
    def outputs_fields(self) -> list[str]:
        return []

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        if self.field is not None:
            values = [f"{self.color}{c[self.field]}{Ansi.RESET}" for c in current]
        else:
            values = []
        text = self.text + self.join_by.join(values)

        if self.log_level == "info":
            logger.info(text)
        elif self.log_level == "debug":
            logger.debug(text)

        # No changes.
        return current


class LogAllThrowException(Exception):
    pass


class LogAllThrow(Transform):
    def __init__(self, type_only: bool = False, join_by: str = "\n\n" + "―" * 80 + "\n\n"):
        """
        Args:
            input_fields: A list of input fields passed as parameters to the condition.
            condition: A function like `condition(input_1, input_2, ..., api: ModelAPI) -> bool`

        Returns only the elements that pass the conditions.
        """
        self.type_only = type_only
        self.join_by = join_by

    @property
    def requires_fields(self) -> list[str]:
        return []

    @property
    def outputs_fields(self) -> list[str]:
        # Since this function throws an error and is meant for debugging, it can lie about it's outputs to pass the sequential inspection.
        return ["output", "annotations"]

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        logger.info(f"{Ansi.YELLOW}―――――――――――――――――――― LogAllThrow ――――――――――――――――――――{Ansi.RESET}")
        for field in current[0].keys():
            value = current[0][field]
            value_type = f"{Ansi.RED}{type(value)} * {len(current)}{Ansi.RESET}"
            text = f"{Ansi.LIGHT_GREEN}{field}{Ansi.RESET}: " + value_type
            if not self.type_only:
                values = [f"{Ansi.BLUE}{c[field]}{Ansi.RESET}" for c in current]
                text += " = " + self.join_by.join(values) + "\n"

            logger.info(text)

        raise LogAllThrowException("A planned logging exception.")


class Metadata(Transform):
    def __init__(self, fields: list[str]):
        """
        Args:
            input_fields: A list of input fields passed as parameters to the condition.
            condition: A function like `condition(input_1, input_2, ..., api: ModelAPI) -> bool`

        Returns only the elements that pass the conditions.
        """
        self.fields = fields

    # Maybe later create a class 'PassThroughTransform(Transform)' which sets these 3 properties. (Or 'NoTransform' ?)
    @property
    def requires_fields(self) -> list[str]:
        return self.fields

    @property
    def outputs_fields(self) -> list[str]:
        return ["metadata"]

    def with_metadata(self, c: dict):
        previous_metadata = c.get("metadata", {})
        new_metadata = {field: c[field] for field in self.fields}
        return {**c, "metadata": previous_metadata | new_metadata}

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        return [self.with_metadata(c) for c in current]


class SentenceSplit(Transform):
    def __init__(self, input_field: str, output_field: str):
        self.output_field = output_field
        self.input_field = input_field

    @property
    def requires_fields(self) -> list[str]:
        return [self.input_field]

    @property
    def outputs_fields(self) -> list[str]:
        return [self.output_field]

    # I also tried this text splitter (https://github.com/mediacloud/sentence-splitter). It can properly recognize sentences. It needs the next sentence to either start with a capital letter or a 4-digit number (year). Unfortunately it has problems with markdown, which is a common output of LLMs.
    @classmethod
    def iter_sentences_old(cls, text: str):
        # This regex:
        #  - '.' and a negative lookahead
        #    - Can't be followed by another numer, comma, colon, or spaces* lowercase.
        #      This is needed for decimals (3.5) and abbreviations (e.g. this).
        #    - Can be followed by an optional \".
        #  - '?' or '!' followed by an optional \".
        #  - Extra chunking shouldn't hurt once I show it preceding context.
        punc_regex = '\\.(?![0-9]|,|:|\\s*[a-z])"?|\\?"?|!"?'
        parts = [part for part in re.split(punc_regex, text) if len(part) > 2]
        for part in parts:
            yield part

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        # Go over ever sentence and join each sentence with the rest of the input dictionary.
        return [{**c, self.output_field: part} for c in current for part in iter_sentences(c[self.input_field])]


class Unify(Transform):
    def __init__(
        self, annotation_fields: list[str], join_strings_by: str = join_string_short, ignore_fields: list[str] = []
    ):
        assert (
            len(set(annotation_fields) & set(ignore_fields)) == 0
        ), "You are not allowed to specify the same field name in `annotation_fields` and `ignore_fields` at the same time."

        self.annotation_fields = annotation_fields
        self.join_strings_by = join_strings_by
        self.ignore_fields = ignore_fields

    @property
    def requires_fields(self) -> list[str]:
        return self.annotation_fields

    @property
    def outputs_fields(self) -> list[str]:
        return []

    @property
    def removes_fields(self) -> list[str]:
        return self.ignore_fields

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        if len(current) == 1:
            return current

        res = {}
        for key in current[0].keys():
            if key in self.ignore_fields:
                continue

            # Collect the same entry from all lists of dictionaries.
            items = [c[key] for c in current]

            # All items being equal --> no unification necessary.
            if all(map(lambda x: items[0] == x, items[1:])):
                res[key] = items[0]
            # Handle annotation fields.
            elif key in self.annotation_fields:
                if not isinstance(items[0], str):
                    raise TypeError(
                        f"Field '{key}' is marked as an annotation_field in the `Unify` transform, but its type is not string. (The type is `{type(items[0])}`.)"
                    )

                res[key] = join_outer_lists(items, json_header="annotations")
            # Rest of strings join by `join_strings_by`.
            elif isinstance(items[0], str):
                res[key] = self.join_strings_by.join(items)
            # Everything else is an error.
            else:
                raise NotImplementedError(
                    f"The `Unify` transform received the field `{key}` of type `{type(items[0])}` which it doesn't know how to handle. To prevent this error, you might want to add this field to `Unify`'s `ignore_fields`."
                )

        # On the output, we get a list containing a single dictionary again.
        return [res]


class PostprocessOutput(Transform):
    # `stopping_sequence` gets applied first so it precedes the `remove_suffix` argument.
    def __init__(self, input_field, output_field, stopping_sequence: str | None = None, remove_suffix=""):
        self.input_field = input_field
        self.output_field = output_field
        self.remove_suffix = remove_suffix
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

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
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

    def apply_template(self, c: dict, api: ModelAPI):
        return template_replace(self.prompt_template, c)

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        return derive_field(current, api, self.apply_template, self.output_field)


class AskPrompt(Transform):
    def __init__(
        self,
        input_field: str,
        output_field: str,
        system_msg: None | str = None,
        start_with: str | None = None,
        completion_kwargs: dict = {},
    ):
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

        content = response.choices[0].message.content
        return content

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        return derive_field(current, api, self.get_model_response, self.output_field)


class ParseAnnotations(Transform):
    def __init__(
        self,
        input_field: str,
        output_field: str,
        annotation_span_categories: list[dict],
        annotation_overlap_allowed: bool,
        output_validation_model: Type[BaseModel],
    ):
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


class ExtractTag(Transform):
    # I thought about replacing `remove_from_input: bool` with `save_modified_input: str` to allow non-destructive modifications. I decided against it because it doesn't feel as intuitive. The same functionality can still be achieved with Duplicate.
    def __init__(
        self,
        input_field: str,
        output_field: str | None,
        tag: str,
        join_occurances=True,
        remove_from_input=True,
        log_as: str | None = None,
    ):
        """
        Args:
            tag: The inside of the tag, i.e. "think" for "<think>...</think>" blocks.
            join_occurances: If true, all occurances will be joined by "\n". Otherwise, `output[output_field]` will be a list of strings.
            remove_from_input: If true, all "<{tag}>...</{tag}>" blocks will be removed from the input field.
            log_as: `log_as="THINKING"` will log occurances as gray "[THINKING]" blocks.

        Each block as well as the modified (if `remove_from_input=True`) is stripped of the padding spaces.
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

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        regex = f"\\s*<{self.tag}>(.*?)</{self.tag}>\\s*"
        output = []
        for c in current:
            # Log and extract "<{tag}>...</{tag}>" sections, stripped.
            input = c[self.input_field]
            tag_blocks = re.findall(regex, input, flags=re.DOTALL)
            tag_blocks = [block.strip() for block in tag_blocks]

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
                modified_input = re.sub(regex, " ", input, flags=re.DOTALL).strip()
                c_changes[self.input_field] = modified_input

            if self.output_field is not None:
                c_changes[self.output_field] = tag_output

            output.append(c | c_changes)

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


# ――――――――――――――――――――――――――――――――――― TESTS ―――――――――――――――――――――――――――――――――――


class TransformTests(unittest.TestCase):
    def __init__(self, *vargs, **kwargs):
        super().__init__(*vargs, **kwargs)
        self.api = MockingAPI()

    def test_duplicate(self):
        current = [{"a": "a"}, {"a": "a"}]
        transform = Duplicate("a", "b")

        expected = [{"a": "a", "b": "a"}, {"a": "a", "b": "a"}]
        result = transform(current, self.api)

        self.assertListEqual(expected, result)

    def test_metadata(self):
        current = [{"a": "a", "b": "b"}, {"a": "a", "b": "b"}]
        transform = Metadata(fields=["a"])

        expected = [{"a": "a", "b": "b", "metadata": {"a": "a"}}, {"a": "a", "b": "b", "metadata": {"a": "a"}}]
        result = transform(current, self.api)

        self.assertListEqual(expected, result)

    def test_derive(self):
        current = [{"a": "a"}, {"a": "a"}]

        def fn(a, api: ModelAPI):
            self.assertEqual(a, "a")
            return "b"

        transform = DeriveField(["a"], "b", fn)

        expected = [{"a": "a", "b": "b"}, {"a": "a", "b": "b"}]
        result = transform(current, self.api)

        self.assertListEqual(expected, result)

    def test_filter(self):
        current = [{"a": "yes"}, {"a": "no"}]

        def fn(a, api: ModelAPI):
            return a == "yes"

        transform = Filter(["a"], fn)

        expected = [{"a": "yes"}]
        result = transform(current, self.api)

        self.assertListEqual(expected, result)

    def test_sentence_split(self):
        current = [{"a": "Sentence 1. Sentence 2."}, {"a": "Sentence 3."}]
        transform = SentenceSplit("a", "b")

        expected = [
            {"a": "Sentence 1. Sentence 2.", "b": "Sentence 1."},
            {"a": "Sentence 1. Sentence 2.", "b": "Sentence 2."},
            {"a": "Sentence 3.", "b": "Sentence 3."},
        ]
        result = transform(current, self.api)

        self.assertListEqual(expected, result)

    def test_postprocess_output(self):
        current = [{"a": "a<<<stop yayaya"}, {"a": "b<<<"}]
        transform = PostprocessOutput("a", "b", stopping_sequence="stop", remove_suffix="<<<")

        expected = [{"a": "a<<<stop yayaya", "b": "a"}, {"a": "b<<<", "b": "b"}]
        result = transform(current, self.api)

        self.assertListEqual(expected, result)

    def test_apply_template(self):
        current = [{"a": "LETTER A", "g": {"g": "gg"}}, {"a": "ay", "g": {"g": "ggs"}}]
        transform = ApplyTemplate("{a}-{g[g]}", "b")

        expected = [
            {"a": "LETTER A", "g": {"g": "gg"}, "b": "LETTER A-gg"},
            {"a": "ay", "g": {"g": "ggs"}, "b": "ay-ggs"},
        ]
        result = transform(current, self.api)

        self.assertListEqual(expected, result)

    def test_ask_prompt(self):
        current = [{"p": "hello"}, {"p": "bye"}]
        transform = AskPrompt("p", "a", system_msg="I am system.", start_with="YAYA")

        expected = [
            {"p": "hello", "a": "MOCK: <system: I am system.> <user: hello> <assistant: YAYA>"},
            {"p": "bye", "a": "MOCK: <system: I am system.> <user: bye> <assistant: YAYA>"},
        ]
        result = transform(current, self.api)
        self.assertListEqual(expected, result)

    def test_parse_annotations(self):
        current = [
            {
                "text": "some text",
                "ann_raw": '{ "annotations": [{ "text": "text", "reason": "it is correct", "annotation_type": 0 }, { "text": "some", "reason": "it is incorrect", "annotation_type": 1 }]}',
            }
        ] * 2

        annotation_span_categories = [
            {"name": "correct", "color": "rgb(0, 255, 0)", "description": "is correct"},
            {"name": "incorrect", "color": "rgb(255, 0, 0)", "description": "is incorrect"},
        ]
        annotation_overlap_allowed = True
        output_validation_model = AnnotationModelFactory.get_output_model(with_reason=True)
        transform = ParseAnnotations(
            "ann_raw", "ann", annotation_span_categories, annotation_overlap_allowed, output_validation_model
        )

        expected = [
            {
                "text": "some text",
                "ann_raw": '{ "annotations": [{ "text": "text", "reason": "it is correct", "annotation_type": 0 }, { "text": "some", "reason": "it is incorrect", "annotation_type": 1 }]}',
                "ann": [
                    {"reason": "it is correct", "start": 5, "text": "text", "type": 0},
                    {"reason": "it is incorrect", "start": 0, "text": "some", "type": 1},
                ],
            }
        ] * 2
        result = transform(current, self.api)

        self.assertListEqual(expected, result)

    def test_extract_tag(self):
        # join_occurances=True
        # remove_from_input=True
        current = [{"a": "code: <code>ccc </code> and <code>ddd</code>"}] * 2
        transform = ExtractTag("a", "code", "code", join_occurances=True, remove_from_input=True)

        expected = [{"a": "code: and", "code": "ccc\nddd"}] * 2
        result = transform(current, self.api)

        self.assertListEqual(expected, result)

    def test_extract_tag_2(self):
        # join_occurances=False
        # remove_from_input=False
        current = [{"a": "code: <code>ccc </code> and <code>ddd</code>"}] * 2
        transform = ExtractTag("a", "code", "code", join_occurances=False, remove_from_input=False)

        expected = [{"a": "code: <code>ccc </code> and <code>ddd</code>", "code": ["ccc", "ddd"]}] * 2
        result = transform(current, self.api)

        self.assertListEqual(expected, result)

    def test_extract_json(self):
        current = [{"a": 'f3 s,e fj3 {"annotations": [{"text": "test", "type": 3}]}'}] * 2
        transform = ExtractJson("a", "json")

        expected = [
            {
                "a": 'f3 s,e fj3 {"annotations": [{"text": "test", "type": 3}]}',
                "json": '{"annotations": [{"text": "test", "type": 3}]}',
            }
        ] * 2
        result = transform(current, self.api)

        self.assertListEqual(expected, result)


if __name__ == "__main__":
    logger.disabled = True
    unittest.main()
