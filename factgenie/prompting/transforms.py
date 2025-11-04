#!/usr/bin/env python3

import abc
import copy
import json
import logging
import numpy as np
import pkg_resources
import re
import time
import unittest

from fast_edit_distance import edit_distance
from itertools import cycle
from pydantic import BaseModel, ValidationError
from typing import Any, Literal, Type

from factgenie.annotations import AnnotationModelFactory
from factgenie.colors import Ansi
from factgenie.prompting import text_processing
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
join_string_medium = "\n" + "┅" * 50 + "\n"
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

    # def close(self):
    #     pass


def derive_field(current: list[dict], api: ModelAPI, function, output_field: str):
    """
    This utility function adds a field to each `dict` in the `list[dict]`. The field's name is set by the argument `output_field` and its value is obtained by calling `function(c, api)`, where `c` is a single `dict`.
    """
    return [{**c, output_field: function(c, api)} for c in current]


def derive_and_upsert_fields(current: list[dict], api: ModelAPI, function):
    """
    A more complex version of `derive_field`.

    `derive_and_upsert_fields` allows to derive multiple fields at once. To enable this functionality, the `function(c, api)` returns a `dict`.
    """
    return [{**c, **function(c, api)} for c in current]


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


class RemoveIfEmpty(Transform):
    def __init__(self, input_field: str):
        """
        Args:
            copy_type: "reference" (default) should be enough, unless you expect in-place modifications to either field. In those cases, use "shallow" or "deep".
        """
        self.input_field = input_field

    @property
    def requires_fields(self) -> list[str]:
        return [self.input_field]

    @property
    def outputs_fields(self) -> list[str]:
        return []

    def remove_if_empty(self, c: dict) -> dict:
        if self.input_field not in c or c[self.input_field].strip() != "":
            return c
        else:
            c = c.copy()
            c.pop(self.input_field)
            return c

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        return [self.remove_if_empty(c) for c in current]


class StringifyConversation(Transform):
    def __init__(self, input_field: str, output_field: str):
        """
        Limits the number of entries (to make debugging easier).
        """
        self.input_field = input_field
        self.output_field = output_field

    @property
    def requires_fields(self) -> list[str]:
        return [self.input_field]

    @property
    def outputs_fields(self) -> list[str]:
        return [self.output_field]

    def stringify(self, c: dict, api: ModelAPI):
        ROLE = ConverseLLM.ROLE
        CONTENT = ConverseLLM.CONTENT
        THINKING = ConverseLLM.REASONING_CONTENT

        def thinking_if_exists(conv_item: dict):
            thinking: str = conv_item.get(THINKING, "").strip()
            if thinking is None:
                return ""
            if len(thinking) > 1:
                return "<💭>" + thinking + "</💭>\n"  # 🤔💭
            else:
                return ""

        conversation = c[self.input_field]
        return "\n\n".join(
            f"[{conv_item[ROLE]}]\n{thinking_if_exists(conv_item)}{conv_item[CONTENT]}" for conv_item in conversation
        )

    def __call__(self, current: list[dict], api: ModelAPI):
        return derive_field(current, api, self.stringify, self.output_field)


class Put(Transform):
    def __init__(self, value: str, output_field: str):
        """
        Puts a literal value to the specified field.
        """
        self.output_field = output_field
        self.value = value

    @property
    def requires_fields(self) -> list[str]:
        return []

    @property
    def outputs_fields(self) -> list[str]:
        return [self.output_field]

    def get_value(self, c: dict, api: ModelAPI):
        return self.value

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        return derive_field(current, api, self.get_value, self.output_field)


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


class LimitEntries(Transform):
    def __init__(self, max_entries: int, warn: bool = True):
        """
        Limits the number of entries to make debugging easier.
        """
        self.max_entries = max_entries
        self.warn = warn

    @property
    def requires_fields(self) -> list[str]:
        return []

    @property
    def outputs_fields(self) -> list[str]:
        return []

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        if not self.warn:
            logger.warning(f"LimitEntries: Limiting entries to {self.max_entries}. DON'T FORGET TO REMOVE THIS!")
        return current[: self.max_entries]


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


class ParseAnnotations(Transform):
    def __init__(
        self,
        input_field: str,
        output_field: str,
        annotation_span_categories: list[dict],
        annotation_overlap_allowed: bool,
        output_validation_model: Type[BaseModel],
        annotation_granularity: str = "words",
    ):
        self.input_field = input_field
        self.output_field = output_field
        self.annotation_span_categories = annotation_span_categories
        self.annotation_overlap_allowed = annotation_overlap_allowed
        self.output_validation_model = output_validation_model
        self.annotation_granularity = annotation_granularity

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

            # Get occurrence index if available
            occurence_index = getattr(annotation, "occurence_index", None)

            # find the `start` index of the error in the text
            if self.annotation_granularity == "words":
                # Use word boundary matching to enforce word-level annotations
                # Escape special regex characters in the annotated span
                escaped_span = re.escape(annotated_span)

                # For word boundaries, we need to be more careful with special characters
                # If the span starts or ends with non-word characters, we should not enforce word boundaries there
                start_needs_boundary = annotated_span and annotated_span[0].isalnum()
                end_needs_boundary = annotated_span and annotated_span[-1].isalnum()

                # Create pattern with conditional word boundaries
                pattern = ""
                if start_needs_boundary:
                    pattern += r"\b"
                pattern += escaped_span
                if end_needs_boundary:
                    pattern += r"\b"

                if occurence_index is not None:
                    # Find all matches and select the one at the specified occurrence index
                    matches = list(re.finditer(pattern, text.lower(), re.IGNORECASE))
                    if 0 <= occurence_index < len(matches):
                        start_pos = matches[occurence_index].start()
                    elif matches:
                        # Invalid occurrence index, fall back to first match
                        logger.warning(
                            f"Invalid occurrence index {occurence_index} for span '{annotated_span}'. Using first occurrence."
                        )
                        start_pos = matches[0].start()
                    else:
                        start_pos = -1
                else:
                    # Original behavior: find first match
                    match = re.search(pattern, text.lower(), re.IGNORECASE)
                    start_pos = match.start() if match else -1
            else:
                # Use character-level matching (original behavior)
                if occurence_index is not None:
                    # Find all occurrences and select the one at the specified occurrence index
                    all_positions = []
                    start_search = 0
                    while True:
                        pos = text.lower().find(annotated_span, start_search)
                        if pos == -1:
                            break
                        all_positions.append(pos)
                        start_search = pos + 1

                    if 0 <= occurence_index < len(all_positions):
                        start_pos = all_positions[occurence_index]
                    elif all_positions:
                        # Invalid occurrence index, fall back to first occurrence
                        logger.warning(
                            f"Invalid occurrence index {occurence_index} for span '{annotated_span}'. Using first occurrence."
                        )
                        start_pos = all_positions[0]
                    else:
                        start_pos = -1
                else:
                    # Original behavior: find first occurrence
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


class ParseOptions(Transform):
    def __init__(
        self,
        input_field: str,
        output_field: str,
        label: str,
        choices: list[str] | None = None,
        choices_path: list[str] | None = None,
    ):
        """
        Parse option selected by `input_field` by taking the choice with minimal edit distance to that option. Before considering the edit distance, the choice will be shortened to match thelength of `input_field`.

        Args:
            input_field: The field containing the answer to match.
            output_field: The answer will be parsed and formatted into a factgenie-compatible stringified json.
            choices: A constant list of possible answers.
            choices_data_subfield: A list of answers varying per question, stored in c["data"][choices_data_subfield].
        """

        assert (choices is not None) ^ (
            choices_path is not None
        ), "You must select excatly one of [choices, choices_data_subfield] in the ParseOptions constructor"

        if choices_path is not None:
            assert len(choices_path) >= 1

        self.input_field = input_field
        self.output_field = output_field
        self.label = label
        self.choices = choices
        self.choices_path = choices_path

    @property
    def requires_fields(self) -> list[str]:
        return [self.input_field] + ([self.choices_path[0]] if self.choices_path is not None else [])

    @property
    def outputs_fields(self) -> list[str]:
        return [self.output_field]

    def get_choices(self, c: dict) -> list[str]:
        if self.choices is not None:
            return self.choices
        else:
            assert self.choices_path is not None
            return text_processing.extract_data(c, self.choices_path)

    def parse_options(self, c: dict, api: ModelAPI):
        """
        Parse selected option. It will match the option with minimal edit distance to the text. (Only the same-length substring will be matched.)
        """

        answer_str = c[self.input_field]

        def mb_letter_edit_distance(answer_str, choice_str):
            # We will try to match our answer with two substrings.
            # Imagine answer_str = "Increase" and choice_str = "A) Increase"
            #  1) "Increase" with "A) Incre"
            #  2) "Increase" with    "Increase"
            ed1 = edit_distance(answer_str, choice_str[: len(answer_str)])
            # Unless the answer is short (e.g. just "B" or "B)", in which case we don't want to do the case (2).
            if len(answer_str) <= 2:
                return ed1
            ed2 = edit_distance(answer_str, choice_str[3 : 3 + len(answer_str)])
            return min(ed1, ed2)

        # Only measure distance between a same-length prefix of the choice in order to not bias answer-matching unfairly. (Imagine it answers just 'B)', then the shortest edit distance is to the shortest answer.)
        choices = self.get_choices(c)
        edit_distances = [mb_letter_edit_distance(answer_str, choice) for choice in choices]
        # max_ed=5

        index = int(np.argmin(np.array(edit_distances)))

        # EXAMPLE: METADATA:
        # "options": [{"label": "answer", "values": ["a", "b", "c", "d"]}]
        # EXAMPLE: ROOT:
        # "options": [{"label": "answer", "index": "3", "value": "d", "optionList": ["Select an option...", "a", "b", "c", "d"]}]
        options_output = [
            {
                "label": self.label,
                "index": index,
                "value": choices[index],
                "optionList": ["Select an option..."] + choices,
            }
        ]

        return options_output

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        return derive_field(current, api, self.parse_options, self.output_field)


class SetOptions(Transform):
    def __init__(
        self,
        input_field: str,
        output_field: str,
        label: str,
        choices: list[str],
    ):
        self.input_field = input_field
        self.output_field = output_field
        self.label = label
        self.choices = choices

    @property
    def requires_fields(self) -> list[str]:
        return [self.input_field]

    @property
    def outputs_fields(self) -> list[str]:
        return [self.output_field]

    def set_options(self, c: dict, api: ModelAPI):
        selected = c[self.input_field]

        if selected not in self.choices:
            logger.error(f"Not one of the options (selected '{selected}', possible options are {self.choices})")
            return ""

        # return f"""
        # [
        #     {{
        #         "label": "{self.label}",
        #         "value": "{selected}",
        #         "index": "{self.choices.index(selected)}",
        #         "optionList": {self.choices},
        #     }}
        # ]
        # """

        return [
            {
                "label": self.label,
                "value": selected,
                "index": str(self.choices.index(selected)),  # Should be int but currently isn't in human annotations.
                "optionList": self.choices,
            }
        ]

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        return derive_field(current, api, self.set_options, self.output_field)


class ExtractRegex(Transform):
    # I thought about replacing `remove_from_input: bool` with `save_modified_input: str` to allow non-destructive modifications. I decided against it because it doesn't feel as intuitive. The same functionality can still be achieved with Duplicate.
    def __init__(
        self,
        input_field: str,
        output_field: str | None,
        join_occurances: bool = True,
        remove_from_input: bool = True,
        log_as: str | None = None,
        prevent_empty_overwrite: bool = False,
    ):
        """
        Args:
            join_occurances: If true, all occurances will be joined by "\n". Otherwise, `output[output_field]` will be a list of strings.
            remove_from_input: If true, all "<{tag}>...</{tag}>" blocks will be removed from the input field.
            log_as: `log_as="THINKING"` will log occurances as gray "[THINKING]" blocks.
            prevent_empty_overwrite: If nothing is found and the field exists, do not overwrite it.

        Each block as well as the modified (if `remove_from_input=True`) is stripped of the padding spaces.
        """
        self.input_field = input_field
        self.output_field = output_field
        self.join_occurances = join_occurances
        self.remove_from_input = remove_from_input
        self.log_as = log_as
        self.prevent_empty_overwrite = prevent_empty_overwrite

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

    @abc.abstractmethod
    def make_regex(self) -> str:
        """
        Constructs a regex. This regex will contain a single match group for the text that will be extracted. Everything matched by the regex (including outside of the first match group) will be removed from the input if `self.remove_from_input == True`.

        If the input field contains multiple matches and `self.join_matches == True`, they will be joined by "\n".
        """
        pass

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        regex = self.make_regex()
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
                # If `self.prevent_overwrite`, then don't overwrite existing field with nothing.
                if self.prevent_empty_overwrite and self.output_field in c and len(tag_blocks) == 0:
                    pass  # Prevent overwrite.
                else:
                    c_changes[self.output_field] = tag_output

            output.append(c | c_changes)

        return output


class ExtractTag(ExtractRegex):
    def __init__(
        self,
        input_field: str,
        output_field: str | None,
        tag: str,
        join_occurances: bool = True,
        remove_from_input: bool = True,
        log_as: str | None = None,
        prevent_empty_overwrite: bool = False,
    ):
        super().__init__(input_field, output_field, join_occurances, remove_from_input, log_as, prevent_empty_overwrite)
        self.tag = tag

    def make_regex(self) -> str:
        return f"\\s*<{self.tag}>(?:\n?)(.*?)(?:\n?)</{self.tag}>\\s*"


class ExtractCodeBlock(ExtractRegex):
    def __init__(
        self,
        input_field: str,
        output_field: str | None,
        language: str,
        join_occurances=True,
        remove_from_input=True,
        log_as: str | None = None,
        prevent_empty_overwrite: bool = False,
    ):
        super().__init__(input_field, output_field, join_occurances, remove_from_input, log_as, prevent_empty_overwrite)
        self.language = language

    def make_regex(self) -> str:
        return f"\\s*```{self.language}\n(.*?)\n```\\s*"


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


class ConverseLLM(Transform):
    ROLE = "role"
    CONTENT = "content"
    USER = "user"
    ASSISTANT = "assistant"
    REASONING_CONTENT = "reasoning_content"

    # The reason for putting them here is to create a consistent interface and to preserve the proper history.
    DEFAULT_EXTRACTORS: list[Transform] = [
        ExtractTag(
            CONTENT,
            REASONING_CONTENT,
            tag="think",
            join_occurances=True,
            remove_from_input=True,
            prevent_empty_overwrite=True,
        ),
        Log(text="THINKING: ", field=REASONING_CONTENT, color=Ansi.LIGHT_GRAY),
        RemoveIfEmpty(REASONING_CONTENT),
    ]

    def __init__(
        self,
        prompt_field: str,
        conversation_field: str,
        restart_conversation: bool = False,
        system_msg: None | str = None,
        start_with: str | None = None,
        start_with_field: str | None = None,
        completion_kwargs: dict = {},
        extractors: list[Transform] = DEFAULT_EXTRACTORS,
    ):
        """
        A conversation is a list of dictionaries. Each of those dictionaries has keys "role", "content", and optionally more (such as "thinking_content").

        Args:
            input_field: The field storing the prompt text to ask.
            conversation_field: The input/output field storing the current conversation.
                - Example: `["hi", "Hello, how can I help you?", "what's the weather", "Today it's sunny."]`.
                - If the field doesn't exist, it will be initialized to `[]`.
            restart_conversation: If true, conversation_field will be reset to an empty converstaion (`[]`) before starting.
            system_msg: The system message of the model.
            starts_with XOR starts_with_field: The text (or field) containing the start of the model's reply.
            completion_kwargs: Extra arguments for `litellm.completion(...)`.
            extractors: A special list of transforms that will be ran on the current response `[{"role": "assistant", "content": "..."}]`. Look at `ConverseLLM.DEFAULT_EXTRACTORS` for more information.
        """
        assert start_with is None or start_with_field is None, "You may only use one of [start_with, start_with_field]"

        self.prompt_field = prompt_field
        self.conversation_field = conversation_field
        self.restart_conversation = restart_conversation
        self.system_msg = system_msg
        self.start_with = start_with
        self.start_with_field = start_with_field
        self.completion_kwargs = completion_kwargs
        self.extractors = extractors

    @property
    def requires_fields(self) -> list[str]:
        return [self.prompt_field] + [self.start_with_field] if self.start_with_field is not None else []

    @property
    def outputs_fields(self) -> list[str]:
        return [self.conversation_field]

    @classmethod
    def construct_message(cls, prompt: str, system_msg: str | None, start_with: str | None, history: list[dict] | None):
        messages = []

        # ───── system message ─────
        if system_msg is not None:
            messages.append({"role": "system", "content": system_msg})

        # ───── history (user-assistant loop) ─────
        if history is not None:
            # for role, content in zip(cycle(["user", "assistant"]), history):
            for history_item in history:
                role = history_item[ConverseLLM.ROLE]
                content = history_item[ConverseLLM.CONTENT]
                messages.append({"role": role, "content": content})

        # ───── user message ─────
        messages.append({"role": "user", "content": prompt})

        # ───── current assistant message ─────
        if start_with is not None and start_with.strip() != "":
            messages.append({"role": "assistant", "content": start_with})

        return messages

    def continue_conversation(self, c: dict, api: ModelAPI):
        """Get model response with timing and logging."""
        prompt = c[self.prompt_field]

        start_with = None
        if self.start_with is not None:
            start_with = self.start_with
        elif self.start_with_field is not None:
            start_with = c[self.start_with_field]

        if self.restart_conversation or self.conversation_field not in c.keys() or c[self.conversation_field] is None:
            history = []
        else:
            history = c[self.conversation_field]

        messages = self.construct_message(prompt, system_msg=self.system_msg, start_with=start_with, history=history)

        start = time.time()
        response = api.get_model_response_with_retries(messages, prompt_strat_kwargs=self.completion_kwargs)
        logger.info(f"Received response in {time.time() - start:.2f} seconds.")

        logger.debug(f"Prompt tokens: {response.usage.prompt_tokens}")
        logger.debug(f"Response tokens: {response.usage.completion_tokens}")

        assert isinstance(response.choices[0].message.content, str)

        # Get the reply...
        # We always keep `start_with` in the history.
        response_text: str = (start_with if start_with is not None else "") + response.choices[0].message.content
        reasoning_content = getattr(response.choices[0].message, self.REASONING_CONTENT, "")

        # Create user history entry...
        user = {self.ROLE: self.USER, self.CONTENT: prompt}

        # Create assistant history entry...
        assistant = {self.ROLE: self.ASSISTANT, self.CONTENT: response_text}

        # Do a quick version check
        try:
            litellm_version_str = pkg_resources.get_distribution("litellm").version
            litellm_version = pkg_resources.parse_version(litellm_version_str)
            if litellm_version.major <= 1 and litellm_version.minor <= 70 and reasoning_content == "":
                logger.warning(f"Currently installed litellm (version {litellm_version_str}) is likely too outdated to properly show reasoning content. Please update your litellm (`pip install -U litellm`).")
        except:
            pass
        
        if reasoning_content != "":
            assistant |= {self.REASONING_CONTENT: reasoning_content}

        for extractor in self.extractors:
            assistant = extractor([assistant], api)[0]

        return history + [user] + [assistant]

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        return derive_field(current, api, self.continue_conversation, self.conversation_field)


class ConversationExtractResponse(Transform):
    def __init__(
        self,
        input_field: str,
        output_field: str,
        conversation_key: str = ConverseLLM.CONTENT,
        index: int = -1,
    ):
        """
        Extracts a specific item (string) from a conversation (a list of strings).
        Args:
            input_field: The name of the field holding the conversation.
            output_field: Which field to save the result to.
            index: Which item from the conversation to extract. Default (-1) is the last item.
        """
        self.input_field = input_field
        self.output_field = output_field
        self.conversation_key = conversation_key
        self.index = index

    @property
    def requires_fields(self) -> list[str]:
        return [self.input_field]

    @property
    def outputs_fields(self) -> list[str]:
        return [self.output_field]

    def extract_response(self, c: dict, api: ModelAPI):
        history = c[self.input_field]
        selected_history_item = history[self.index]
        return selected_history_item[self.conversation_key]

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        return derive_field(current, api, self.extract_response, self.output_field)


class ConversationUpdateResponse(Transform):
    def __init__(
        self,
        input_field: str,
        updated_response_field: str,
        conversation_key: str = ConverseLLM.CONTENT,
        index: int = -1,
    ):
        """
        Updates a specific item (string) of a conversation (a list of strings).
        Args:
            input_field: The name of the field holding the conversation.
            updated_response_field: The name of the field holding the updated part converstaion part.
            index: Which item from the conversation to extract. Default (-1) is the last item.
        """
        self.input_field = input_field
        self.updated_response_field = updated_response_field
        self.conversation_key = conversation_key
        self.index = index

    @property
    def requires_fields(self) -> list[str]:
        return [self.input_field, self.updated_response_field]

    @property
    def outputs_fields(self) -> list[str]:
        return []

    def update_response(self, c: dict, api: ModelAPI):
        conv = c[self.input_field]
        conv[self.index] = conv[self.index] | {self.conversation_key: c[self.updated_response_field]}
        return conv

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        return derive_field(current, api, self.update_response, self.input_field)


class ConversationAppendResponse(Transform):
    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"

    def __init__(
        self,
        conversation_field: str,
        response_field: str,
        role: str,
    ):
        """
        Updates a specific item (string) of a conversation (a list of strings).
        Args:
            conversation_field: The name of the field holding the conversation.
            response_field: The name of the field holding the updated part converstaion part.
            role: A role shold be either 'user' or 'assistant'.
        """
        self.conversation_field = conversation_field
        self.response_field = response_field
        self.role = role

    @property
    def requires_fields(self) -> list[str]:
        return [self.response_field]

    @property
    def outputs_fields(self) -> list[str]:
        return []

    def update_response(self, c: dict, api: ModelAPI):
        conv = c[self.conversation_field] if self.conversation_field in c else []
        return conv + [{ConverseLLM.ROLE: self.role, ConverseLLM.CONTENT: c[self.response_field]}]

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        return derive_field(current, api, self.update_response, self.conversation_field)


class AskPrompt(Transform):
    def __init__(
        self,
        input_field: str,
        output_field: str,
        system_msg: None | str = None,
        start_with: str | None = None,
        start_with_field: str | None = None,
        keep_start_with: bool = False,
        completion_kwargs: dict = {},
        reasoning_field: str | None = None,
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
                reasoning_field: <the reasoning content from LLM> (if available),
            },
        ]
        ```

        The `prompt_template` may refer to any input field, such as "{part}" or "{data}" from the example above.

        Args:
            input_field: Field containing the prompt text.
            output_field: Field to store the LLM response content.
            system_msg: Optional system message for the prompt.
            start_with: Optional text to start the assistant response with.
            completion_kwargs: Additional kwargs for the completion API.
            reasoning_field: Field name to store reasoning content (default: "thinking_trace").
        """
        assert start_with is None or start_with_field is None, "You may only use one of [start_with, start_with_field]"

        self.input_field = input_field
        self.output_field = output_field
        self.keep_start_with = keep_start_with

        # We reuse ConverseLLM for maximal feature-parity.
        self.converse_llm = ConverseLLM(
            input_field,
            output_field,
            restart_conversation=True,
            system_msg=system_msg,
            start_with=start_with,
            start_with_field=start_with_field,
            completion_kwargs=completion_kwargs,
        )

        self.reasoning_field = reasoning_field

    @property
    def requires_fields(self) -> list[str]:
        return self.converse_llm.requires_fields

    @property
    def outputs_fields(self) -> list[str]:
        return [self.output_field] + ([self.reasoning_field] if self.reasoning_field is not None else [])

    def get_model_response(self, c: dict, api: ModelAPI):
        """Get model response with timing and logging."""

        conv = self.converse_llm

        # Get the singular reply.
        response_dict = conv.continue_conversation(c, api)[-1]
        response = response_dict[ConverseLLM.CONTENT]

        # Crop out the `start_with[_field]` unless requested otherwise.
        if not self.keep_start_with:
            start_len = 0
            if conv.start_with is not None:
                start_len = len(conv.start_with)
            elif conv.start_with_field is not None:
                start_len = len(c[conv.start_with_field])

            response = response[start_len:]

        REASONING_CONTENT = ConverseLLM.REASONING_CONTENT
        if REASONING_CONTENT in response_dict:
            reasoning_content = response_dict[REASONING_CONTENT]
            logger.debug(f"Extracted reasoning content of length: {len(reasoning_content)}")
        else:
            # Default to a string because the followup transforms might depend on the field having a value.
            reasoning_content = ""

        output = {self.output_field: response}
        if self.reasoning_field is not None:
            output |= {self.reasoning_field: reasoning_content}

        return output

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        return derive_and_upsert_fields(current, api, self.get_model_response)


# ――――――――――――――――――――――――――――――――――― TESTS ―――――――――――――――――――――――――――――――――――


class TransformTests(unittest.TestCase):
    THOUGHT = "thinking"

    def __init__(self, *vargs, **kwargs):
        super().__init__(*vargs, **kwargs)
        self.api = MockingAPI()
        self.reasoning_api = MockingAPI(include_thought=self.THOUGHT)

    def test_duplicate(self):
        current = [{"a": "a"}, {"a": "a"}]
        transform = Duplicate("a", "b")

        expected = [{"a": "a", "b": "a"}, {"a": "a", "b": "a"}]
        result = transform(current, self.api)

        self.assertListEqual(expected, result)

    def test_put(self):
        current = [{"a": "a"}, {"a": "a"}]
        transform = Put("b", "b")

        expected = [{"a": "a", "b": "b"}, {"a": "a", "b": "b"}]
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

    def test_converse_llm_start(self):
        self.maxDiff = 2000
        current = [{"p": "hello"}] * 2
        transform = ConverseLLM("p", "c", system_msg="I am system.", start_with="YAYA")

        expected = [
            {
                "p": "hello",
                "c": [
                    {"role": "user", "content": "hello"},
                    {
                        "role": "assistant",
                        "content": "YAYAMOCK: <system: I am system.> <user: hello> <assistant: YAYA>",
                    },
                ],
            },
        ] * 2
        result = transform(current, self.api)
        self.assertListEqual(expected, result)

    def test_ask_prompt_start_with_field(self):
        current = [{"f": "ALOHA", "prompt": "hello"}] * 2

        transform = AskPrompt("prompt", "a", system_msg="I am system.", start_with_field="f")

        expected = [
            {"f": "ALOHA", "prompt": "hello", "a": "MOCK: <system: I am system.> <user: hello> <assistant: ALOHA>"}
        ] * 2
        result = transform(current, self.api)

        self.assertListEqual(expected, result)

    def test_ask_prompt_with_history(self):
        ROLE = ConverseLLM.ROLE
        CONTENT = ConverseLLM.CONTENT

        current = 2 * [
            {
                "c": [
                    {ROLE: "user", CONTENT: "hi"},
                    {ROLE: "assistant", CONTENT: "Hello!"},
                    {ROLE: "user", CONTENT: "how is the weather"},
                    {ROLE: "assistant", CONTENT: "It is sunny."},
                ],
                "p": "what about tomorrow?",
            }
        ]

        # The other params of ConverseLLM are indirectly tested through AskPrompt's tests.
        transform = ConverseLLM("p", "c")

        expected = 2 * [
            {
                "c": [
                    {ROLE: "user", CONTENT: "hi"},
                    {ROLE: "assistant", CONTENT: "Hello!"},
                    {ROLE: "user", CONTENT: "how is the weather"},
                    {ROLE: "assistant", CONTENT: "It is sunny."},
                    {ROLE: "user", CONTENT: "what about tomorrow?"},
                    {
                        ROLE: "assistant",
                        CONTENT: "MOCK: <user: hi> <assistant: Hello!> <user: how is the weather> <assistant: It is sunny.> <user: what about tomorrow?>",
                    },
                ],
                "p": "what about tomorrow?",
            }
        ]
        result = transform(current, self.api)

        self.assertListEqual(expected, result)

    def test_conversation_extract(self):
        ROLE = ConverseLLM.ROLE
        CONTENT = ConverseLLM.CONTENT
        current = 2 * [
            {
                "c": [
                    {ROLE: "user", CONTENT: "C1"},
                    {ROLE: "user", CONTENT: "C2"},
                    {ROLE: "user", CONTENT: "C3"},
                    {ROLE: "user", CONTENT: "C4"},
                    {ROLE: "assistant", CONTENT: "C5"},
                    {ROLE: "user", CONTENT: "C6"},
                ],
                "b": "some random field",
            }
        ]

        # The other params of ConverseLLM are indirectly tested through AskPrompt's tests.
        transform = ConversationExtractResponse("c", "extracted", index=-2)

        expected = 2 * [
            {
                "c": [
                    {ROLE: "user", CONTENT: "C1"},
                    {ROLE: "user", CONTENT: "C2"},
                    {ROLE: "user", CONTENT: "C3"},
                    {ROLE: "user", CONTENT: "C4"},
                    {ROLE: "assistant", CONTENT: "C5"},
                    {ROLE: "user", CONTENT: "C6"},
                ],
                "b": "some random field",
                "extracted": "C5",
            }
        ]
        result = transform(current, self.api)

        self.assertListEqual(expected, result)

    def test_conversation_update(self):
        ROLE = ConverseLLM.ROLE
        CONTENT = ConverseLLM.CONTENT

        current = 2 * [
            {
                "c": [
                    {ROLE: "user", CONTENT: "C1"},
                    {ROLE: "user", CONTENT: "C2"},
                    {ROLE: "user", CONTENT: "C3"},
                    {ROLE: "user", CONTENT: "C4"},
                    {ROLE: "assistant", CONTENT: "C5"},
                    {ROLE: "user", CONTENT: "C6"},
                ],
                "u": "U5",
            }
        ]

        # The other params of ConverseLLM are indirectly tested through AskPrompt's tests.
        transform = ConversationUpdateResponse("c", "u", index=-2)

        expected = 2 * [
            {
                "c": [
                    {ROLE: "user", CONTENT: "C1"},
                    {ROLE: "user", CONTENT: "C2"},
                    {ROLE: "user", CONTENT: "C3"},
                    {ROLE: "user", CONTENT: "C4"},
                    {ROLE: "assistant", CONTENT: "U5"},
                    {ROLE: "user", CONTENT: "C6"},
                ],
                "u": "U5",
            }
        ]
        result = transform(current, self.api)

        self.assertListEqual(expected, result)

    def test_conversation_append(self):
        ROLE = ConverseLLM.ROLE
        CONTENT = ConverseLLM.CONTENT

        current = 2 * [
            {
                "c": [
                    {ROLE: "user", CONTENT: "C1"},
                    {ROLE: "user", CONTENT: "C2"},
                    {ROLE: "user", CONTENT: "C3"},
                    {ROLE: "user", CONTENT: "C4"},
                    {ROLE: "assistant", CONTENT: "C5"},
                    {ROLE: "user", CONTENT: "C6"},
                ],
                "a": "C7",
            }
        ]

        # The other params of ConverseLLM are indirectly tested through AskPrompt's tests.
        transform = ConversationAppendResponse("c", "a", role="NEW")

        expected = 2 * [
            {
                "c": [
                    {ROLE: "user", CONTENT: "C1"},
                    {ROLE: "user", CONTENT: "C2"},
                    {ROLE: "user", CONTENT: "C3"},
                    {ROLE: "user", CONTENT: "C4"},
                    {ROLE: "assistant", CONTENT: "C5"},
                    {ROLE: "user", CONTENT: "C6"},
                    {ROLE: "NEW", CONTENT: "C7"},
                ],
                "a": "C7",
            }
        ]
        result = transform(current, self.api)

        self.assertListEqual(expected, result)

    def test_ask_prompt_with_custom_reasoning_field(self):
        current = [{"p": "hello"}]
        transform = AskPrompt("p", "a", reasoning_field="custom_thinking")

        result = transform(current, self.reasoning_api)

        # Should have the main output field
        self.assertIn("a", result[0])
        self.assertEqual(result[0]["a"], "MOCK: <user: hello>")
        self.assertEqual(result[0]["custom_thinking"], self.THOUGHT)

    def test_ask_prompt_with_custom_reasoning_field_2(self):
        THOUGHT = "HEY"
        current = [{"p": f"hello<think>{THOUGHT}</think> you"}]
        transform = AskPrompt("p", "a", reasoning_field="custom_thinking")

        result = transform(current, self.api)

        # Should have the main output field
        self.assertIn("a", result[0])
        self.assertEqual(result[0]["a"], "MOCK: <user: hello you>")
        self.assertEqual(result[0]["custom_thinking"], THOUGHT)

    def test_ask_prompt_with_no_thought(self):
        current = [{"p": "hello"}]
        transform = AskPrompt("p", "a", reasoning_field="custom_thinking")

        result = transform(current, self.api)

        # Should have the main output field
        self.assertIn("a", result[0])
        self.assertEqual(result[0]["a"], "MOCK: <user: hello>")
        self.assertEqual(result[0]["custom_thinking"], "")

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
            "ann_raw", "ann", annotation_span_categories, annotation_overlap_allowed, output_validation_model, "words"
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

    def test_extract_code_block(self):
        self.maxDiff = None
        current = [{"a": "code:\n```python\nprint('hello world')\nprint('I am the guy')\n```"}] * 2
        transform = ExtractCodeBlock("a", "code", "python", join_occurances=True, remove_from_input=True)

        expected = [{"a": "code:", "code": "print('hello world')\nprint('I am the guy')"}] * 2
        result = transform(current, self.api)

        self.assertListEqual(expected, result)

    def test_extract_tag_preserves_existing_field(self):
        # Test that ExtractTag doesn't overwrite existing field when no tags are found.
        current = [{"a": "no tags here", "thinking_trace": "existing reasoning content"}]
        transform = ExtractTag(
            "a", "thinking_trace", "think", join_occurances=True, remove_from_input=False, prevent_empty_overwrite=True
        )

        # Should preserve the existing thinking_trace since no <think> tags were found.
        expected = [{"a": "no tags here", "thinking_trace": "existing reasoning content"}]
        result = transform(current, self.api)

        self.assertListEqual(expected, result)

    def test_extract_tag_preserve_overrides_when_needed(self):
        # Test that ExtractTag doesn't overwrite existing field if it found a tag.
        current = [{"a": f"tags here <think>{self.THOUGHT}</think>", "thinking_trace": "existing reasoning content"}]
        transform = ExtractTag(
            "a", "thinking_trace", "think", join_occurances=True, remove_from_input=False, prevent_empty_overwrite=True
        )

        # Should override the thinking_trace with the new tag content.
        expected = [{"a": f"tags here <think>{self.THOUGHT}</think>", "thinking_trace": self.THOUGHT}]
        result = transform(current, self.api)

        self.assertListEqual(expected, result)

    def test_extract_tag_preserve_always_outputs_field(self):
        # Test that ExtractTag with prevent_empty_override will output the required field if it doesn't exist.
        current = [{"a": "no tags here"}]
        transform = ExtractTag(
            "a", "thinking_trace", "think", join_occurances=True, remove_from_input=False, prevent_empty_overwrite=True
        )

        # Should be empty now.
        expected = [{"a": "no tags here", "thinking_trace": ""}]
        result = transform(current, self.api)

        self.assertListEqual(expected, result)

    def test_extract_tag_new_line(self):
        # join_occurances=False
        # remove_from_input=False
        current = [{"a": "code: <code>\nccc \n</code> and <code>ddd</code>"}] * 2
        transform = ExtractTag("a", "code", "code", join_occurances=False, remove_from_input=False)

        expected = [{"a": "code: <code>\nccc \n</code> and <code>ddd</code>", "code": ["ccc", "ddd"]}] * 2
        result = transform(current, self.api)

        self.assertListEqual(expected, result)

    def test_extract_tag_overwrites_when_content_found(self):
        # Test that ExtractTag does overwrite existing field when tags are found
        current = [{"a": "some <think>new thinking</think> here", "thinking_trace": "old reasoning"}]
        transform = ExtractTag("a", "thinking_trace", "think", join_occurances=True, remove_from_input=False)

        # Should overwrite with the new content found in tags
        expected = [{"a": "some <think>new thinking</think> here", "thinking_trace": "new thinking"}]
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

    def test_parse_option_easy(self):
        LABEL = "answer"
        CHOICES = ["A) Increasing", "B) Decreasing"]
        OUT_CHOIES = ["Select an option..."] + CHOICES
        current = [{"a": "A) Increasing"}] * 2
        transform = ParseOptions("a", "options", LABEL, choices=CHOICES)

        expected = [
            {
                "a": "A) Increasing",
                "options": [
                    {
                        "index": 0,
                        "label": LABEL,
                        "optionList": OUT_CHOIES,
                        "value": "A) Increasing",
                    }
                ],
            }
        ] * 2
        result = transform(current, self.api)

        self.assertListEqual(expected, result)

    def test_parse_option_path(self):
        LABEL = "answer"
        CHOICES = ["A) Increasing", "B) Decreasing"]
        OUT_CHOIES = ["Select an option..."] + CHOICES
        current = [{"a": "A) Increasing", "b": {"c": CHOICES}}] * 2
        transform = ParseOptions("a", "options", LABEL, choices_path=["b", "c"])

        expected = [
            {
                "a": "A) Increasing",
                "b": {"c": CHOICES},
                "options": [
                    {
                        "index": 0,
                        "label": LABEL,
                        "optionList": OUT_CHOIES,
                        "value": "A) Increasing",
                    }
                ],
            }
        ] * 2
        result = transform(current, self.api)

        self.assertListEqual(expected, result)

    def test_parse_option_short(self):
        LABEL = "answer"
        CHOICES = ["A) Increasing", "B) Decreasing"]
        OUT_CHOIES = ["Select an option..."] + CHOICES
        current = [{"a": "A"}] * 2
        transform = ParseOptions("a", "options", LABEL, choices=CHOICES)

        expected = [
            {
                "a": "A",
                "options": [
                    {
                        "index": 0,
                        "label": LABEL,
                        "optionList": OUT_CHOIES,
                        "value": "A) Increasing",
                    }
                ],
            }
        ] * 2
        result = transform(current, self.api)

        self.assertListEqual(expected, result)

    def test_parse_option_bad_start_1(self):
        LABEL = "answer"
        CHOICES = ["A) Increasing", "B) Decreasing"]
        OUT_CHOIES = ["Select an option..."] + CHOICES
        current = [{"a": "Increasing"}] * 2
        transform = ParseOptions("a", "options", LABEL, choices=CHOICES)

        expected = [
            {
                "a": "Increasing",
                "options": [
                    {
                        "index": 0,
                        "label": LABEL,
                        "optionList": OUT_CHOIES,
                        "value": "A) Increasing",
                    }
                ],
            }
        ] * 2
        result = transform(current, self.api)

        self.assertListEqual(expected, result)

    def test_parse_option_bad_start_2(self):
        LABEL = "answer"
        CHOICES = ["A) Decreasing", "B) Increasing"]
        OUT_CHOIES = ["Select an option..."] + CHOICES
        current = [{"a": "Increasing"}] * 2
        transform = ParseOptions("a", "options", LABEL, choices=CHOICES)
        expected = [
            {
                "a": "Increasing",
                "options": [
                    {
                        "index": 1,
                        "label": LABEL,
                        "optionList": OUT_CHOIES,
                        "value": "B) Increasing",
                    }
                ],
            }
        ] * 2
        result = transform(current, self.api)

        self.assertListEqual(expected, result)

    def test_stringify_conversation(self):
        ROLE = ConverseLLM.ROLE
        USER = ConverseLLM.USER
        ASSISTANT = ConverseLLM.ASSISTANT
        REASONING_CONTENT = ConverseLLM.REASONING_CONTENT
        CONTENT = ConverseLLM.CONTENT
        current = [
            {"c": [{ROLE: USER, CONTENT: "hi"}, {ROLE: ASSISTANT, CONTENT: "hey you", REASONING_CONTENT: "hmmm"}]}
        ]

        transform = StringifyConversation("c", "cc")

        expected = [
            {
                "c": [{ROLE: USER, CONTENT: "hi"}, {ROLE: ASSISTANT, CONTENT: "hey you", REASONING_CONTENT: "hmmm"}],
                "cc": "[user]\nhi\n\n[assistant]\n<💭>hmmm</💭>\nhey you",
            }
        ]
        result = transform(current, self.api)

        self.assertListEqual(expected, result)

    def test_limit_entries(self):
        current = [{"a": "a"}] * 5
        transform = LimitEntries(3, warn=False)

        expected = [{"a": "a"}] * 3
        result = transform(current, self.api)

        self.assertListEqual(expected, result)


if __name__ == "__main__":
    logger.disabled = True
    unittest.main()
