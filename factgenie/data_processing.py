import json
import logging
import os
import random
import re
import time
import traceback
from abc import abstractmethod
from ast import literal_eval
from typing import Literal
from pydantic import BaseModel, Field, ValidationError

from pandas.core.algorithms import is_extension_array_dtype

logger = logging.getLogger("factgenie")


def extract_json_from_raw(content):
    """
    Extract JSON object from raw model output, handling potential thinking traces.
    """
    import re

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


def parse_annotations(
    text: str, annotations_json: str, out_cls: type, annotation_span_categories: list[dict], overlap_allowed=False
):
    """
    Args:
        text: The text to be annotated.
        annotations_json: A json text containing the annotations.
        out_cls: The pytdantic model of annotations (`OutputAnnotations` or `OutputAnnotationsNoReason`).
        annotation_span_categories: A list of annotation categories. Each having an entry 'name'. Serves for logging.
        overlap_allowed: Whether overlap is allowed.
    """
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
