#!/usr/bin/env python3

from typing import Any, Dict, List, Optional, Type

from pydantic import AliasChoices, BaseModel, Field


class SpanAnnotation(BaseModel):
    reason: str = Field(description="The reason for the annotation.")
    text: str = Field(description="The text which is annotated.")
    # Do not name it type since it is a reserved keyword in JSON schema
    annotation_type: int = Field(
        description="Index to the list of span annotation types defined for the annotation campaign.",
        validation_alias=AliasChoices("annotation_type", "type"),
    )


class SpanAnnotationNoReason(BaseModel):
    text: str = Field(description="The text which is annotated.")
    # Do not name it type since it is a reserved keyword in JSON schema
    annotation_type: int = Field(
        description="Index to the list of span annotation types defined for the annotation campaign.",
        validation_alias=AliasChoices("annotation_type", "type"),
    )


class SpanAnnotationOccurenceIndex(BaseModel):
    reason: str = Field(description="The reason for the annotation.")
    text: str = Field(description="The text which is annotated.")
    # Do not name it type since it is a reserved keyword in JSON schema
    annotation_type: int = Field(
        description="Index to the list of span annotation types defined for the annotation campaign.",
        validation_alias=AliasChoices("annotation_type", "type"),
    )
    occurence_index: int = Field(
        description="The occurrence index to disambiguate between multiple occurrences of the span content. Integer value from 0 to N-1, where N is the number of occurrences."
    )


class SpanAnnotationNoReasonOccurenceIndex(BaseModel):
    text: str = Field(description="The text which is annotated.")
    # Do not name it type since it is a reserved keyword in JSON schema
    annotation_type: int = Field(
        description="Index to the list of span annotation types defined for the annotation campaign.",
        validation_alias=AliasChoices("annotation_type", "type"),
    )
    occurence_index: int = Field(
        description="The occurrence index to disambiguate between multiple occurrences of the span content. Integer value from 0 to N-1, where N is the number of occurrences."
    )


class OutputAnnotations(BaseModel):
    annotations: List[SpanAnnotation] = Field(description="The list of annotations.")


class OutputAnnotationsNoReason(BaseModel):
    annotations: List[SpanAnnotationNoReason] = Field(description="The list of annotations.")


class OutputAnnotationsOccurenceIndex(BaseModel):
    annotations: List[SpanAnnotationOccurenceIndex] = Field(description="The list of annotations.")


class OutputAnnotationsNoReasonOccurenceIndex(BaseModel):
    annotations: List[SpanAnnotationNoReasonOccurenceIndex] = Field(description="The list of annotations.")


class AnnotationModelFactory:
    """Factory for creating appropriate annotation output models based on configuration."""

    @staticmethod
    def get_output_model(with_reason: bool = True, with_occurence_index: bool = False) -> Type[BaseModel]:
        """
        Returns the appropriate output annotation model based on configuration.

        Args:
            with_reason: If True, includes reason field in annotations.
            with_occurence_index: If True, includes occurence_index field in annotations.

        Returns:
            The appropriate Pydantic model class.
        """
        if with_reason and with_occurence_index:
            return OutputAnnotationsOccurenceIndex
        elif with_reason and not with_occurence_index:
            return OutputAnnotations
        elif not with_reason and with_occurence_index:
            return OutputAnnotationsNoReasonOccurenceIndex
        else:
            return OutputAnnotationsNoReason

    @staticmethod
    def get_span_model(with_reason: bool = True, with_occurence_index: bool = False) -> Type[BaseModel]:
        """
        Returns the appropriate span annotation model based on configuration.

        Args:
            with_reason: If True, includes reason field in annotations.
            with_occurence_index: If True, includes occurence_index field in annotations.

        Returns:
            The appropriate Pydantic model class.
        """
        if with_reason and with_occurence_index:
            return SpanAnnotationOccurenceIndex
        elif with_reason and not with_occurence_index:
            return SpanAnnotation
        elif not with_reason and with_occurence_index:
            return SpanAnnotationNoReasonOccurenceIndex
        else:
            return SpanAnnotationNoReason
