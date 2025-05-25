#!/usr/bin/env python3

from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field


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
    annotations: List[SpanAnnotation] = Field(description="The list of annotations.")


class OutputAnnotationsNoReason(BaseModel):
    annotations: List[SpanAnnotationNoReason] = Field(description="The list of annotations.")


class AnnotationModelFactory:
    """Factory for creating appropriate annotation output models based on configuration."""

    @staticmethod
    def get_output_model(with_reason: bool = True) -> Type[BaseModel]:
        """
        Returns the appropriate output annotation model based on whether reasons are required.

        Args:
            with_reason: If True, returns OutputAnnotations with reasons.
                         If False, returns OutputAnnotationsNoReason.

        Returns:
            The appropriate Pydantic model class.
        """
        if with_reason:
            return OutputAnnotations
        else:
            return OutputAnnotationsNoReason

    @staticmethod
    def get_span_model(with_reason: bool = True) -> Type[BaseModel]:
        """
        Returns the appropriate span annotation model based on whether reasons are required.

        Args:
            with_reason: If True, returns SpanAnnotation with reasons.
                         If False, returns SpanAnnotationNoReason.

        Returns:
            The appropriate Pydantic model class.
        """
        if with_reason:
            return SpanAnnotation
        else:
            return SpanAnnotationNoReason
