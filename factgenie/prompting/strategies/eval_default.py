import logging

from factgenie.annotations import AnnotationModelFactory
from factgenie.prompting.strategies import register_llm_eval, SequentialStrategy
from factgenie.prompting import transforms as t

logger = logging.getLogger("factgenie")


@register_llm_eval(name="default")
class StructuredAnnotationStrategy(SequentialStrategy):
    def get_transform_sequence(self) -> list[t.Transform]:
        TEXT = SequentialStrategy.TEXT
        PROMPT = "prompt"
        ANNOTATIONS_RAW = "annotations_raw"
        ANNOTATIONS = SequentialStrategy.ANNOTATIONS

        system_msg = self.config.get("system_msg", None)
        starts_with = self.config.get("start_with", None)

        annotation_span_categories = self.config["annotation_span_categories"]
        annotation_overlap_allowed = self.config.get("annotation_overlap_allowed", False)
        with_reason = self.extra_args.get("with_reason", True)
        output_validation_model = AnnotationModelFactory.get_output_model(with_reason)

        return [
            # 1. Ask prompt.
            t.ApplyTemplate(self.config["prompt_template"], PROMPT),
            t.Log(text="Prompt: ", field=PROMPT, log_level="debug"),
            t.Log(text="Annotated text: ", field=TEXT),
            t.AskPrompt(
                PROMPT,
                ANNOTATIONS_RAW,
                system_msg,
                starts_with,
                completion_kwargs={"response_format": output_validation_model},
            ),
            # 2. Parse annotations.
            t.ParseAnnotations(
                ANNOTATIONS_RAW,
                ANNOTATIONS,
                annotation_span_categories,
                annotation_overlap_allowed,
                output_validation_model,
            ),
            # Metadata.
            t.Metadata(fields=[PROMPT]),
        ]
