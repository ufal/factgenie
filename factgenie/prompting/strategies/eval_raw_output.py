import logging

from factgenie.colors import Ansi
from factgenie.annotations import AnnotationModelFactory
from factgenie.prompting import transforms as t
from factgenie.prompting.strategies import SequentialStrategy, register_llm_eval

logger = logging.getLogger("factgenie")


@register_llm_eval(name="parse_raw")
class RawOutputAnnotationStrategy(SequentialStrategy):
    def get_transform_sequence(self) -> list[t.Transform]:
        TEXT = SequentialStrategy.TEXT
        PROMPT = "prompt"
        ANNOTATIONS_RAW = "annotations_raw"
        THINKING_TRACE = "thinking_trace"
        EXTRACTED = "extracted"
        ANNOTATIONS = SequentialStrategy.ANNOTATIONS

        system_msg = self.config.get("system_msg", None)
        starts_with = self.config.get("start_with", None)

        annotation_span_categories = self.config["annotation_span_categories"]
        annotation_overlap_allowed = self.config.get("annotation_overlap_allowed", False)
        annotation_granularity = self.config.get("annotation_granularity", "words")
        with_reason = self.extra_args.get("with_reason", True)
        output_validation_model = AnnotationModelFactory.get_output_model(with_reason)

        return [
            # 1. Ask prompt.
            t.ApplyTemplate(self.config["prompt_template"], PROMPT),
            t.Log(text="Prompt: ", field=PROMPT, log_level="debug"),
            t.Log(text="Annotated text: ", field=TEXT),
            t.AskPrompt(PROMPT, ANNOTATIONS_RAW, system_msg, starts_with),
            # 2. Extract think trace and annotations.
            t.ExtractTag(
                ANNOTATIONS_RAW,
                THINKING_TRACE,
                tag="think",
                join_occurances=True,
                remove_from_input=True,
                log_as="THINKING",
            ),
            t.Log(text="Thinking trace: ", field=THINKING_TRACE, color=Ansi.DARK_GRAY, log_level="debug"),
            t.ExtractJson(ANNOTATIONS_RAW, EXTRACTED),
            t.ParseAnnotations(
                EXTRACTED,
                ANNOTATIONS,
                annotation_span_categories,
                annotation_overlap_allowed,
                output_validation_model,
                annotation_granularity,
            ),
            # Metadata.
            t.Metadata(fields=[PROMPT]),
        ]
