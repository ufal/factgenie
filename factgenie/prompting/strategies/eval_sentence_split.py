import logging

from factgenie.annotations import AnnotationModelFactory
from factgenie.prompting import transforms as t
from factgenie.prompting.strategies import SequentialStrategy, register_llm_eval

logger = logging.getLogger("factgenie")


@register_llm_eval(name="sentence_split")
class SentenceSplitAnnotationStrategy(SequentialStrategy):
    def get_transform_sequence(self) -> list[t.Transform]:
        TEXT = SequentialStrategy.TEXT
        PART = "part"
        PROMPT = "annotation_prompt"
        ANNOTATION_RESPONSE = "annotation_response"
        ANNOTATIONS = SequentialStrategy.ANNOTATIONS

        annotation_span_categories = self.config["annotation_span_categories"]
        annotation_overlap_allowed = self.config.get("annotation_overlap_allowed", False)
        annotation_granularity = self.config.get("annotation_granularity", "words")
        with_reason = self.extra_args.get("with_reason", True)
        output_validation_model = AnnotationModelFactory.get_output_model(with_reason)

        return [
            # 1. Split sentences
            t.SentenceSplit(TEXT, PART),
            t.Log(text="Sentences: ", field=PART),
            # 2. Ask prompt.
            t.ApplyTemplate(self.config["prompt_template"], PROMPT),
            t.Log(text="Prompt: ", field=PROMPT, log_level="debug"),
            t.AskPrompt(PROMPT, ANNOTATION_RESPONSE),
            # 3. Parse annotations.
            t.Unify([ANNOTATION_RESPONSE], join_strings_by=t.join_string_long),
            t.ParseAnnotations(
                ANNOTATION_RESPONSE,
                ANNOTATIONS,
                annotation_span_categories,
                annotation_overlap_allowed,
                output_validation_model,
                annotation_granularity,
            ),
            # Metadata.
            t.Metadata([PROMPT]),
        ]
