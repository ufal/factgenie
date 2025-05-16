import logging

from factgenie.prompting.strategies import register_llm_gen, SequentialStrategy
from factgenie.prompting import transforms as t

logger = logging.getLogger("factgenie")


@register_llm_gen(name="default")
class GenerationStrategy(SequentialStrategy):
    def get_transform_sequence(self) -> list[t.Transform]:
        PROMPT = "prompt"
        OUTPUT = SequentialStrategy.OUTPUT

        system_msg = self.config.get("system_msg", None)
        starts_with = self.config.get("start_with", None)

        stopping_sequence = self.extra_args.get("stopping_sequence", None)
        remove_suffix = self.extra_args.get("remove_suffix", None)

        return [
            # 1. Generation.
            t.ApplyTemplate(self.config["prompt_template"], PROMPT),
            t.AskPrompt(PROMPT, OUTPUT, system_msg, starts_with),
            t.PostprocessOutput(OUTPUT, OUTPUT, stopping_sequence, remove_suffix),
            # Logging and metadata.
            t.Log(text="Output: ", field=OUTPUT),
            t.Metadata(fields=[PROMPT]),
        ]
