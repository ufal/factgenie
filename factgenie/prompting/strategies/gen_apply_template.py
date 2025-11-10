import logging

from factgenie.prompting.strategies import register_llm_gen, SequentialStrategy
from factgenie.prompting import transforms as t

logger = logging.getLogger("factgenie")


@register_llm_gen(name="apply_template")
class ApplyTemplateStrategy(SequentialStrategy):
    def get_transform_sequence(self) -> list[t.Transform]:
        OUTPUT = SequentialStrategy.OUTPUT

        return [
            t.ApplyTemplate(self.config["prompt_template"], OUTPUT),
            t.Log(text="Output: ", field=OUTPUT),
        ]
