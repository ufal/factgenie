import logging

from factgenie.prompting.model_apis import ModelAPI
from factgenie.prompting.strategies import register_llm_gen, SequentialStrategy
from factgenie.prompting import transforms as t

logger = logging.getLogger("factgenie")


# An example implementation of a custom Transform:
# This transform takes text from input_field, converts it to uppercase, and saves it as output_field.
class MakeUppercase(t.Transform):
    # Conventions:
    #  - If the transform has an `input_field` argument, it should be the first argument.
    #  - If the transform has an `output_field` argument, it should be the second argument.
    # Here we just save which fields we the transform should use.
    def __init__(self, input_field, output_field):
        self.input_field = input_field
        self.output_field = output_field

    # The field `self.input_field` is required for this transform to work.
    @property
    def requires_fields(self) -> list[str]:
        return [self.input_field]

    # It produces a single field called `self.output_field`.
    @property
    def outputs_fields(self) -> list[str]:
        return [self.output_field]

    # This property would only return true if calling this transform would clear all the keys from the input lists of dictionaries (except for `self.output_fields`). This property returns False by default and doesn't need to be explicitly implemented.
    @property
    def clears_other_fields(self) -> bool:
        return False

    # We create a helper function that will be called on each dictionary from the list of dictionaries separately.
    # It extracts the selected field, uppercases it, and returns it.
    def uppercase(self, c: dict, api: ModelAPI) -> list[dict]:
        text = c[self.input_field]
        capitalized = text.upper()
        return capitalized

    # In the actual call method, we utilize the `derive_field` helper function. This helper function goes through the list of dictionaries, calls your selected function on each dictionary, and saves the returned value under your selected key in the output dictionary.
    def __call__(self, current: list[dict], api: ModelAPI):
        return t.derive_field(current, api, self.uppercase, self.output_field)


# A sequential strategy will initially construct a list of dictionaries looking like:
#     [{"data": data, "text": text}]
# where `data` is the relevant data and "text" is only present in LLM_EVAL mode.
#
# It will then apply all the transforms in order. Each transform taking a list of dictionaries as input and returning a new list of dictionaries as output. Typically, a transform will utilize the existing keys in the dictionary to construct a new key-value pair in the dictionary.
#
#
# If you are processing the whole text at once, the list will remain of lenght=1 the whole time. However, it is also possible split up the input into different parts, work on them separately, and then join them back together. An example of what this might look like:
#
# Initial state:
#     [{"data": data, "text": text}]
# After applying `SentenceSplit("text", "sentence")` transform on it:
#     [{"data": data, "text": text, "sentence": "Some random first sentence."}, {"data": data, "text": text, "sentence": "Look a second sentence."}, ...]
# After PromptTemplate("How do you rate '{sentence}'", "prompt"), you will get:
#     [{..., "prompt": "How do you rate 'Some random first sentence.'"}, {..., "prompt": "How do you rate 'Look a second sentence.'"}, ...];
# After AskPrompt("prompt", "answer"), you will get:
#     [{..., "answer": "This sentence is too random, 3/10."}, {..., "answer": "This is a great sentence, 8/10."}]
#
#
# The output of the sequential strategy is expected to contain either the key "output" (if LLM_GEN) or "annotations" (if LLM_EVAL). You may optoinally also save any other keys in the campaign file using the `Metadata` transform. This can is useful for research reproducibility (e.g. saving the exact prompt).


# This example strategy is basically `./gen_default.py` plus our MakeUppercase transform.
@register_llm_gen("example")
class ExampleStrategy(SequentialStrategy):
    def get_transform_sequence(self) -> list[t.Transform]:
        # It is recommended to define string constants for transform's fields here.
        # Use SequentialStrategy.{TEXT, DATA, OUTPUT, ANNOTATIONS} when possible.
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
            MakeUppercase(OUTPUT, OUTPUT),
            # Logging and metadata.
            t.Log(text="Output: ", field=OUTPUT),
            t.Metadata(fields=[PROMPT]),
        ]


# All files in this directory will be imported automatically 🎊.
