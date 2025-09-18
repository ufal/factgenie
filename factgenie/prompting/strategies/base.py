#!/usr/bin/env python3

import abc
import logging
import traceback
import unittest

from factgenie.campaign import CampaignMode
from factgenie.prompting import transforms as t
from factgenie.prompting.model_apis import MockingAPI, ModelAPI
from factgenie.prompting.registry import (
    Registry,
    UnregisteredTracker,
    track_subclasses,
    untracked,
)

logger = logging.getLogger("factgenie")


@track_subclasses
class PromptingStrategy(abc.ABC):
    def __init__(self, config: dict, mode: str):
        self.config = config
        self.mode = mode
        self.extra_args = config.get("extra_args", {})
        self.completion_extra_kwargs = {}

    @abc.abstractmethod
    def get_output(self, api: ModelAPI, data, text=None) -> dict:
        """
        Abstract method that each subclass must implement to get output from the model.

        Args:
            api: The ModelAPI instance to use for calling the model
            data: The source data to be used in the prompt (if present)
            text: The text to be annotated (annotation tasks only)

        Returns:
            A dictionary containing:
                - 'output': (for LLM_GEN) a string containing the output text.
                - 'annotations': (for LLM_EVAL) a list of annotations.
                - 'metadata': A dictionary containing any information from the prompting you wish to have saved in the campaign output. Typically containing a single key 'prompt'.
                - 'thinking_trace': (optional) The reasoning trace from the LLM if available.
                - Other keys may be preserved depending on the strategy implementation.
            A dictionary with the expected output fields and any additional preserved fields
        """
        # TODO: Add the description for what the list of annotations looks like (inside """... Returns: ...""").
        pass

    def __call__(self, api: ModelAPI, data, text=None):
        return self.get_output(api, data, text)


class MissingRequirementException(Exception):
    pass


class MissingOutputsException(Exception):
    pass


class InvalidOutputLengthException(Exception):
    pass


class SequentialStrategy(PromptingStrategy):
    # Constants to be used by derived sequential strategies to avoid any typo mistakes (and become rename-proof).
    DATA = "data"
    TEXT = "text"
    OUTPUT = "output"
    ANNOTATIONS = "annotations"

    def __init__(self, config, mode: str):
        super().__init__(config, mode)
        self.transform_sequence = self.get_transform_sequence()
        self.verify_sequence()  # must be called after 'self.transform_sequence' is set and after 'super().__init__(...)' is called.

    @abc.abstractmethod
    def get_transform_sequence(self) -> list[t.Transform]:
        pass

    def verify_sequence(self):
        if self.mode == CampaignMode.LLM_GEN:
            current_keys = {self.DATA}
            expected_outputs = {self.OUTPUT}
        elif self.mode == CampaignMode.LLM_EVAL:
            current_keys = {self.DATA, self.TEXT}
            expected_outputs = {self.ANNOTATIONS}
        else:
            raise NotImplementedError(f"{self.mode} is not implemented")

        for i, step in enumerate(self.transform_sequence):
            # Throw if there is an unfulfilled requirement.
            unfulfilled = set(step.requires_fields) - current_keys
            if len(unfulfilled) > 0:
                error = f"Sequence for '{type(self)}' is not valid. At index {i}, '{type(step)}' is missing required fields: [{', '.join(list(unfulfilled))}]. Currently accessible fields are: [{', '.join(list(current_keys))}]."
                logger.error(error)
                raise MissingRequirementException(error)

            if step.clears_other_fields:
                current_keys = set()

            current_keys -= set(step.removes_fields)
            current_keys |= set(step.outputs_fields)  # Set union.

        unfulfilled = expected_outputs - current_keys
        if len(unfulfilled) > 0:
            error = f"Sequence for {type(self)} is missing output field: [{', '.join(list(unfulfilled))}]."
            logger.error(error)
            raise MissingOutputsException(error)

        # To see how the result will be used:
        # ../workflows.py --> save_record (result is what we output)
        # ../llm_campaign.py --> run_llm_campaign (search for 'model.generate_output')

        # What is saved (res = the dict returned by us):
        # saved['output'] = res['output'] in LLM_GEN, and LLM_GEN's output in LLM_EVAL.
        # saved['annotations'] if LLM_EVAL.
        # saved['thinking_trace'] if exists.
        # saved['metadata'] = default metadata | res['metadata'] (union).
        #  - Metadata are just for 'proof'/'replicability'. Inclusion of generation parts such as the 'prompt' gives a pretty strong evidence that the prompt actually looked like described.

        logger.info(f"Sequence for '{type(self)}' is valid.")

    def get_output(self, api: ModelAPI, data, text=None):
        try:
            # Initial condition
            current = [{"data": data}]
            if text is not None:
                current[0]["text"] = text

            for step in self.transform_sequence:
                current = step(current, api)

            if len(current) != 1:
                raise InvalidOutputLengthException(
                    f"The output of the sequential strategy is expected to be a list of dictionaries of length 1 but has length {len(current)}. This error is typically caused by utilizing a transform such as `SentenceSplit`, which increases the length of this list, and then forgetting to use the `Unify` transform to bring the length of the list back to 1."
                )
            assert len(current) == 1, ""
            answer = current[0]

            return answer

        except Exception as e:
            traceback.print_exc()
            logger.error(e)
            raise e


# Registry.
register_llm_gen = Registry(PromptingStrategy, "register_llm_gen")
register_llm_eval = Registry(PromptingStrategy, "register_llm_eval")
unregistered_prompting_strategy_tracker = UnregisteredTracker(PromptingStrategy, [register_llm_gen, register_llm_eval])


# ――――――――――――――――――――――――――――――――――― TESTS ―――――――――――――――――――――――――――――――――――


class SeqTests(unittest.TestCase):
    @untracked
    class PromptSeqTestStrategy(SequentialStrategy):
        def __init__(self, config: dict, mode: str, prompt_template: str, output_name: str):
            self.prompt_template = prompt_template
            self.output_name = output_name
            super().__init__(config, mode)

        def get_transform_sequence(self) -> list[t.Transform]:
            return [
                t.ApplyTemplate(self.prompt_template, "prompt"),
                t.AskPrompt("prompt", self.output_name),
            ]

    @untracked
    class IncorrectLengthTestStrategy(SequentialStrategy):
        def __init__(self, config: dict, mode: str):
            self.prompt_template = "Hello 1. Hello 2."
            self.output_name = "output"
            super().__init__(config, mode)

        def get_transform_sequence(self) -> list[t.Transform]:
            return [
                t.ApplyTemplate(self.prompt_template, "prompt"),
                t.SentenceSplit("prompt", "part"),
                t.AskPrompt("prompt", self.output_name),
            ]

    def __init__(self, *vargs, **kwargs):
        super().__init__(*vargs, **kwargs)
        self.api = MockingAPI()

    def test_sequential_work(self):
        good_prompt = "{text} and {data} are okay"

        # Should not raise.
        strat = SeqTests.PromptSeqTestStrategy({}, CampaignMode.LLM_EVAL, good_prompt, "annotations")
        result = strat.get_output(self.api, "/data/", "/text/")
        self.assertEqual(result["annotations"], "MOCK: <user: /text/ and /data/ are okay>")

    def test_sequential_work_2(self):
        good_prompt = "{data} is okay"

        # Should not raise.
        strat = SeqTests.PromptSeqTestStrategy({}, CampaignMode.LLM_GEN, good_prompt, "output")
        result = strat.get_output(self.api, "/data/")
        self.assertEqual(result["output"], "MOCK: <user: /data/ is okay>")

    def test_missing_requirement_1(self):
        bad_prompt = "{text} is not okay"
        self.assertRaises(
            MissingRequirementException,
            lambda: SeqTests.PromptSeqTestStrategy({}, CampaignMode.LLM_GEN, bad_prompt, "output"),
        )

    def test_missing_requirement_2(self):
        bad_prompt = "{typo} is not okay"
        self.assertRaises(
            MissingRequirementException,
            lambda: SeqTests.PromptSeqTestStrategy({}, CampaignMode.LLM_EVAL, bad_prompt, "annotations"),
        )

    def test_missing_output(self):
        good_prompt = "{data} is okay"
        self.assertRaises(
            MissingOutputsException,
            lambda: SeqTests.PromptSeqTestStrategy({}, CampaignMode.LLM_EVAL, good_prompt, "nonsense"),
        )

    def test_incorrect_length(self):
        # No input is necessary for the IncorrectLengthTestStrategy to split the length of output into 2.
        self.assertRaises(
            InvalidOutputLengthException,
            lambda: SeqTests.IncorrectLengthTestStrategy({}, CampaignMode.LLM_GEN)(self.api, {}, ""),
        )


if __name__ == "__main__":
    logger.disabled = True
    unittest.main()
