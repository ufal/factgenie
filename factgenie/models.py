#!/usr/bin/env python3

import logging
import os

# LiteLLM seems to be triggering deprecation warnings in Pydantic, so we suppress them
import warnings
from ast import literal_eval

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from factgenie.annotations import AnnotationModelFactory
from factgenie.api import (
    AnthropicAPI,
    GeminiAPI,
    ModelAPI,
    OllamaAPI,
    OpenAIAPI,
    VertexAIAPI,
    VllmAPI,
)
from factgenie.campaign import CampaignMode
from factgenie.prompting import (
    GenerationStrategy,
    PromptingStrategy,
    RawOutputStrategy,
    StructuredOutputStrategy,
)

# also disable info logs from litellm
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Proxy").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)

# disable requests logging
logging.getLogger("httpx").setLevel(logging.ERROR)

logger = logging.getLogger("factgenie")

DIR_PATH = os.path.dirname(__file__)
LLM_ANNOTATION_DIR = os.path.join(DIR_PATH, "annotations")
LLM_GENERATION_DIR = os.path.join(DIR_PATH, "outputs")


class ModelFactory:
    """
    Factory for creating model instances based on configuration.
    This class is responsible for parsing the configuration, selecting the appropriate model API, and initializing the prompting strategy.
    """

    @staticmethod
    def get_model_apis():
        # List of available model APIs that the user can select from, stored as `api_provider` in the config.
        return {
            "openai": OpenAIAPI,
            "ollama": OllamaAPI,
            "vllm": VllmAPI,
            "anthropic": AnthropicAPI,
            "gemini": GeminiAPI,
            "vertexai": VertexAIAPI,
        }

    @staticmethod
    def get_prompt_strategies():
        return {
            CampaignMode.LLM_EVAL: {
                "default": StructuredOutputStrategy,
                "parse_raw": RawOutputStrategy,
            },
            CampaignMode.LLM_GEN: {
                "default": GenerationStrategy,
            },
        }

    @staticmethod
    def parse_api_provider(config):
        if "type" in config:
            logger.warning(
                "The `type` field is deprecated. Please use `api_provider` instead. This will be removed in a future version."
            )

        # Supporting the deprecated `type` field
        api_provider = config.get("api_provider", config.get("type"))

        # Supporting the deprecated suffixes
        if api_provider.endswith("_metric"):
            api_provider = api_provider[: -len("_metric")]
        elif api_provider.endswith("_gen"):
            api_provider = api_provider[: -len("_gen")]

        return api_provider

    @staticmethod
    def from_config(config, mode):
        api_provider = ModelFactory.parse_api_provider(config)

        prompt_strat = config.get("prompt_strat", "default")
        if "prompt_strat" not in config:
            logger.warning("Prompting strategy was not specified, using 'default'...")

        model_apis = ModelFactory.get_model_apis()
        prompt_strats = ModelFactory.get_prompt_strategies()[mode]

        # ensure the api_type and prompt_strat are valid
        if api_provider not in model_apis:
            raise ValueError(f"Model type {api_provider} is not implemented.")
        if prompt_strat not in prompt_strats:
            raise ValueError(f"Model type {prompt_strat} is not implemented.")

        return Model(config, mode, model_apis[api_provider](config), prompt_strats[prompt_strat](config))


class Model:
    def __init__(self, config: dict, mode: CampaignMode, model_api: ModelAPI, prompt_strat: PromptingStrategy):
        self.config = config
        self.campaign_mode = mode
        self.parse_model_args()
        self.model_api = model_api
        self.prompt_strat = prompt_strat

    def generate_output(self, data, text=None):
        """For backward compatibility with existing code."""
        return self.prompt_strat.get_model_output(api=self.model_api, data=data, text=text)

    def get_annotator_id(self):
        return "llm-" + ModelFactory.parse_api_provider(self.config) + "-" + self.config["model"]

    def get_config(self):
        return self.config

    def parse_model_args(self):
        if "model_args" not in self.config:
            return

        # implicitly convert all model_args to literals based on their format
        for arg in self.config["model_args"]:
            try:
                self.config["model_args"][arg] = literal_eval(self.config["model_args"][arg])
            except:
                pass
