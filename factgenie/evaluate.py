#!/usr/bin/env python3

from openai import OpenAI
from textwrap import dedent
import argparse
import yaml
import json
import sys

from pathlib import Path
import os
import coloredlogs
import logging
import time
import requests

# logging.basicConfig(format="%(message)s", level=logging.INFO, datefmt="%H:%M:%S")
coloredlogs.install(level="INFO", fmt="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DIR_PATH = os.path.dirname(__file__)
LLM_ANNOTATION_DIR = os.path.join(DIR_PATH, "annotations")


class LLMMetricFactory:
    @staticmethod
    def get_metric(config):
        metric_type = config["type"]
        model = config["model"]

        if metric_type == "openai":
            return OpenAIMetric(config)
        elif metric_type == "ollama":
            # we implemented specific input postprocessing for Llama 3
            if model.startswith("llama3"):
                return Llama3Metric(config)
            else:
                return OllamaMetric(config)
        else:
            raise NotImplementedError(f"The metric type {metric_type} is not implemented")


class LLMMetric:
    def __init__(self, metric_name, config):
        self.metric_name = metric_name

        self.system_msg = config.get("system_msg", None)
        if self.system_msg is None:
            logger.warning("System message (`system_msg`) field not set, using an empty string")
            self.system_msg = ""

        self.metric_prompt_template = config.get("prompt_template", None)

        if self.metric_prompt_template is None:
            raise ValueError("Prompt template (`prompt_template`) field is missing in the config")

    def create_annotation(self, text, j, example_idx):
        annotation_list = []
        current_pos = 0

        for error in j["errors"]:
            # find the `start` index of the error in the text
            start_pos = text.lower().find(error["text"].lower(), current_pos)

            if current_pos != 0 and start_pos == -1:
                # try from the beginning
                start_pos = text.find(error["text"])

            if start_pos == -1:
                logger.warning(f"Cannot find error {error} in text {text}, skipping")
                continue

            error["start"] = start_pos
            annotation_list.append(error)

            current_pos = start_pos + len(error["text"])

        return annotation_list


class OpenAIMetric(LLMMetric):
    def __init__(self, config):
        super().__init__(metric_name="llm-openai-" + config["model"], config=config)
        self.client = OpenAI()
        self.model = config["model"]

    def annotate_example(self, data, text):
        try:
            prompt = self.metric_prompt_template.format(data=data, text=text)

            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self.system_msg},
                    {"role": "user", "content": prompt},
                ],
            )
            annotation_str = response.choices[0].message.content

            j = json.loads(annotation_str)
            logger.info(j)
            return j
        except Exception as e:
            logger.error(e)
            return {"errors": []}


class OllamaMetric(LLMMetric):
    def __init__(self, config):
        super().__init__("llm-ollama-" + config["model"], config)
        self.API_URL = config.get("api_url", None)

        if self.API_URL is None:
            raise ValueError("API URL (`api_url`) field is missing in the config")

        self.model_args = config["model_args"]
        self.model = config["model"]
        self.seed = self.model_args.get("seed", None)

    def postprocess_output(self, output):
        output = output.strip()
        j = json.loads(output)
        return j

    def annotate_example(self, data, text):
        try:
            prompt = self.metric_prompt_template.format(data=data, text=text)

            response = requests.post(
                self.API_URL,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {"seed": self.seed, "temperature": 0},
                },
            )
            annotation_str = response.json()["response"]

            j = self.postprocess_output(annotation_str)
            logger.info(j)
            return j
        except Exception as e:
            logger.error(e)
            return {"errors": []}


class Llama3Metric(OllamaMetric):
    def postprocess_output(self, output):
        output = output.strip()
        j = json.loads(output)

        # the model often tends to produce a nested list
        if type(j["errors"][0]) == list:
            j["errors"] = j["errors"][0]

        return j
