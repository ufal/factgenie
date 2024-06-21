#!/usr/bin/env python3

import traceback
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
import copy

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
        self.annotation_key = config.get("annotation_key", "errors")

        self.system_msg = config.get("system_msg", None)
        if self.system_msg is None:
            logger.warning("System message (`system_msg`) field not set, using an empty string")
            self.system_msg = ""

        self.metric_prompt_template = config.get("prompt_template", None)

        if self.metric_prompt_template is None:
            raise ValueError("Prompt template (`prompt_template`) field is missing in the config")

    def postprocess_annotations(self, text, model_json):
        annotation_list = []
        current_pos = 0

        for error in model_json[self.annotation_key]:
            # find the `start` index of the error in the text
            start_pos = text.lower().find(error["text"].lower(), current_pos)

            if current_pos != 0 and start_pos == -1:
                # try from the beginning
                start_pos = text.find(error["text"])

            if start_pos == -1:
                logger.warning(f"Cannot find error {error} in text {text}, skipping")
                continue

            error["start"] = start_pos
            annotation_list.append(copy.deepcopy(error))

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

            logger.debug(f"Calling OpenAI API with prompt: {prompt}")
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

            return self.postprocess_annotations(text=text, model_json=j)
        except Exception as e:
            logger.error(e)
            return {"error": str(e)}


class OllamaMetric(LLMMetric):
    def __init__(self, config):
        super().__init__("llm-ollama-" +config["model"] + config.get("metric_name_suffix", ""), config)
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
        prompt = self.metric_prompt_template.format(data=data, text=text)
        request_d = {
                "model": self.model,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {"seed": self.seed, "temperature": 0},
        }
        msg = f"Ollama API {self.API_URL} with args:\n\t{request_d}"
        try:
            logger.debug(f"Calling {msg}")
            response = requests.post(self.API_URL, json=request_d)
            annotation_str = response.json()["response"]

            j = self.postprocess_output(annotation_str)
            logger.info(j)
            return self.postprocess_annotations(text=text, model_json=j)
        except Exception as e:
            logger.error(f"Called {msg}\n\n and received {response=} before the error:{e}")
            traceback.print_exc()
            return {"error": str(e)}


class Llama3Metric(OllamaMetric):
    def postprocess_output(self, output):
        output = output.strip()
        j = json.loads(output)

        # the model often tends to produce a nested list
        if len(j[self.annotation_key]) == 1 and type(j[self.annotation_key][0]) == list:
            j[self.annotation_key] = j[self.annotation_key][0]

        return j
