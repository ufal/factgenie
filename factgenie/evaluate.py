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


class LLMMetric:
    def __init__(self, metric_name, load_args=None):
        self.metric_name = metric_name

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

    # def run(self, data):
    #     annotations = []

    #     for i, (data_input, model_out) in enumerate(data):
    #         try:
    #             # logger.info(f"{self.dataset_name} | {self.setup_name} | {self.model_name} | {i+1}/{len(data)}")
    #             logger.info(f"{i+1}/{len(data)}")
    #             j = self.annotate_example(data=data_input, text=model_out)
    #             annotation = self.create_annotation(model_out, j, example_idx=i)
    #             annotations.append(annotation)

    #             logger.info("=" * 80)
    #         except Exception as e:
    #             logger.error(f"Error while annotating example: {e}")
    #             logger.error(f"Example: {model_out}")
    #             logger.error(f"Data: {data_input}")

    #     return annotations


class Llama3Metric(LLMMetric):
    def __init__(self, load_args=None):
        super().__init__("llama3", load_args)

        self.API_KEY = os.getenv("TG_WEBUI_API_KEY")
        self.API_URL = f"http://quest.ms.mff.cuni.cz/nlg/text-generation-api/v1/chat/completions"

        base_path = os.path.dirname(os.path.realpath(__file__))
        with open(f"{base_path}/llm-eval/llama3_metric.yaml") as f:
            config = yaml.safe_load(f)

        self.system_msg = config["system_msg"]
        self.metric_prompt_template = config["prompt_template"]
        self.model_args = config["model_args"]

    def annotate_example(self, data, text):
        prompt = self.metric_prompt_template.format(data=data, text=text)

        messages = [{"role": "user", "content": prompt}]
        data = {"mode": "instruct", "messages": messages, **self.model_args}

        response = requests.post(
            self.API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.API_KEY}",
            },
            json=data,
            verify=False,
        )
        try:
            output_text = response.json()["choices"][0]["message"]["content"]

            # TODO: remove
            # extract JSON by finding the first '{' and last '}'
            output_text = output_text[output_text.find("{") : output_text.rfind("}") + 1]
            j = json.loads(output_text)
            logger.info(j)
            return j
        except:
            print(f"API error message: {response}")


class GPT4Metric(LLMMetric):
    def __init__(self, load_args=None):
        # super().__init__("gpt-3.5-turbo-1106", load_args)
        super().__init__("gpt-4-1106-preview", load_args)
        self.client = OpenAI()

        with open("evaluation/gpt4_metric.yaml") as f:
            config = yaml.safe_load(f)

        self.system_msg = config["system_msg"]
        self.metric_prompt_template = config["prompt_template"]

    def annotate_example(self, data, text):
        try:
            prompt = self.metric_prompt_template.format(data=data, text=text)

            response = self.client.chat.completions.create(
                model=self.metric_name,
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
