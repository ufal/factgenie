#!/usr/bin/env python3
import logging

logger = logging.getLogger(__name__)

from factgenie.loaders.dataset import Dataset
from pathlib import Path
from natsort import natsorted
import requests
import zipfile
import os
import re
import json
import json2table


class BasicDataset(Dataset):
    @classmethod
    def download(cls, dataset_id, data_download_dir, out_download_dir, splits, outputs, dataset_config, **kwargs):
        # default implementation for downloading datasets and outputs using the `data-link` and `outputs-link` fields in the dataset config
        if dataset_config.get("data-link"):
            link = dataset_config["data-link"]
            logger.info(f"Downloading dataset from {link}")
            # download the dataset as a zip file and unpack it
            response = requests.get(link)

            with open(f"{data_download_dir}/{dataset_id}.zip", "wb") as f:
                f.write(response.content)

            logger.info(f"Downloaded {dataset_id}")

            with zipfile.ZipFile(f"{data_download_dir}/{dataset_id}.zip", "r") as zip_ref:
                zip_ref.extractall(data_download_dir)

            os.remove(f"{data_download_dir}/{dataset_id}.zip")
        else:
            raise NotImplementedError("Dataset download not implemented.")

        # outputs are (unlike a dataset) optional
        if dataset_config.get("outputs-link"):
            link = dataset_config["outputs-link"]
            logger.info(f"Downloading outputs from {link}")
            # download the outputs as a zip file and unpack it
            response = requests.get(link)

            with open(f"{out_download_dir}/{dataset_id}.zip", "wb") as f:
                f.write(response.content)

            logger.info(f"Downloaded {dataset_id} outputs")

            with zipfile.ZipFile(f"{out_download_dir}/{dataset_id}.zip", "r") as zip_ref:
                zip_ref.extractall(out_download_dir)

            os.remove(f"{out_download_dir}/{dataset_id}.zip")


class PlainTextDataset(BasicDataset):
    def load_examples(self, split, data_path):
        examples = []

        with open(f"{data_path}/{split}.txt") as f:
            lines = f.readlines()
            for line in lines:
                examples.append(line.strip())

        return examples

    def render(self, example):
        html = "<div>"
        html += "<p>"
        html += example.replace("\\n", "<br>")
        html += "</p>"
        html += "</div>"

        return html


class JSONDataset(BasicDataset):
    def load_examples(self, split, data_path):
        examples = []

        with open(f"{data_path}/{split}.json") as f:
            examples = json.load(f)

        return examples

    def render(self, example):
        # default method, can be overwritten by dataset classes
        html = json2table.convert(
            example,
            build_direction="LEFT_TO_RIGHT",
            table_attributes={
                "class": "table table-sm caption-top meta-table table-responsive font-mono rounded-3 table-bordered"
            },
        )
        return html


class JSONLDataset(BasicDataset):
    def load_examples(self, split, data_path):
        examples = []

        with open(f"{data_path}/{split}.jsonl") as f:
            for line in f:
                examples.append(json.loads(line))

        return examples

    def render(self, example):
        # default method, can be overwritten by dataset classes
        html = json2table.convert(
            example,
            build_direction="LEFT_TO_RIGHT",
            table_attributes={
                "class": "table table-sm caption-top meta-table table-responsive font-mono rounded-3 table-bordered"
            },
        )
        return html


class CSVDataset(BasicDataset):
    def load_examples(self, split, data_path):
        # load the corresponding CSV file
        # each example will be a key-value pair where the key is the column name and the value is the value in the row
        examples = []

        with open(f"{data_path}/{split}.csv") as f:
            lines = f.readlines()
            header = lines[0].strip().split(",")
            for line in lines[1:]:
                values = line.strip().split(",")
                example = dict(zip(header, values))
                examples.append(example)

        return examples

    def render(self, example):
        html = (
            "<table class='table table-sm caption-top meta-table table-responsive font-mono rounded-3 table-bordered'>"
        )
        html += "<caption>Example</caption>"
        html += "<tbody>"

        for key, value in example.items():
            html += f"<tr><th>{key}</th><td>{value}</td></tr>"

        html += "</tbody>"
        html += "</table>"

        return html


class HTMLDataset(BasicDataset):
    def load_examples(self, split, data_path):
        # load the HTML files in the directory, sorted by filename numerically
        examples = []
        split_dir = Path(f"{data_path}/{split}")
        filenames = natsorted(split_dir.iterdir())

        for filename in filenames:
            if filename.suffix == ".html":
                with open(filename) as f:
                    content = f.read()

                    # we need to redirect all the calling for the assets to the correct path
                    # the assets are served by the "/files" endpoint
                    path = f"/files/{self.id}/{split}"

                    # replace the paths in the HTML content
                    content = re.sub(r'src="', f'src="{path}/', content)
                    examples.append(content)

        return examples

    def render(self, example):
        return example