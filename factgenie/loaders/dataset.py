#!/usr/bin/env python3
import logging
import requests
import json
import os
import zipfile
import importlib
import inspect


from pathlib import Path
from collections import defaultdict
from slugify import slugify
from abc import ABC, abstractmethod

from factgenie import DATA_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)


def get_dataset_classes():
    module_name = "factgenie.loaders"
    module = importlib.import_module(module_name)

    classes = {}

    # for each submodule, find all classes subclassing `Dataset`
    for name, obj in inspect.getmembers(module):
        if inspect.ismodule(obj):
            submodule = obj
            for name, obj in inspect.getmembers(submodule):
                if inspect.isclass(obj) and issubclass(obj, Dataset) and obj != Dataset:
                    submodule_name = obj.__module__[len(module_name) + 1 :]
                    classes[f"{submodule_name}.{obj.__name__}"] = obj

    return classes


class Dataset(ABC):
    """
    Abstract class for datasets.
    """

    def __init__(self, dataset_id, **kwargs):
        self.id = dataset_id
        self.data_path = DATA_DIR / self.id
        self.output_path = OUTPUT_DIR / self.id

        self.splits = kwargs.get("splits", ["train", "dev", "test"])
        self.description = kwargs.get("description", "")
        self.type = kwargs.get("type", "default")

        # load data
        self.examples = {}

        for split in self.splits:
            examples = self.load_examples(split=split, data_path=self.data_path)
            examples = self.postprocess_data(examples=examples)

            self.examples[split] = examples

        # load outputs
        self.outputs = self.load_generated_outputs(self.output_path)

    # --------------------------------
    # TODO: implement in subclasses
    # --------------------------------

    @abstractmethod
    def load_examples(self, split, data_path):
        """
        Load the data for the dataset.

        Parameters
        ----------
        split : str
            Split to load the data for.
        data_path : str
            Path to the data directory.

        Returns
        -------
        examples : list
            List of examples for the given split.
        """
        pass

    @abstractmethod
    def render(self, example):
        """
        Render the example in HTML.

        Parameters
        ----------
        example : dict
            Example to render.

        Returns
        -------
        html : str
            HTML representation of the example.

        """
        pass

    @classmethod
    def download(
        cls,
        dataset_id,
        data_download_dir,
        out_download_dir,
        annotation_download_dir,
        splits,
        outputs,
        dataset_config,
        **kwargs,
    ):
        """
        Download the dataset (optionally along with model outputs and annotations) from an external source.

        Does not need to be implemented if the dataset is added locally into the `data` directory.

        Parameters
        ----------
        dataset_id : str
            ID of the dataset.
        data_download_dir : str
            Path to the directory where the dataset should be downloaded.
        out_download_dir : str
            Path to the directory where the outputs should be downloaded (optional).
        annotation_download_dir : str
            Path to the directory where the annotations should be downloaded (optional).
        splits : list
            List of splits to download.
        outputs : list
            List of outputs to download.
        dataset_config : dict
            Dataset configuration.
        """
        pass

    # --------------------------------
    # end TODO
    # --------------------------------

    def load_generated_outputs(self, output_path):
        """
        Load the generated outputs for the dataset.

        Parameters
        ----------
        output_path : str
            Path to the output directory.

        Returns
        -------
        outputs : dict
            Dictionary with the generated outputs for each split and setup, e.g. {"train": {"setup1": output1, "setup2": output2}, "test": {"setup1": output1, "setup2": output2}}.
        """
        outputs = defaultdict(dict)

        # find recursively all JSONL files in the output directory
        outs = list(Path(output_path).glob("**/*.jsonl"))

        for out in outs:
            with open(out) as f:
                for line in f:
                    j = json.loads(line)

                    split = j["split"]
                    setup_id = j["setup_id"]
                    example_idx = j["example_idx"]

                    if split not in outputs:
                        outputs[split] = {}

                    if setup_id not in outputs[split]:
                        outputs[split][setup_id] = {}

                    outputs[split][setup_id][example_idx] = j

                logger.info(f"Loaded output file: {out}")

        return outputs

    def postprocess_data(self, examples):
        """
        Postprocess the data after loading.

        Parameters
        ----------
        examples : dict
            Dictionary with a list of examples for each split, e.g. {"train": [ex1, ex2, ...], "test": [ex1, ex2, ...]}.
        """
        return examples

    def get_outputs_for_split(self, split):
        """
        Get the list of generated outputs for the given split.
        """
        return self.outputs[split]

    def get_output_for_idx_by_setup(self, split, output_idx, setup_id):
        """
        Get the generated output for the given split, output index, and setup ID.
        """
        if setup_id in self.outputs[split]:
            model_out = self.outputs[split][setup_id]

            if output_idx in model_out:
                return model_out[output_idx]["out"]

        logger.warning(f"No output found for {setup_id=}, {output_idx=}, {split=}")
        return None

    def get_outputs_for_idx(self, split, output_idx):
        """
        Get the generated outputs for the given split and output index.
        """
        outs_all = []

        for outs in self.outputs[split].values():
            if output_idx in outs:
                outs_all.append(outs[output_idx])

        return outs_all

    def get_example(self, split, example_idx):
        """
        Get the example at the given index for the given split.
        """
        example = self.examples[split][example_idx]

        return example

    def get_example_count(self, split=None):
        """
        Get the number of examples in the dataset.
        """
        if split is None:
            return sum([len(exs) for exs in self.examples.values()])

        return len(self.examples[split])

    def get_splits(self):
        """
        Get the list of splits in the dataset.
        """
        return self.splits

    def get_description(self):
        """
        Get the string with description of the dataset.
        """
        return self.description

    def get_type(self):
        """
        Get the type of the dataset displayed in the web interface.

        Returns
        -------
        type : str
            Type of the dataset: {"default", "json", "text", "table"}
        """

        return self.type
