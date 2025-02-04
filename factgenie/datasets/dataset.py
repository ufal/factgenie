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

from factgenie import INPUT_DIR, OUTPUT_DIR

logger = logging.getLogger("factgenie")


def get_dataset_classes():
    module_name = "factgenie.datasets"
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
        self.data_path = INPUT_DIR / self.id

        self.splits = kwargs.get("splits", ["train", "dev", "test"])
        self.description = kwargs.get("description", "")

        # load data
        self.examples = {}

        for split in self.splits:
            examples = self.load_examples(split=split, data_path=self.data_path)
            examples = self.postprocess_data(examples=examples)

            self.examples[split] = examples

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

    def postprocess_data(self, examples):
        """
        Postprocess the data after loading.

        Parameters
        ----------
        examples : dict
            Dictionary with a list of examples for each split, e.g. {"train": [ex1, ex2, ...], "test": [ex1, ex2, ...]}.
        """
        return examples

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
