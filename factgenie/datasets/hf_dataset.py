#!/usr/bin/env python3
import logging
import os
from datasets import load_dataset

from factgenie.datasets.dataset import Dataset

logger = logging.getLogger("factgenie")


class HFDataset(Dataset):
    @classmethod
    def download(cls, dataset_id, data_download_dir, out_download_dir, splits, outputs, dataset_config, **kwargs):
        hf_id = kwargs.get("hf_id", dataset_id)
        out_column = kwargs.get("out_column", None)

        for split in splits:
            logger.info(f"Downloading {hf_id}/{split}")
            hf_dataset = load_dataset(hf_id, split=split)

            # separate the outputs from the dataset
            if out_column:
                outputs = hf_dataset[out_column]
                hf_dataset = hf_dataset.remove_columns(out_column)
            else:
                outputs = None

            # save the dataset to the data_download_dir
            data_path = data_download_dir / split

            os.makedirs(data_path, exist_ok=True)
            hf_dataset.to_json(data_path / "dataset.jsonl")
            logger.info(f"Saved {hf_id} {split} dataset to {data_path}")

            # save the outputs to the out_download_dir
            if outputs:
                out_path = out_download_dir / split
                os.makedirs(out_path, exist_ok=True)
                outputs.save_to_disk(out_path, format="json")

                logger.info(f"Saved {hf_id} {split} outputs to {out_path}")
