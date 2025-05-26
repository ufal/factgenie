#!/usr/bin/env python3
import json
import logging
import os
import traceback
import zipfile
from pathlib import Path

import requests
from slugify import slugify

from factgenie.datasets.dataset import Dataset
from factgenie.utils import resumable_download

logger = logging.getLogger("factgenie")


class QuintdDataset(Dataset):
    @classmethod
    def download_dataset(
        cls,
        dataset_id,
        data_download_dir,
        splits,
    ):
        if dataset_id == "owid":
            # we need to download a zip file and unpack it
            data_url = "https://github.com/kasnerz/quintd/raw/refs/heads/main/data/quintd-1/data/owid/owid.zip"

            resumable_download(url=data_url, filename=f"{data_download_dir}/owid.zip", force_download=True)
            logger.info(f"Downloaded owid.zip")

            with zipfile.ZipFile(f"{data_download_dir}/owid.zip", "r") as zip_ref:
                zip_ref.extractall(data_download_dir)

            os.remove(f"{data_download_dir}/owid.zip")
            logger.info(f"Unpacked owid.zip")

        else:
            data_url = "https://raw.githubusercontent.com/kasnerz/quintd/refs/heads/main/data/quintd-1/data/{dataset_id}/{split}.json"

            for split in splits:
                resumable_download(
                    url=data_url.format(dataset_id=dataset_id.replace("-", "_"), split=split),
                    filename=f"{data_download_dir}/{split}.json",
                    force_download=False,
                )

                logger.info(f"Downloaded {split}.json")

    @classmethod
    def download_outputs(cls, dataset_id, out_download_dir, splits, outputs):
        extra_ids = {"dev": "direct", "test": "default"}
        output_url = "https://raw.githubusercontent.com/kasnerz/quintd/refs/heads/main/data/quintd-1/outputs/{split}/{dataset_id}/{extra_id}/{setup_id}.json"

        for split in splits:
            for setup_id in outputs:
                if split == "dev" and setup_id == "gpt-3.5":
                    # we do not have these outputs
                    continue

                extra_id = extra_ids[split]
                try:
                    url = output_url.format(
                        dataset_id=dataset_id.replace("-", "_"), split=split, extra_id=extra_id, setup_id=setup_id
                    )

                    j = json.loads(requests.get(url).content)

                    metadata = j["setup"]
                    metadata["model_args"] = metadata.pop("params")
                    metadata["prompt_template"] = metadata.pop("prompt")

                    os.makedirs(out_download_dir / setup_id, exist_ok=True)

                    with open(out_download_dir / setup_id / f"{dataset_id}-{split}.jsonl", "w") as f:
                        for i, gen in enumerate(j["generated"]):
                            record = {
                                "dataset": "quintd1-" + dataset_id,
                                "split": split,
                                "setup_id": setup_id,
                                "example_idx": i,
                                "metadata": metadata,
                                "output": gen["out"],
                            }
                            record["metadata"]["prompt"] = gen["in"]
                            f.write(json.dumps(record) + "\n")

                    logger.info(f"Downloaded outputs for {split}/{setup_id}")
                except Exception as e:
                    traceback.print_exc()
                    logger.warning(f"Could not download output file '{split}/{setup_id}.json'")

    @classmethod
    def download_annotations(cls, dataset_id, annotation_download_dir, splits):
        urls = {
            "quintd1-gpt-4": "https://owncloud.cesnet.cz/index.php/s/qC6NkAhU5M9Ox9U/download",
            "quintd1-human": "https://owncloud.cesnet.cz/index.php/s/IlTdmuhOzxWqEK2/download",
        }

        for campaign_id, url in urls.items():
            if not os.path.exists(f"{annotation_download_dir}/{campaign_id}"):
                os.makedirs(f"{annotation_download_dir}/{campaign_id}")

                resumable_download(
                    url=url, filename=f"{annotation_download_dir}/{campaign_id}.zip", force_download=False
                )
                logger.info(f"Downloaded {campaign_id}.zip")

                with zipfile.ZipFile(f"{annotation_download_dir}/{campaign_id}.zip", "r") as zip_ref:
                    zip_ref.extractall(f"{annotation_download_dir}/{campaign_id}")

                # open campaign_id/metadata.json and change the key `source` to `mode`
                with open(f"{annotation_download_dir}/{campaign_id}/metadata.json", "r") as f:
                    metadata = json.load(f)

                metadata["mode"] = metadata.pop("source")

                # remap original colors to darker ones
                mapping = {
                    "#ffbcbc": "rgb(214, 39, 40)",
                    "#e9d2ff": "rgb(148, 103, 189)",
                    "#fff79f": "rgb(230, 171, 2)",
                    "#bbbbbb": "rgb(102, 102, 102)",
                }

                for category in metadata["config"]["annotation_span_categories"]:
                    category["color"] = mapping[category["color"]]

                with open(f"{annotation_download_dir}/{campaign_id}/metadata.json", "w") as f:
                    json.dump(metadata, f, indent=4)

                # replace gpt-35 with gpt-3-5 (correctly slugified version) in all files
                for file in Path(f"{annotation_download_dir}/{campaign_id}/files").rglob("*.jsonl"):
                    with open(file, "r") as f:
                        lines = f.readlines()

                    with open(file, "w") as f:
                        for line in lines:
                            line = line.replace("gpt-35", "gpt-3-5")
                            j = json.loads(line)

                            record_metadata = metadata["config"].copy()
                            record_metadata["campaign_id"] = campaign_id
                            record_metadata["annotator_id"] = j.pop("annotator_id")
                            record_metadata["annotator_group"] = 0

                            j["metadata"] = record_metadata

                            f.write(json.dumps(j) + "\n")

                os.remove(f"{annotation_download_dir}/{campaign_id}.zip")

            else:
                logger.info(f"Annotations {campaign_id} already downloaded, skipping")

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
        dataset_id = dataset_id.removeprefix("quintd1-")

        cls.download_dataset(dataset_id, data_download_dir, splits)
        cls.download_outputs(dataset_id, out_download_dir, splits, outputs)
        cls.download_annotations(dataset_id, annotation_download_dir, splits)
