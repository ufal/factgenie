#!/usr/bin/env python3
import logging
import requests
import json
import traceback
import zipfile
import os
from slugify import slugify
from factgenie.loaders.dataset import Dataset
from factgenie.utils import resumable_download
from pathlib import Path

logger = logging.getLogger(__name__)


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
                    url=data_url.format(dataset_id=dataset_id, split=split),
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

                out_path = Path(f"{out_download_dir}/{split}/{setup_id}")
                os.makedirs(out_path / "files", exist_ok=True)

                extra_id = extra_ids[split]
                try:
                    url = output_url.format(dataset_id=dataset_id, split=split, extra_id=extra_id, setup_id=setup_id)

                    j = json.loads(requests.get(url).content)

                    with open(f"{out_download_dir}/{split}/{setup_id}/metadata.json", "w") as f:
                        metadata = j["setup"]
                        metadata["model_args"] = metadata.pop("params")
                        metadata["prompt_template"] = metadata.pop("prompt")

                        json.dump(metadata, f, indent=4, ensure_ascii=False)

                    new_out = []
                    for i, gen in enumerate(j["generated"]):
                        record = {
                            "dataset": dataset_id,
                            "split": split,
                            "setup_id": setup_id,
                            "example_idx": i,
                            "in": gen["in"],
                            "out": gen["out"],
                        }
                        new_out.append(record)

                    with open(out_path / "files" / f"{setup_id}.jsonl", "w") as f:
                        for gen in new_out:
                            f.write(json.dumps(gen) + "\n")

                    logger.info(f"Downloaded outputs for {split}/{setup_id}")
                except Exception as e:
                    traceback.print_exc()
                    logger.warning(f"Could not download output file '{split}/{setup_id}.json'")

    @classmethod
    def download_annotations(cls, dataset_id, annotation_download_dir, splits):
        gpt_4_url = "https://owncloud.cesnet.cz/index.php/s/qC6NkAhU5M9Ox9U/download"
        human_url = "https://owncloud.cesnet.cz/index.php/s/IlTdmuhOzxWqEK2/download"

        if not os.path.exists(f"{annotation_download_dir}/quintd1-gpt-4"):
            os.makedirs(f"{annotation_download_dir}/quintd1-gpt-4")

            resumable_download(
                url=gpt_4_url, filename=f"{annotation_download_dir}/quintd1-gpt-4.zip", force_download=False
            )
            logger.info(f"Downloaded quintd1-gpt-4.zip")

            with zipfile.ZipFile(f"{annotation_download_dir}/quintd1-gpt-4.zip", "r") as zip_ref:
                zip_ref.extractall(f"{annotation_download_dir}/quintd1-gpt-4")

            os.remove(f"{annotation_download_dir}/quintd1-gpt-4.zip")
        else:
            logger.info(f"Annotations quintd1-gpt-4 already downloaded, skipping")

        if not os.path.exists(f"{annotation_download_dir}/quintd1-human"):
            os.makedirs(f"{annotation_download_dir}/quintd1-human")

            resumable_download(
                url=human_url, filename=f"{annotation_download_dir}/quintd1-human.zip", force_download=False
            )
            logger.info(f"Downloaded quintd1-human.zip")

            with zipfile.ZipFile(f"{annotation_download_dir}/quintd1-human.zip", "r") as zip_ref:
                zip_ref.extractall(f"{annotation_download_dir}/quintd1-human")

            os.remove(f"{annotation_download_dir}/quintd1-human.zip")
        else:
            logger.info(f"Annotations quintd1-gpt-4 already downloaded, skipping")

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
