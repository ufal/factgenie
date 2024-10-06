#!/usr/bin/env python3
import os
import datetime
import json
import glob
import time
import logging
from typing import Dict
import pandas as pd
import random
import time
import coloredlogs
import traceback
import yaml
import queue
import shutil
import inspect
import importlib
import zipfile
import markdown
import traceback

import urllib
from tqdm import tqdm

from io import BytesIO
from slugify import slugify
from flask import jsonify, make_response
from collections import defaultdict
from pathlib import Path
from factgenie.campaigns import (
    HumanCampaign,
    LLMCampaignEval,
    ExternalCampaign,
    LLMCampaignGen,
    CampaignStatus,
    ExampleStatus,
)
from jinja2 import Template

from factgenie import (
    ANNOTATIONS_DIR,
    GENERATIONS_DIR,
    OUTPUT_DIR,
    DATA_DIR,
    TEMPLATES_DIR,
    LLM_EVAL_CONFIG_DIR,
    LLM_GEN_CONFIG_DIR,
    CROWDSOURCING_CONFIG_DIR,
    RESOURCES_CONFIG_PATH,
    DATASET_CONFIG_PATH,
    PREVIEW_STUDY_ID,
)

file_handler = logging.FileHandler("error.log")
file_handler.setLevel(logging.ERROR)


logger = logging.getLogger(__name__)


# https://maxhalford.github.io/blog/flask-sse-no-deps/
class MessageAnnouncer:
    def __init__(self):
        self.listeners = []

    def listen(self):
        self.listeners.append(queue.Queue(maxsize=5))
        return self.listeners[-1]

    def announce(self, msg):
        # We go in reverse order because we might have to delete an element, which will shift the
        # indices backward
        for i in reversed(range(len(self.listeners))):
            try:
                self.listeners[i].put_nowait(msg)
            except queue.Full:
                del self.listeners[i]


def format_sse(data: str, event=None) -> str:
    """Formats a string and an event name in order to follow the event stream convention.

    >>> format_sse(data=json.dumps({'abc': 123}), event='Jackson 5')
    'event: Jackson 5\\ndata: {"abc": 123}\\n\\n'

    """
    msg = f"data: {data}\n\n"
    if event is not None:
        msg = f"event: {event}\n{msg}"
    return msg


def success():
    resp = jsonify(success=True)
    return resp


def error(j):
    resp = jsonify(success=False, error=j)
    return resp


def get_dataset(app, dataset_id):
    return app.db["datasets_obj"].get(dataset_id)


def load_configs(mode):
    """
    Goes through all the files in the LLM_CONFIG_DIR
    instantiate the LLMMetric class
    and inserts the object in the metrics dictionary

    Returns:
        metrics: dictionary of LLMMetric objects with keys of metric names
    """
    configs = {}

    if mode == "llm_eval":
        config_dir = LLM_EVAL_CONFIG_DIR
    elif mode == "llm_gen":
        config_dir = LLM_GEN_CONFIG_DIR
    else:
        config_dir = CROWDSOURCING_CONFIG_DIR

    for file in os.listdir(config_dir):
        if file.endswith(".yaml"):
            try:
                with open(config_dir / file) as f:
                    config = yaml.safe_load(f)
                    configs[file] = config
            except Exception as e:
                logger.error(f"Error while loading metric {file}")
                traceback.print_exc()
                continue

    return configs


def load_campaign(app, campaign_id, mode):
    campaign_index = generate_campaign_index(app, force_reload=False)

    if campaign_id in campaign_index[mode]:
        return campaign_index[mode][campaign_id]

    if mode == "llm_eval":
        campaign = LLMCampaignEval(campaign_id=campaign_id)
    elif mode == "llm_gen":
        campaign = LLMCampaignGen(campaign_id=campaign_id)
    elif mode == "crowdsourcing":
        campaign = HumanCampaign(campaign_id=campaign_id)

    campaign_index[mode][campaign_id] = campaign

    return campaign


def generate_campaign_index(app, force_reload=True):
    if not force_reload and "campaign_index" in app.db:
        return app.db["campaign_index"]

    campaigns = defaultdict(dict)

    # find all subdirs in CROWDSOURCING_DIR
    for directory in [ANNOTATIONS_DIR, GENERATIONS_DIR]:
        for campaign_dir in Path(directory).iterdir():
            try:
                if not campaign_dir.is_dir():
                    continue

                metadata = json.load(open(campaign_dir / "metadata.json"))
                campaign_source = metadata.get("source")
                campaign_id = metadata["id"]

                if campaign_source == "crowdsourcing":
                    campaign = HumanCampaign(campaign_id=campaign_id)
                elif campaign_source == "llm_eval":
                    campaign = LLMCampaignEval(campaign_id=campaign_id)
                elif campaign_source == "llm_gen":
                    campaign = LLMCampaignGen(campaign_id=campaign_id)
                elif campaign_source == "external":
                    campaign = ExternalCampaign(campaign_id=campaign_id)
                elif campaign_source == "hidden":
                    continue
                else:
                    logger.warning(f"Unknown campaign source: {campaign_source}")
                    continue

                campaigns[campaign_source][campaign_id] = campaign
            except:
                traceback.print_exc()
                logger.error(f"Error while loading campaign {campaign_dir}")

    app.db["campaign_index"] = campaigns

    return app.db["campaign_index"]


def get_sorted_campaign_list(app, sources):
    campaign_index = generate_campaign_index(app, force_reload=True)

    campaigns = []
    for source in sources:
        campaigns.extend(campaign_index[source].values())

    campaigns.sort(key=lambda x: x.metadata["created"], reverse=True)
    campaigns = {
        c.metadata["id"]: {"metadata": c.metadata, "stats": c.get_stats(), "data": c.db.to_dict(orient="records")}
        for c in campaigns
    }
    return campaigns


def load_annotations_for_campaign(subdir):
    annotations_campaign = defaultdict(list)

    # find metadata for the campaign
    metadata_path = ANNOTATIONS_DIR / subdir / "metadata.json"
    if not metadata_path.exists():
        return None

    with open(metadata_path) as f:
        metadata = json.load(f)

    if metadata["source"] == "hidden":
        return None

    jsonl_files = (ANNOTATIONS_DIR / subdir / "files").glob("*.jsonl")

    for jsonl_file in jsonl_files:
        with open(jsonl_file) as f:
            for line in f:
                annotation = json.loads(line)
                annotation["metadata"] = metadata

                key = (
                    slugify(annotation["dataset"]),
                    slugify(annotation["split"]),
                    annotation["example_idx"],
                    slugify(annotation["setup_id"]),
                )
                annotations_campaign[key].append(annotation)

    return annotations_campaign


def generate_annotation_index(app):
    # contains annotations for each generated output
    annotations = defaultdict(list)

    # for all subdirectories in ANNOTATIONS_DIR, load content of all the jsonl files
    for subdir in os.listdir(ANNOTATIONS_DIR):
        try:
            annotations_campaign = load_annotations_for_campaign(subdir)

            if annotations_campaign is None:
                continue

            for key, annotation_set in annotations_campaign.items():
                annotations[key].extend(annotation_set)
        except:
            # if app.config["debug"]:
            traceback.print_exc()
            logger.error(f"Error while loading annotations for {subdir}")

    app.db["annotation_index"] = annotations
    return annotations


def export_campaign_outputs(app, mode, campaign_id):
    target_dir = GENERATIONS_DIR if mode == "llm_gen" else ANNOTATIONS_DIR

    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for root, _dirs, files in os.walk(os.path.join(target_dir, campaign_id)):
            for file in files:
                zip_file.write(
                    os.path.join(root, file),
                    os.path.relpath(os.path.join(root, file), os.path.join(target_dir, campaign_id)),
                )

    # Set response headers for download
    timestamp = int(time.time())
    response = make_response(zip_buffer.getvalue())
    response.headers["Content-Type"] = "application/zip"
    response.headers["Content-Disposition"] = f"attachment; filename={campaign_id}_{timestamp}.zip"
    return response


def get_annotations(app, dataset_id, split, example_idx, setup_id):
    annotation_index = app.db["annotation_index"]
    key = (slugify(dataset_id), slugify(split), example_idx, slugify(setup_id))

    return annotation_index.get(key, [])


def get_example_data(app, dataset_id, split, example_idx):
    dataset = get_dataset(app=app, dataset_id=dataset_id)

    example = dataset.get_example(split=split, example_idx=example_idx)
    html = dataset.render(example=example)

    # temporary solution for external files
    # prefix all the "/files" calls with "app.config["host_prefix"]"
    html = html.replace('src="/files', f'src="{app.config["host_prefix"]}/files')

    generated_outputs = dataset.get_outputs_for_idx(split=split, output_idx=example_idx)

    for i, output in enumerate(generated_outputs):
        setup_id = output["setup_id"]
        annotations = get_annotations(app, dataset_id, split, example_idx, setup_id)

        generated_outputs[i]["annotations"] = annotations

    return {
        "html": html,
        "raw_data": example,
        "total_examples": dataset.get_example_count(split),
        "generated_outputs": generated_outputs,
    }


def get_model_outputs_overview(app, datasets, non_empty=False):
    model_outputs = {}

    for dataset_id, dataset_config in datasets.items():
        dataset = get_dataset(app=app, dataset_id=dataset_id)
        splits = dataset.get_splits()

        model_outputs[dataset_id] = {}

        for split in splits:
            outputs = dataset.get_outputs_for_split(split)

            # extract all key values from the outputs
            model_outputs[dataset_id][split] = {setup_id: len(output) for setup_id, output in outputs.items()}

    if non_empty:
        for dataset_id, splits in model_outputs.items():
            model_outputs[dataset_id] = {k: v for k, v in splits.items() if v}

        model_outputs = {k: v for k, v in model_outputs.items() if v}

    return model_outputs


def free_idle_examples(db):
    start = int(time.time())

    # check if there are annotations which are idle for more than 2 hours: set them to free
    idle_examples = db[(db["status"] == ExampleStatus.ASSIGNED) & (db["start"] < start - 2 * 60 * 60)]
    for i in idle_examples.index:
        db.loc[i, "status"] = ExampleStatus.FREE
        db.loc[i, "start"] = ""
        db.loc[i, "end"] = ""
        db.loc[i, "annotator_id"] = ""

    return db


def select_batch_idx(db, seed):
    free_examples = db[db["status"] == ExampleStatus.FREE]
    assigned_examples = db[db["status"] == ExampleStatus.ASSIGNED]

    if len(free_examples) == 0 and len(assigned_examples) == 0:
        raise ValueError("No examples available")

    # if no free examples but still assigned examples, take the oldest assigned example
    if len(free_examples) == 0 and len(assigned_examples) > 0:
        free_examples = assigned_examples
        free_examples = free_examples.sort_values(by=["start"])
        free_examples = free_examples.head(1)

        logger.info(f"Annotating extra example {free_examples.index[0]}")

    example = free_examples.sample(random_state=seed)
    batch_idx = int(example.batch_idx.values[0])
    logger.info(f"Selecting batch {batch_idx}")

    return batch_idx


def get_annotator_batch(app, campaign, db, service_ids):
    # simple locking over the CSV file to prevent double writes
    with app.db["lock"]:
        annotator_id = service_ids["annotator_id"]

        logging.info(f"Acquiring lock for {annotator_id}")
        start = int(time.time())

        seed = random.seed(str(start) + str(service_ids.values()))

        try:
            batch_idx = select_batch_idx(db, seed)
        except ValueError:
            # no available batches
            return []

        if annotator_id != PREVIEW_STUDY_ID:
            db = free_idle_examples(db)

            # update the CSV
            db.loc[db["batch_idx"] == batch_idx, "status"] = ExampleStatus.ASSIGNED
            db.loc[db["batch_idx"] == batch_idx, "start"] = start
            db.loc[db["batch_idx"] == batch_idx, "annotator_id"] = annotator_id

            campaign.update_db(db)

        annotator_batch = campaign.get_examples_for_batch(batch_idx)

        for example in annotator_batch:
            example.update(
                {"campaign_id": campaign.campaign_id, "batch_idx": batch_idx, "start_timestamp": start, **service_ids}
            )

        logging.info(f"Releasing lock for {annotator_id}")

    return annotator_batch


def generate_llm_campaign_db(mode, datasets, campaign_id, campaign_data):
    # load all outputs
    all_examples = []

    for c in campaign_data:
        dataset = datasets[c["dataset"]]
        for i in range(dataset.get_example_count(c["split"])):
            record = {
                "dataset": c["dataset"],
                "split": c["split"],
                "example_idx": i,
            }
            if mode == "llm_eval":
                record["setup_id"] = c["setup_id"]
            else:
                # setup_id based on the generation model
                record["setup_id"] = campaign_id

            all_examples.append(record)

    df = pd.DataFrame.from_records(all_examples)

    # create a column for batch index and assign each example to a batch
    df["annotator_group"] = 0
    df["annotator_id"] = ""
    df["status"] = ExampleStatus.FREE
    df["start"] = ""

    return df


def generate_campaign_db(app, campaign_data, config):
    # load all outputs
    all_examples = []

    examples_per_batch = config["examples_per_batch"]
    sort_order = config["sort_order"]

    for c in campaign_data:
        dataset = app.db["datasets_obj"][c["dataset"]]
        for i in range(dataset.get_example_count(c["split"])):
            all_examples.append(
                {
                    "dataset": c["dataset"],
                    "split": c["split"],
                    "example_idx": i,
                    "setup_id": c["setup_id"],
                }
            )

    random.seed(42)

    # current flags:
    # - shuffle-all: shuffle all examples and setups
    # - sort-example-ids-shuffle-setups: sort examples by example_idx, shuffle setups
    # - sort-example-ids-keep-setups: sort examples by example_idx, keep the setup order
    # - keep-all: keep all examples and setups in the default order
    # we are also still supporting the old "example-level" and "dataset-level" flags

    if sort_order == "dataset-level" or sort_order == "shuffle-all":
        random.shuffle(all_examples)
    elif sort_order == "example-level" or sort_order == "sort-example-ids-shuffle-setups":
        random.shuffle(all_examples)
        all_examples = sorted(all_examples, key=lambda x: (x["example_idx"], x["dataset"], x["split"]))
    elif sort_order == "sort-example-ids-keep-setups":
        all_examples = sorted(all_examples, key=lambda x: (x["example_idx"], x["dataset"], x["split"]))
    elif sort_order == "keep-all":
        pass
    else:
        raise ValueError(f"Unknown sort order {sort_order}")

    df = pd.DataFrame.from_records(all_examples)

    # create a column for batch index and assign each example to a batch
    df["batch_idx"] = df.index // examples_per_batch
    df["annotator_group"] = 0
    df["annotator_id"] = ""
    df["status"] = ExampleStatus.FREE
    df["start"] = ""
    df["end"] = ""

    return df


def load_resources_config():
    with open(RESOURCES_CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    return config


def load_dataset_config():
    if not DATASET_CONFIG_PATH.exists():
        with open(DATASET_CONFIG_PATH, "w") as f:
            f.write("---\n")

    with open(DATASET_CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    return config


def save_dataset_config(config):
    with open(DATASET_CONFIG_PATH, "w") as f:
        yaml.dump(config, f, indent=2, allow_unicode=True)


def set_dataset_enabled(app, dataset_id, enabled):
    config = load_dataset_config()
    config[dataset_id]["enabled"] = enabled

    if enabled:
        dataset = instantiate_dataset(dataset_id, config[dataset_id])
        app.db["datasets_obj"][dataset_id] = dataset
    else:
        app.db["datasets_obj"].pop(dataset_id, None)

    save_dataset_config(config)


def get_local_dataset_overview(app):
    config = load_dataset_config()
    overview = {}

    for dataset_id, dataset_config in config.items():
        class_name = dataset_config["class"]
        params = dataset_config.get("params", {})
        is_enabled = dataset_config.get("enabled", True)
        description = dataset_config.get("description", "")
        splits = dataset_config.get("splits", [])
        dataset_type = dataset_config.get("type", "default")

        if is_enabled:
            dataset = app.db["datasets_obj"].get(dataset_id)

            if dataset is None:
                logger.warning(f"Dataset {dataset_id} is enabled but not loaded")
                continue

            dataset.outputs = dataset.load_generated_outputs(dataset.output_path)

            example_count = {split: dataset.get_example_count(split) for split in dataset.get_splits()}
        else:
            example_count = {}

        overview[dataset_id] = {
            "class": class_name,
            "params": params,
            "enabled": is_enabled,
            "splits": splits,
            "description": description,
            "example_count": example_count,
            "type": dataset_type,
        }

    return overview


def get_datasets_for_download(app):
    config = load_resources_config()

    return config


def download_dataset(app, dataset_id):
    config = load_resources_config()
    dataset_config = config.get(dataset_id)

    if dataset_config is None:
        raise ValueError(f"Dataset {dataset_id} not found in the download config")

    submodule, class_name = dataset_config["class"].split(".")

    dataset_cls = get_dataset_class(submodule, class_name)
    download_dir = DATA_DIR / dataset_id
    output_dir = OUTPUT_DIR / dataset_id
    annotations_dir = ANNOTATIONS_DIR

    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    dataset_cls.download(
        dataset_id=dataset_id,
        data_download_dir=download_dir,
        out_download_dir=output_dir,
        annotation_download_dir=annotations_dir,
        splits=dataset_config["splits"],
        outputs=dataset_config.get("outputs", []),
        dataset_config=dataset_config,
    )

    # add an entry in the dataset config
    config = load_dataset_config()

    config[dataset_id] = {
        "class": dataset_config["class"],
        "description": dataset_config.get("description", ""),
        "splits": dataset_config["splits"],
        "enabled": True,
    }

    dataset = instantiate_dataset(dataset_id, config[dataset_id])
    app.db["datasets_obj"][dataset_id] = dataset

    save_dataset_config(config)

    return dataset


def delete_dataset(app, dataset_id):
    config = load_dataset_config()
    config.pop(dataset_id, None)
    save_dataset_config(config)

    # remove the data directory
    shutil.rmtree(f"factgenie/data/{dataset_id}", ignore_errors=True)

    app.db["datasets_obj"].pop(dataset_id, None)


def export_dataset(app, dataset_id):
    zip_buffer = BytesIO()
    data_path = DATA_DIR / dataset_id

    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for root, dirs, files in os.walk(data_path):
            for file in files:
                zip_file.write(
                    os.path.join(root, file),
                    os.path.relpath(os.path.join(root, file), data_path),
                )

    # Set response headers for download
    response = make_response(zip_buffer.getvalue())
    response.headers["Content-Type"] = "application/zip"
    response.headers["Content-Disposition"] = f"attachment; filename={dataset_id}.zip"

    return response


def export_outputs(app, dataset_id, split, setup_id):
    zip_buffer = BytesIO()
    output_path = OUTPUT_DIR / dataset_id / split / setup_id

    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for root, dirs, files in os.walk(output_path):
            for file in files:
                zip_file.write(
                    os.path.join(root, file),
                    os.path.relpath(os.path.join(root, file), output_path),
                )

    # Set response headers for download
    response = make_response(zip_buffer.getvalue())
    response.headers["Content-Type"] = "application/zip"
    response.headers["Content-Disposition"] = f"attachment; filename={dataset_id}_{split}_{setup_id}.zip"

    return response


def get_dataset_class(submodule, class_name):
    # Dynamically import the class
    module = importlib.import_module("factgenie.loaders")
    submodule = getattr(module, submodule)
    dataset_class = getattr(submodule, class_name)

    return dataset_class


def instantiate_dataset(dataset_id, dataset_config):
    submodule, class_name = dataset_config["class"].split(".")

    dataset_class = get_dataset_class(submodule, class_name)

    return dataset_class(dataset_id, **dataset_config)


def instantiate_datasets():
    config = load_dataset_config()
    datasets = {}

    for dataset_id, dataset_config in config.items():
        is_enabled = dataset_config.get("enabled", True)

        if not is_enabled:
            continue

        datasets[dataset_id] = instantiate_dataset(dataset_id, dataset_config)

    return datasets


def upload_dataset(app, dataset_id, dataset_description, dataset_format, dataset_data):
    params = {
        "text": {"suffix": "txt", "class": "basic.PlainTextDataset", "type": "default"},
        "jsonl": {"suffix": "jsonl", "class": "basic.JSONLDataset", "type": "json"},
        "csv": {"suffix": "csv", "class": "basic.CSVDataset", "type": "table"},
        "html": {"suffix": "zip", "class": "basic.HTMLDataset", "type": "default"},
    }
    dataset_id = slugify(dataset_id)
    data_dir = f"factgenie/data/{dataset_id}"
    os.makedirs(data_dir, exist_ok=True)

    if dataset_format in ["text", "jsonl", "csv"]:
        # save each split in a separate file
        for split, data in dataset_data.items():
            with open(f"{data_dir}/{split}.{params[dataset_format]['suffix']}", "w") as f:
                f.write(data)

    elif dataset_format == "html":
        # dataset_data is the file object
        for split, data in dataset_data.items():
            binary_file = BytesIO(bytes(data))

            # check if there is at least one HTML file in the top-level directory
            with zipfile.ZipFile(binary_file, "r") as zip_ref:
                for file in zip_ref.namelist():
                    if file.endswith(".html") and "/" not in file:
                        break
                else:
                    raise ValueError("No HTML files found in the zip archive")

                zip_ref.extractall(f"{data_dir}/{split}")

    # add an entry in the dataset config
    config = load_dataset_config()
    config[dataset_id] = {
        "class": params[dataset_format]["class"],
        "description": dataset_description,
        "splits": list(dataset_data.keys()),
        "enabled": True,
    }
    save_dataset_config(config)

    app.db["datasets_obj"][dataset_id] = instantiate_dataset(dataset_id, config[dataset_id])


def upload_model_outputs(dataset, split, setup_id, model_outputs):
    path = Path(f"{dataset.output_path}/{split}/{setup_id}/files")
    path.mkdir(parents=True, exist_ok=True)

    generated = model_outputs.strip().split("\n")
    setup_id = slugify(setup_id)

    if setup_id in dataset.outputs[split]:
        raise ValueError(f"Output for {setup_id} already exists in {split}")

    if len(generated) != len(dataset.examples[split]):
        raise ValueError(
            f"Output count mismatch for {setup_id} in {split}: {len(generated)} vs {len(dataset.examples[split])}"
        )

    dataset.outputs[split][setup_id] = {}

    with open(f"{path}/{setup_id}.jsonl", "w") as f:
        for i, out in enumerate(generated):
            j = {
                "dataset": dataset.id,
                "split": split,
                "setup_id": setup_id,
                "example_idx": i,
                "out": out,
            }
            f.write(json.dumps(j) + "\n")

            dataset.outputs[split][setup_id][i] = j

    with open(f"{path.parent}/metadata.json", "w") as f:
        json.dump(
            {
                "id": setup_id,
                "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=4,
        )


def delete_model_outputs(dataset, split, setup_id):
    path = Path(f"{dataset.output_path}/{split}/{setup_id}")

    if path.exists():
        shutil.rmtree(path)

    dataset.outputs[split].pop(setup_id, None)


def llm_campaign_new(mode, campaign_id, config, campaign_data, datasets, overwrite=False):
    campaign_id = slugify(campaign_id)

    target_dir = GENERATIONS_DIR if mode == "llm_gen" else ANNOTATIONS_DIR

    # create a new directory
    if os.path.exists(os.path.join(target_dir, campaign_id)):
        if not overwrite:
            raise ValueError(f"Campaign {campaign_id} already exists")
        else:
            shutil.rmtree(os.path.join(target_dir, campaign_id))

    os.makedirs(os.path.join(target_dir, campaign_id, "files"), exist_ok=True)

    # create the annotation CSV
    db = generate_llm_campaign_db(mode, datasets, campaign_id, campaign_data)
    db_path = os.path.join(target_dir, campaign_id, "db.csv")
    logger.info(f"DB with {len(db)} free examples created for {campaign_id} at {db_path}")
    db.to_csv(db_path, index=False)

    # save metadata
    metadata_path = os.path.join(target_dir, campaign_id, "metadata.json")
    logger.info(f"Metadata for {campaign_id} saved at {metadata_path}")
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "id": campaign_id,
                "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": mode,
                "status": CampaignStatus.IDLE,
                "config": config,
            },
            f,
            indent=4,
        )

    # create the campaign object
    if mode == "llm_eval":
        campaign = LLMCampaignEval(campaign_id=campaign_id)
    elif mode == "llm_gen":
        campaign = LLMCampaignGen(campaign_id=campaign_id)

    return campaign


def generate_default_id(campaign_index, prefix):
    i = 1
    default_campaign_id = f"{prefix}-{i}"
    while default_campaign_id in campaign_index:
        default_campaign_id = f"{prefix}-{i}"
        i += 1

    return default_campaign_id


def get_service_ids(service, args):
    # we always need to have at least the annotator_id
    service_ids = {
        "annotator_id": None,
    }
    if service == "local":
        service_ids["annotator_id"] = args.get("annotatorId", PREVIEW_STUDY_ID)
    elif service == "prolific":
        service_ids["annotator_id"] = args.get("PROLIFIC_PID", PREVIEW_STUDY_ID)
        service_ids["session_id"] = args.get("SESSION_ID", PREVIEW_STUDY_ID)
        service_ids["study_id"] = args.get("STUDY_ID", PREVIEW_STUDY_ID)
    elif service == "mturk":
        service_ids["annotator_id"] = args.get("workerId", PREVIEW_STUDY_ID)
        service_ids["session_id"] = args.get("assignmentId", PREVIEW_STUDY_ID)
        service_ids["study_id"] = args.get("hitId", PREVIEW_STUDY_ID)
    else:
        raise ValueError(f"Unknown service {service}")

    return service_ids


def check_login(app, username, password):
    c_username = app.config["login"]["username"]
    c_password = app.config["login"]["password"]
    assert isinstance(c_username, str) and isinstance(
        c_password, str
    ), "Invalid login credentials 'username' and 'password' should be strings. Escape them with quotes in the yaml config."
    return username == c_username and password == c_password


def save_annotation(annotator_id, campaign_id, dataset_id, split, setup_id, example_idx, annotation_set, start_time):
    save_dir = os.path.join(ANNOTATIONS_DIR, campaign_id, "files")
    os.makedirs(save_dir, exist_ok=True)

    annotation = {
        "annotator_group": 0,
        "annotator_id": annotator_id,
        "dataset": dataset_id,
        "setup_id": setup_id,
        "split": split,
        "example_idx": example_idx,
        "annotations": annotation_set,
    }

    # save the annotation
    with open(os.path.join(save_dir, f"{annotator_id}-{dataset_id}-{split}-{start_time}.jsonl"), "a") as f:
        f.write(json.dumps(annotation) + "\n")
    return annotation


def save_output(campaign_id, dataset_id, split, example_idx, output, start_time):
    save_dir = os.path.join(GENERATIONS_DIR, campaign_id, "files")
    os.makedirs(save_dir, exist_ok=True)
    prompt, generated = output.get("prompt"), output.get("output")

    # save the output
    record = {
        "dataset": dataset_id,
        "split": split,
        "setup_id": campaign_id,
        "example_idx": example_idx,
        "in": prompt,
        "out": generated,
    }

    with open(os.path.join(save_dir, f"{dataset_id}-{split}-{start_time}.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")

    return record


def duplicate_eval(app, mode, campaign_id, new_campaign_id):
    target_dir = GENERATIONS_DIR if mode == "llm_gen" else ANNOTATIONS_DIR

    # copy the directory except for the annotations
    old_campaign_dir = os.path.join(target_dir, campaign_id)
    new_campaign_dir = os.path.join(target_dir, new_campaign_id)

    # if new campaign dir exists, return error
    if os.path.exists(new_campaign_dir):
        return error("Campaign already exists")

    shutil.copytree(old_campaign_dir, new_campaign_dir, ignore=shutil.ignore_patterns("files"))

    # copy the db
    old_db = pd.read_csv(os.path.join(old_campaign_dir, "db.csv"))
    new_db = old_db.copy()
    new_db["status"] = ExampleStatus.FREE

    # clean the columns
    new_db["annotator_id"] = ""
    new_db["start"] = ""
    new_db["end"] = ""

    new_db.to_csv(os.path.join(new_campaign_dir, "db.csv"), index=False)

    # update the metadata
    metadata_path = os.path.join(new_campaign_dir, "metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)

    metadata["id"] = new_campaign_id
    metadata["created"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata["status"] = CampaignStatus.IDLE

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    # if it is a human campaign, copy also the templates
    if metadata["source"] == "crowdsourcing":
        shutil.copytree(
            os.path.join(TEMPLATES_DIR, "campaigns", campaign_id),
            os.path.join(TEMPLATES_DIR, "campaigns", new_campaign_id),
        )

    return success()


def save_config(filename, config, mode):
    # https://github.com/yaml/pyyaml/issues/121#issuecomment-1018117110
    def yaml_multiline_string_pipe(dumper, data):
        text_list = [line.rstrip() for line in data.splitlines()]
        fixed_data = "\n".join(text_list)
        if len(text_list) > 1:
            return dumper.represent_scalar("tag:yaml.org,2002:str", fixed_data, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", fixed_data)

    yaml.add_representer(str, yaml_multiline_string_pipe)

    if mode == "llm_eval":
        save_dir = LLM_EVAL_CONFIG_DIR
    elif mode == "llm_gen":
        save_dir = LLM_GEN_CONFIG_DIR
    else:
        save_dir = CROWDSOURCING_CONFIG_DIR

    with open(os.path.join(save_dir, filename), "w") as f:
        yaml.dump(config, f, indent=2, allow_unicode=True)


def parse_llm_eval_config(config):
    config = {
        "type": config.get("metricType"),
        "model": config.get("modelName"),
        "prompt_template": config.get("promptTemplate"),
        "system_msg": config.get("systemMessage"),
        "api_url": config.get("apiUrl"),
        "model_args": config.get("modelArguments"),
        "extra_args": config.get("extraArguments"),
        "annotation_span_categories": config.get("annotationSpanCategories"),
    }
    return config


def parse_llm_gen_config(config):
    config = {
        "type": config.get("metricType"),
        "model": config.get("modelName"),
        "prompt_template": config.get("promptTemplate"),
        "system_msg": config.get("systemMessage"),
        "start_with": config.get("startWith"),
        "api_url": config.get("apiUrl"),
        "model_args": config.get("modelArguments"),
        "extra_args": config.get("extraArguments"),
    }
    return config


def parse_crowdsourcing_config(config):
    config = {
        "annotator_instructions": config.get("annotatorInstructions"),
        "has_display_overlay": config.get("hasDisplayOverlay"),
        "final_message": config.get("finalMessage"),
        "examples_per_batch": int(config.get("examplesPerBatch")),
        "idle_time": int(config.get("idleTime")),
        "annotation_granularity": config.get("annotationGranularity"),
        "service": config.get("service"),
        "sort_order": config.get("sortOrder"),
        "annotation_span_categories": config.get("annotationSpanCategories"),
        "flags": config.get("flags"),
        "options": config.get("options"),
    }

    return config


def generate_checkboxes(flags):
    if not flags:
        return ""

    flags_segment = "<div class='mb-4'><p><b>Please check if you agree with any of the following statements:</b></p>"
    for i, flag in enumerate(flags):
        flags_segment += f"""
            <div class="form-check flag-checkbox">
                <input class="form-check-input" type="checkbox" value="{i}" id="checkbox-{i}">
                <label class="form-check-label" for="checkbox-{i}">
                    {flag}
                </label>
            </div>
        """
    flags_segment += "</div>"

    return flags_segment


def generate_options(options):
    if not options:
        return ""

    options_segment = "<div class='mt-2 mb-3'>"
    for i, option in enumerate(options):
        if option["type"] == "select":
            options_segment += f"""
                <div class="form-group crowdsourcing-option option-select mb-4">
                    <div><label for="select-{i}">{option["label"]}</label></div>
                    <select class="form-select select-crowdsourcing mb-1" id="select-crowdsourcing-{i}">
            """
            for j, value in enumerate(option["values"]):
                options_segment += f"""<option class="select-crowdsourcing-{i}-value" value="{j}">{value}</option>
                """
            options_segment += """
                    </select>
                </div>
            """
        elif option["type"] == "slider":
            # option["values"] are textual values to be displayed below the slider
            options_segment += f"""
                <div class="form-group crowdsourcing-option option-slider mb-4">
                    <div><label for="slider-{i}">{option["label"]}</label></div>
                    <div class="slider-container">
                    <input type="range" class="form-range slider-crowdsourcing" id="slider-crowdsourcing-{i}" min="0" max="{len(option["values"])-1}" list="slider-crowdsourcing-{i}-values">
                    </div>
                    <datalist id="slider-crowdsourcing-{i}-values" class="datalist-crowdsourcing">
            """
            for value in option["values"]:
                options_segment += f"""<option class="slider-crowdsourcing-{i}-value" value="{value}" label="{value}"></option>
                """
            options_segment += """
                    </datalist>
                </div>
            """
        else:
            raise ValueError(f"Unknown option type {option['type']}")

    options_segment += """<script src="{{ host_prefix }}/static/js/render-sliders.js"></script>"""
    return options_segment


def create_crowdsourcing_page(campaign_id, config):
    html_path = os.path.join(TEMPLATES_DIR, "campaigns", campaign_id, "annotate.html")

    os.makedirs(os.path.join(TEMPLATES_DIR, "campaigns", campaign_id), exist_ok=True)

    parts = []
    for part in ["header", "body", "footer"]:
        part_path = os.path.join(TEMPLATES_DIR, "campaigns", "annotate_{}.html".format(part))

        with open(part_path, "r") as f:
            parts.append(f.read())

    instructions_html = markdown.markdown(config["annotator_instructions"])
    final_message_html = markdown.markdown(config["final_message"])
    has_display_overlay = config.get("has_display_overlay", True)

    # format only the body, keeping the unfilled templates in header and footer
    template = Template(parts[1])

    rendered_content = template.render(
        instructions=instructions_html,
        final_message=final_message_html,
        annotation_span_categories=config.get("annotation_span_categories", []),
        has_display_overlay='style="display: none"' if not has_display_overlay else "",
        flags=generate_checkboxes(config.get("flags", [])),
        options=generate_options(config.get("options", [])),
    )

    # concatenate with header and footer
    content = parts[0] + rendered_content + parts[2]

    with open(html_path, "w") as f:
        f.write(content)


def run_llm_campaign(mode, campaign_id, announcer, campaign, datasets, model, threads):
    start_time = int(time.time())

    # set metadata status
    campaign.metadata["status"] = CampaignStatus.RUNNING
    campaign.update_metadata()
    db = campaign.db

    logger.info(f"Starting LLM campaign {campaign_id}")

    for i, row in db.iterrows():
        if threads[campaign_id]["running"] == False:
            break

        if row["status"] == ExampleStatus.FINISHED:
            continue

        dataset_id = row["dataset"]
        split = row["split"]
        setup_id = row.get("setup_id")
        example_idx = row["example_idx"]

        dataset = datasets[dataset_id]
        example = dataset.get_example(split, example_idx)

        if mode == "llm_eval":
            generated_output = dataset.get_output_for_idx_by_setup(
                split=split, output_idx=example_idx, setup_id=setup_id
            )
            output = model.annotate_example(example, generated_output)

        elif mode == "llm_gen":
            output = model.generate_output(example)

        if "error" in output:
            # remove the `running` flag
            threads[campaign_id]["running"] = False
            campaign.metadata["status"] = CampaignStatus.IDLE
            campaign.update_metadata()

            return error(output["error"])

        if mode == "llm_eval":
            annotator_id = model.get_annotator_id()

            record = save_annotation(
                annotator_id, campaign_id, dataset_id, split, setup_id, example_idx, output, start_time
            )
            record["output"] = record.pop("annotations")
        elif mode == "llm_gen":
            record = save_output(campaign_id, dataset_id, split, example_idx, output, start_time)

            # solely for the frontend
            record["setup_id"] = setup_id
            record["output"] = record.pop("out")

        db.loc[i, "status"] = ExampleStatus.FINISHED
        db.loc[i, "end"] = int(time.time())
        campaign.update_db(db)

        finished_examples_cnt = len(campaign.get_finished_examples())
        payload = {"finished_examples_cnt": finished_examples_cnt, "annotation": record}

        msg = format_sse(data=json.dumps(payload))
        if announcer is not None:
            announcer.announce(msg=msg)
        logger.info(f"{campaign_id}: {finished_examples_cnt}/{len(db)} examples")

    # if all fields are finished, set the metadata to finished
    if len(db.status.unique()) == 1 and db.status.unique()[0] == ExampleStatus.FINISHED:
        campaign.metadata["status"] = CampaignStatus.FINISHED
        campaign.update_metadata()

    if mode == "llm_eval":
        final_message = (
            f"All examples have been annotated. You can find the annotations in {ANNOTATIONS_DIR}/{campaign_id}/files."
        )
    elif mode == "llm_gen":
        final_message = (
            f"All examples have been generated. You can find the outputs in {GENERATIONS_DIR}/{campaign_id}/files."
        )

    return jsonify(success=True, status=campaign.metadata["status"], final_message=final_message)


def save_generation_outputs(app, campaign_id, setup_id):
    """
    Load the files from the `GENERATIONS_DIR` and save them in the appropriate subdirectory in `OUTPUT_DIR`.
    """

    campaign = load_campaign(app, campaign_id, mode="llm_gen")
    metadata = campaign.metadata

    # load the metadata
    with open(GENERATIONS_DIR / campaign_id / "metadata.json") as f:
        metadata = json.load(f)

    # load the outputs
    outputs = defaultdict(list)

    for file in os.listdir(GENERATIONS_DIR / campaign_id / "files"):
        if file.endswith(".jsonl"):
            with open(GENERATIONS_DIR / campaign_id / "files" / file) as f:
                for line in f:
                    record = json.loads(line)
                    # replace the campaign_id with the desired setup_id
                    record["setup_id"] = setup_id
                    key = (record["dataset"], record["split"])
                    outputs[key].append(record)

    # save the outputs
    for (dataset_id, split), examples in outputs.items():
        path = OUTPUT_DIR / dataset_id / split / setup_id
        os.makedirs(path / "files", exist_ok=True)

        # save the metadata
        with open(path / f"metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        with open(path / "files" / f"{setup_id}.jsonl", "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")

    return success()


# source: https://github.com/lhotse-speech/lhotse/blob/bc2c0a294b1437b90d1581d4f214348d2f8bfc12/lhotse/utils.py#L465
def resumable_download(
    url,
    filename,
    force_download,
    completed_file_size=None,
    missing_ok=True,
):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Check if the file exists and get its size
    file_exists = os.path.exists(filename)
    if file_exists:
        if force_download:
            logging.info(f"Removing existing file and downloading from scratch because force_download=True: {filename}")
            os.unlink(filename)
            file_size = 0
        else:
            file_size = os.path.getsize(filename)

        if completed_file_size and file_size == completed_file_size:
            return
    else:
        file_size = 0

    # Set the request headers to resume downloading
    # Also set user-agent header to stop picky servers from complaining with 403
    ua_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/603.1.30 (KHTML, like Gecko) Version/10.1 Safari/603.1.30",
    }

    headers = {
        "Range": "bytes={}-".format(file_size),
        **ua_headers,
    }

    # Create a request object with the URL and headers
    req = urllib.request.Request(url, headers=headers)

    # Open the file for writing in binary mode and seek to the end
    # r+b is needed in order to allow seeking at the beginning of a file
    # when downloading from scratch
    mode = "r+b" if file_exists else "wb"
    with open(filename, mode) as f:

        def _download(rq, size):
            f.seek(size, 0)
            # just in case some garbage was written to the file, truncate it
            f.truncate()

            # Open the URL and read the contents in chunks
            with urllib.request.urlopen(rq) as response:
                chunk_size = 1024
                total_size = int(response.headers.get("content-length", 0)) + size
                with tqdm(
                    total=total_size,
                    initial=size,
                    unit="B",
                    unit_scale=True,
                    desc=str(filename),
                ) as pbar:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))

        try:
            _download(req, file_size)
        except urllib.error.HTTPError as e:
            # "Request Range Not Satisfiable" means the requested range
            # starts after the file ends OR that the server does not support range requests.
            if e.code == 404 and missing_ok:
                logging.warning(f"{url} does not exist (error 404). Skipping this file.")
                if Path(filename).is_file():
                    os.remove(filename)
            elif e.code == 416:
                content_range = e.headers.get("Content-Range", None)
                if content_range is None:
                    # sometimes, the server actually supports range requests
                    # but does not return the Content-Range header with 416 code
                    # This is out of spec, but let us check twice for pragmatic reasons.
                    head_req = urllib.request.Request(url, method="HEAD")
                    head_res = urllib.request.urlopen(head_req)
                    if head_res.headers.get("Accept-Ranges", "none") != "none":
                        content_length = head_res.headers.get("Content-Length")
                        content_range = f"bytes */{content_length}"

                if content_range == f"bytes */{file_size}":
                    # If the content-range returned by server also matches the file size,
                    # then the file is already downloaded
                    logging.info(f"File already downloaded: {filename}")
                else:
                    logging.info("Server does not support range requests - attempting downloading from scratch")
                    _download(urllib.request.Request(url, headers=ua_headers), 0)
            else:
                raise e


def migrate():
    """
    Ensure backwards compatibility after changes in the app
    """
    from factgenie import OLD_DATASET_CONFIG_PATH, OLD_MAIN_CONFIG_PATH, MAIN_CONFIG_PATH
    import shutil

    # if an old config file exist, we perform the migration
    if not OLD_MAIN_CONFIG_PATH.exists():
        return

    logger.warning("Factgenie updated, performing migration tasks...")

    # ------------------
    # update old config files
    # ------------------

    logger.warning("Moving config.yml to config/config.yml")
    shutil.move(OLD_MAIN_CONFIG_PATH, MAIN_CONFIG_PATH)

    logger.warning("Moving loaders/datasets.yml to config/datasets.yml")
    shutil.move(OLD_DATASET_CONFIG_PATH, DATASET_CONFIG_PATH)

    # load the dataset config, keep only the datasets that are not available for download (i.e. not in the resources.yml)
    with open(DATASET_CONFIG_PATH) as f:
        dataset_config = yaml.safe_load(f)
        # use ids as top level keys
        dataset_config = dataset_config["datasets"]

    with open(RESOURCES_CONFIG_PATH) as f:
        resources_config = yaml.safe_load(f)

    standard_datasets = set(resources_config.keys()).union({"logicnlg", "xsum"})
    dataset_config = {k: v for k, v in dataset_config.items() if k not in standard_datasets}

    if dataset_config:
        logger.warning(f"Keeping the following local datasets: {list(dataset_config.keys())}")

        logger.warning(
            f"You may need to manually update loaders of the datasets you added locally in the previous version of factgenie. You can find their configurations in {DATASET_CONFIG_PATH}."
        )

    # if any dataset has `class: base.X`, rename it to `basic.X`
    for dataset_id, dataset in dataset_config.items():
        if dataset["class"].startswith("base."):
            dataset_config[dataset_id]["class"] = dataset["class"].replace("base.", "basic.")

    with open(DATASET_CONFIG_PATH, "w") as f:
        yaml.dump(dataset_config, f, indent=2, allow_unicode=True)

    # ------------------
    # convert old model outputs to the new JSONL format
    # ------------------
    logger.warning("Converting old model outputs to a new JSONL format...")
    outs = list(Path(OUTPUT_DIR).glob("**/*.json"))

    # ignore metadata.json
    outs = [out for out in outs if not out.name == "metadata.json"]

    for out in outs:
        with open(out) as f:
            j = json.load(f)

        logger.warning(f"Converting old model outputs to a new JSONL format for {j['dataset']}...")

        new_out = []
        for i, gen in enumerate(j["generated"]):
            record = {
                "dataset": j["dataset"],
                "split": j.get("split", out.parent.name),
                "setup_id": j["setup"]["id"],
                "example_idx": i,
                "out": gen["out"],
            }
            if gen.get("in"):
                record["in"] = gen["in"]
            new_out.append(record)

        output_path = (
            OUTPUT_DIR / j["dataset"] / j.get("split", out.parent.name) / "files" / out.with_suffix(".jsonl").name
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for gen in new_out:
                f.write(json.dumps(gen) + "\n")

        metadata = j["setup"]

        if "params" in metadata:
            metadata["model_args"] = metadata.pop("params")

        if "prompt" in metadata:
            metadata["prompt_template"] = metadata.pop("prompt")

        with open(output_path.parent.parent / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        # remove the old JSON file
        out.unlink()

    ann_outs = list(Path(ANNOTATIONS_DIR).glob("**/*.jsonl"))

    logger.warning("Updating annotation outputs...")
    for file_path in ann_outs:
        with open(file_path, "r") as infile, open(file_path + ".tmp", "w") as outfile:
            for line in infile:
                obj = json.loads(line)
                if "setup" in obj and "id" in obj["setup"]:
                    obj["setup_id"] = obj["setup"]["id"]
                    del obj["setup"]
                outfile.write(json.dumps(obj) + "\n")
        os.replace(file_path + ".tmp", file_path)

    logger.warning("Migration complete")
