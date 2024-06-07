#!/usr/bin/env python3

import os
import json
import glob
import time
import logging
import pandas as pd
import random
import time
import coloredlogs
import traceback
import yaml
from flask import jsonify
from collections import defaultdict
from pathlib import Path
from factgenie.campaigns import Campaign, HumanCampaign, ModelCampaign
from factgenie.evaluate import LLMMetric, LLMMetricFactory

DIR_PATH = os.path.dirname(__file__)
TEMPLATES_DIR = os.path.join(DIR_PATH, "templates")
STATIC_DIR = os.path.join(DIR_PATH, "static")
ANNOTATIONS_DIR = os.path.join(DIR_PATH, "annotations")
LLM_CONFIG_DIR = os.path.join(DIR_PATH, "llm-eval")


file_handler = logging.FileHandler("error.log")
file_handler.setLevel(logging.ERROR)


logging.basicConfig(
    format="%(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[file_handler, logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO", logger=logger, fmt="%(asctime)s %(levelname)s %(message)s")


def success():
    resp = jsonify(success=True)
    return resp


def get_dataset(app, dataset_name):
    return app.db["datasets_obj"].get(dataset_name)


def generate_metric_index(app):
    # go through all the files in the LLM_CONFIG_DIR
    # instantiate the class and return the object
    metrics = {}

    for file in os.listdir(LLM_CONFIG_DIR):
        if file.endswith(".yaml"):
            with open(os.path.join(LLM_CONFIG_DIR, file)) as f:
                config = yaml.safe_load(f)
            try:
                metric = LLMMetricFactory.get_metric(config)
                metrics[metric.metric_name] = metric
            except Exception as e:
                logger.error(f"Error while loading metric {file}")
                traceback.print_exc()
                continue

    app.db["metric_index"] = metrics


def generate_campaign_index(app):
    campaigns = defaultdict(dict)

    # find all subdirs in CROWDSOURCING_DIR
    for campaign_dir in Path(ANNOTATIONS_DIR).iterdir():
        if not campaign_dir.is_dir():
            continue

        metadata = json.load(open(os.path.join(campaign_dir, "metadata.json")))
        campaign_source = metadata.get("source")
        campaign_id = metadata["id"]

        if campaign_source == "human":
            campaign = HumanCampaign(campaign_id=campaign_id)
        elif campaign_source == "model":
            campaign = ModelCampaign(campaign_id=campaign_id)

        campaigns[campaign_source][campaign_id] = campaign

    app.db["campaign_index"] = campaigns


def generate_annotation_index(app):
    # contains annotations for each generated output
    annotations = defaultdict(list)

    # for all subdirectories in ANNOTATIONS_DIR, load content of all the jsonl files
    for subdir in os.listdir(ANNOTATIONS_DIR):

        try:
            # find metadata for the campaign
            metadata_path = os.path.join(ANNOTATIONS_DIR, subdir, "metadata.json")
            if not os.path.exists(metadata_path):
                continue

            with open(metadata_path) as f:
                metadata = json.load(f)

            jsonl_files = glob.glob(os.path.join(ANNOTATIONS_DIR, subdir, "files/*.jsonl"))

            for jsonl_file in jsonl_files:
                with open(jsonl_file) as f:
                    for line in f:
                        annotation = json.loads(line)
                        annotation["metadata"] = metadata

                        key = (
                            annotation["dataset"],
                            annotation["split"],
                            annotation["example_idx"],
                            annotation["setup"]["id"],
                        )
                        annotations[key].append(annotation)
        except:
            if app.config["debug"]:
                traceback.print_exc()
                logger.error(f"Error while loading annotations for {subdir}")
                raise

    app.db["annotation_index"] = annotations

    return annotations


def get_annotations(app, dataset_name, split, example_idx, setup_id):
    annotation_index = app.db["annotation_index"]
    key = (dataset_name, split, example_idx, setup_id)

    return annotation_index.get(key, [])


def get_example_data(app, dataset_name, split, example_idx):
    dataset = get_dataset(app=app, dataset_name=dataset_name)

    example = dataset.get_example(split=split, example_idx=example_idx)
    html = dataset.render(example=example)
    generated_outputs = dataset.get_generated_outputs(split=split, output_idx=example_idx)

    for output in generated_outputs:
        setup_id = output["setup"]["id"]
        annotations = get_annotations(app, dataset_name, split, example_idx, setup_id)

        output["annotations"] = annotations

    dataset_info = dataset.get_info()

    return {
        "html": html,
        "raw_data": example,
        "total_examples": dataset.get_example_count(split),
        "dataset_info": dataset_info,
        "generated_outputs": generated_outputs,
    }


def free_idle_examples(db):
    start = int(time.time())

    # check if there are annotations which are idle for more than 2 hours: set them to free
    idle_examples = db[(db["status"] == "assigned") & (db["start"] < start - 2 * 60 * 60)]
    for i in idle_examples.index:
        db.loc[i, "status"] = "free"
        db.loc[i, "start"] = ""
        db.loc[i, "annotator_id"] = ""

    return db


def select_batch_idx(db, seed):
    free_examples = db[db["status"] == "free"]

    # if no free examples, take the oldest assigned example
    if len(free_examples) == 0:
        free_examples = db[db["status"] == "assigned"]
        free_examples = free_examples.sort_values(by=["start"])
        free_examples = free_examples.head(1)

        logger.info(f"Annotating extra example {free_examples.index[0]}")

    example = free_examples.sample(random_state=seed)
    batch_idx = int(example.batch_idx.values[0])
    logger.info(f"Selecting batch {batch_idx}")

    return batch_idx


def get_annotator_batch(app, campaign, db, prolific_pid, session_id, study_id):
    # simple locking over the CSV file to prevent double writes
    with app.db["lock"]:
        logging.info(f"Acquiring lock for {prolific_pid}")
        start = int(time.time())

        seed = random.seed(str(start) + prolific_pid + session_id + study_id)

        batch_idx = select_batch_idx(db, seed)
        if prolific_pid != "test":
            db = free_idle_examples(db)

            # update the CSV
            db.loc[batch_idx, "status"] = "assigned"
            db.loc[batch_idx, "start"] = start
            db.loc[batch_idx, "annotator_id"] = prolific_pid

            campaign.update_db(db)

        annotator_batch = campaign.get_examples_for_batch(batch_idx)

        for example in annotator_batch:
            example.update(
                {
                    "campaign_id": campaign.campaign_id,
                    "batch_idx": batch_idx,
                    "annotator_id": prolific_pid,
                    "session_id": session_id,
                    "study_id": study_id,
                    "start_timestamp": start,
                }
            )

        logging.info(f"Releasing lock for {prolific_pid}")

    return annotator_batch


def generate_llm_eval_db(app, campaign_data):
    # load all outputs
    all_examples = []

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

    df = pd.DataFrame.from_records(all_examples)

    # create a column for batch index and assign each example to a batch
    df["annotator_id"] = ""
    df["status"] = "free"
    df["start"] = ""

    return df


def generate_campaign_db(app, campaign_data, examples_per_batch, sort_order):
    # load all outputs
    all_examples = []

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

    if sort_order == "example":
        random.seed(42)
        random.shuffle(all_examples)
        # group outputs by example ids, dataset and split
        all_examples = sorted(all_examples, key=lambda x: (x["example_idx"], x["dataset"], x["split"]))
    elif sort_order == "random":
        random.seed(42)
        random.shuffle(all_examples)
    elif sort_order == "default":
        pass
    else:
        raise ValueError(f"Unknown sort order {sort_order}")
    df = pd.DataFrame.from_records(all_examples)

    # create a column for batch index and assign each example to a batch
    df["batch_idx"] = df.index // examples_per_batch
    df["annotator_id"] = ""
    df["status"] = "free"
    df["start"] = ""

    return df


def get_dataset_overview(app):
    overview = {}
    for name, dataset in app.db["datasets_obj"].items():
        overview[name] = {
            "splits": dataset.get_splits(),
            "description": dataset.get_info(),
            "example_count": dataset.get_example_count(),
            "type": dataset.type,
        }

    return overview
