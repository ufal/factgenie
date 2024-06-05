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
import threading
import traceback
import shutil
import datetime
import urllib.parse
from flask import Flask, render_template, jsonify, request, session
from collections import defaultdict
from pathlib import Path

from factgenie.campaigns import get_campaigns, Campaign
from factgenie.loaders import DATASET_CLASSES
from factgenie.evaluate import LLMMetric, Llama3Metric

DIR_PATH = os.path.dirname(__file__)
TEMPLATES_DIR = os.path.join(DIR_PATH, "templates")
STATIC_DIR = os.path.join(DIR_PATH, "static")
ANNOTATIONS_DIR = os.path.join(DIR_PATH, "annotations")


app = Flask("factgenie", template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
app.config.update(SECRET_KEY=os.urandom(24))
app.db = {}
app.db["annotation_index"] = {}
app.db["lock"] = threading.Lock()


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


@app.template_filter("ctime")
def timectime(timestamp):
    try:
        s = datetime.datetime.fromtimestamp(timestamp)
        return s.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp


@app.template_filter("elapsed")
def time_elapsed(timestamp):
    try:
        s = datetime.datetime.fromtimestamp(timestamp)
        diff = str(datetime.datetime.now() - s)
        return diff.split(".")[0]
    except:
        return timestamp


@app.template_filter("annotate_url")
def annotate_url(current_url):
    # get the base url (without any "browse", "crowdsourcing" or "crowdsourcing/campaign" in it)
    parsed = urllib.parse.urlparse(current_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}/"
    return f"{base_url}annotate"


def get_dataset(dataset_name):
    return app.db["datasets_obj"].get(dataset_name)


def generate_campaign_index(source):
    app.db["campaign_index"] = get_campaigns(source=source)


def generate_annotation_index():
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


def get_annotations(dataset_name, split, example_idx, setup_id):
    annotation_index = app.db["annotation_index"]
    key = (dataset_name, split, example_idx, setup_id)

    return annotation_index.get(key, [])


def get_example_data(dataset_name, split, example_idx):
    dataset = get_dataset(dataset_name=dataset_name)

    example = dataset.get_example(split=split, example_idx=example_idx)
    html = dataset.render(example=example)
    generated_outputs = dataset.get_generated_outputs(split=split, output_idx=example_idx)

    for output in generated_outputs:
        setup_id = output["setup"]["id"]
        annotations = get_annotations(dataset_name, split, example_idx, setup_id)

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


def get_annotator_batch(campaign, db, prolific_pid, session_id, study_id):
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


def generate_campaign_db(campaign_data, examples_per_batch, sort_order):
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

    random.seed(42)
    random.shuffle(all_examples)

    if sort_order == "example":
        # group outputs by example ids, dataset and split
        all_examples = sorted(all_examples, key=lambda x: (x["example_idx"], x["dataset"], x["split"]))
    elif sort_order == "random":
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


def get_dataset_overview():
    overview = {}
    for name, dataset in app.db["datasets_obj"].items():
        overview[name] = {
            "splits": dataset.get_splits(),
            "description": dataset.get_info(),
            "example_count": dataset.get_example_count(),
            "type": dataset.type,
        }

    return overview


# -----------------
# Flask endpoints
# -----------------


@app.route("/", methods=["GET", "POST"])
def index():
    logger.info(f"Main page loaded")

    return render_template(
        "index.html",
        allow_browse=app.config["allow_browse"],
        allow_annotate=app.config["allow_annotate"],
        host_prefix=app.config["host_prefix"],
    )


@app.route("/about", methods=["GET", "POST"])
def about():
    logger.info(f"About page loaded")

    return render_template(
        "about.html",
        host_prefix=app.config["host_prefix"],
    )


@app.route("/annotate", methods=["GET", "POST"])
def annotate():
    if app.config["allow_annotate"] is False:
        return render_template("disabled.html")

    logger.info(f"Annotate page loaded")

    generate_campaign_index(source="human")
    campaign_id = request.args.get("campaign")
    campaign = app.db["campaign_index"][campaign_id]
    compl_code = campaign.metadata["prolific_code"]
    prolific_pid = request.args.get("PROLIFIC_PID", "test")
    session_id = request.args.get("SESSION_ID", "test")
    study_id = request.args.get("STUDY_ID", "test")

    db = campaign.db
    metadata = campaign.metadata
    annotation_set = get_annotator_batch(campaign, db, prolific_pid, session_id, study_id)
    return render_template(
        f"campaigns/{campaign.campaign_id}/annotate.html",
        host_prefix=app.config["host_prefix"],
        annotation_set=annotation_set,
        annotator_id=prolific_pid,
        compl_code=compl_code,
        metadata=metadata,
    )


@app.route("/browse", methods=["GET", "POST"])
def browse():
    if app.config["allow_browse"] is False:
        return render_template("disabled.html")

    logger.info(f"Browse page loaded")

    generate_annotation_index()

    dataset_name = request.args.get("dataset")
    split = request.args.get("split")
    example_idx = request.args.get("example_idx")

    if dataset_name and split and example_idx:
        display_example = {"dataset": dataset_name, "split": split, "example_idx": int(example_idx)}
        logger.info(f"Serving permalink for {display_example}")
    else:
        display_example = None

    datasets = get_dataset_overview()

    return render_template(
        "browse.html",
        display_example=display_example,
        datasets=datasets,
        host_prefix=app.config["host_prefix"],
        annotations=app.db["annotation_index"],
    )


@app.route("/crowdsourcing", methods=["GET", "POST"])
def crowdsourcing():
    logger.info(f"Crowdsourcing page loaded")

    datasets = app.db["datasets_obj"]
    model_outs = {x: [] for x in ["datasets", "splits", "setup_ids", "valid_triplets"]}

    for dataset_name, dataset in datasets.items():
        splits = dataset.get_splits()
        model_outs["datasets"].append(dataset_name)

        for split in splits:
            output_setups = dataset.outputs[split].keys()
            model_outs["splits"].append(split)

            for setup_id in output_setups:
                model_outs["setup_ids"].append(setup_id)
                model_outs["valid_triplets"].append(
                    {
                        "dataset": dataset_name,
                        "split": split,
                        "setup_id": setup_id,
                        "example_count": dataset.get_example_count(split),
                    }
                )

    for key in ["datasets", "splits", "setup_ids"]:
        model_outs[key] = sorted(list(set(model_outs[key])))

    generate_campaign_index(source="human")

    campaign_index = app.db["campaign_index"]
    campaigns = defaultdict(dict)

    for campaign_id, campaign in campaign_index.items():
        campaigns[campaign_id]["metadata"] = campaign.metadata
        campaigns[campaign_id]["stats"] = campaign.get_stats()

    return render_template(
        "crowdsourcing.html",
        model_outs=model_outs,
        campaigns=campaigns,
        default_error_categories=app.config["default_error_categories"],
        host_prefix=app.config["host_prefix"],
    )


@app.route("/crowdsourcing/campaign", methods=["GET", "POST"])
def campaign():
    generate_campaign_index(source="human")

    campaign_id = request.args.get("campaign")
    db = app.db["campaign_index"][campaign_id].db
    # replace NaN with empty string
    db = db.where(pd.notnull(db), "")

    # group by batch idx
    # add a column with the number of examples for each batch
    # for other columns keep first item
    db = db.groupby("batch_idx").agg(
        {
            "dataset": "first",
            "split": "first",
            "example_idx": "count",
            "setup_id": "first",
            "status": "first",
            "start": "first",
            "annotator_id": "first",
        }
    )
    db = db.rename(columns={"example_idx": "example_cnt"}).reset_index()
    db = db.to_dict(orient="records")

    return render_template(
        "campaign.html",
        campaign_id=campaign_id,
        db=db,
        host_prefix=app.config["host_prefix"],
    )


@app.route("/delete_campaign", methods=["POST"])
def delete_campaign():
    data = request.get_json()
    campaign_name = data.get("campaign_id")

    shutil.rmtree(os.path.join(ANNOTATIONS_DIR, campaign_name))
    shutil.rmtree(os.path.join(TEMPLATES_DIR, "campaigns", campaign_name))
    del app.db["campaign_index"][campaign_name]

    return success()


@app.route("/start_campaign", methods=["POST"])
def start_campaign():
    data = request.get_json()

    campaign_id = data.get("campaignId")
    examples_per_batch = int(data.get("examplesPerBatch"))
    idle_time = int(data.get("idleTime"))
    prolific_code = data.get("prolificCode")
    campaign_data = data.get("campaignData")
    error_categories = data.get("errorCategories")
    sort_order = data.get("sortOrder")

    # create a new directory
    if os.path.exists(os.path.join(ANNOTATIONS_DIR, campaign_id)):
        return jsonify({"error": "Campaign already exists"})

    os.makedirs(os.path.join(ANNOTATIONS_DIR, campaign_id, "files"), exist_ok=True)

    # create the annotation CSV
    db = generate_campaign_db(campaign_data, examples_per_batch, sort_order)
    db.to_csv(os.path.join(ANNOTATIONS_DIR, campaign_id, "db.csv"), index=False)

    # save metadata
    with open(os.path.join(ANNOTATIONS_DIR, campaign_id, "metadata.json"), "w") as f:
        json.dump(
            {
                "id": campaign_id,
                "idle_time": idle_time,
                "prolific_code": prolific_code,
                "data": campaign_data,
                "sort_order": sort_order,
                "source": "human",
                "error_categories": error_categories,
                "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=4,
        )

    # copy templates/campaigns/annotate_default.html into templates/campaigns/{campaign_id} as "annotate.html"
    os.makedirs(os.path.join(TEMPLATES_DIR, "campaigns", campaign_id), exist_ok=True)

    shutil.copy(
        os.path.join(TEMPLATES_DIR, "campaigns", "annotate_default.html"),
        os.path.join(TEMPLATES_DIR, "campaigns", campaign_id, "annotate.html"),
    )

    # create the campaign object
    campaign = Campaign(campaign_id=campaign_id)

    app.db["campaign_index"][campaign_id] = campaign

    return success()


@app.route("/datasets", methods=["GET", "POST"])
def datasets():
    logger.info(f"Datasets loaded")
    datasets = get_dataset_overview()

    return render_template("datasets.html", datasets=datasets, host_prefix=app.config["host_prefix"])


@app.route("/run_llm_eval", methods=["GET", "POST"])
def run_llm_eval():
    data = request.get_json()

    metric = Llama3Metric()
    campaign_id = "llama3"
    batch_idx = 0
    annotator_id = "llama3"
    split = "test"
    dataset_name = "gsmarena"
    setup_id = "mistral"
    now = int(time.time())
    examples_per_batch = 1

    campaign_id = data.get("campaignId")
    campaign_data = data.get("campaignData")
    sort_order = data.get("sortOrder")

    # create a new directory
    if os.path.exists(os.path.join(ANNOTATIONS_DIR, campaign_id)):
        return jsonify({"error": "Campaign already exists"})

    os.makedirs(os.path.join(ANNOTATIONS_DIR, campaign_id, "files"), exist_ok=True)

    # create the annotation CSV
    db = generate_campaign_db(campaign_data, examples_per_batch, sort_order)
    db.to_csv(os.path.join(ANNOTATIONS_DIR, campaign_id, "db.csv"), index=False)

    # save metadata
    with open(os.path.join(ANNOTATIONS_DIR, campaign_id, "metadata.json"), "w") as f:
        json.dump(
            {
                "id": campaign_id,
                "data": campaign_data,
                "sort_order": sort_order,
                "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=4,
        )

    # create the campaign object
    campaign = Campaign(campaign_id=campaign_id)

    app.db["campaign_index"][campaign_id] = campaign

    # debug: get input - output examples from the ice_hockey dataset
    dataset = app.db["datasets_obj"][dataset_name]
    annotations = []

    for example_idx in range(5):
        data_input = str(dataset.get_example(split=split, example_idx=example_idx))
        model_output = dataset.get_generated_outputs(split=split, output_idx=example_idx)

        annotation_set = metric.annotate_example(data_input, model_output)
        annotation = {
            "annotations": annotation_set,
            "annotator_id": annotator_id,
            "batch_idx": batch_idx,
            "campaign_id": campaign_id,
            "dataset": dataset_name,
            "example_idx": example_idx,
            "setup": {"id": setup_id},
            "split": split,
            "start_timestamp": now,
        }
        annotations.append(annotation)

    save_dir = os.path.join(ANNOTATIONS_DIR, campaign_id, "files")
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, f"{batch_idx}-{annotator_id}-{now}.jsonl"), "w") as f:
        for row in annotations:
            f.write(json.dumps(row) + "\n")

    return success()


@app.route("/example", methods=["GET", "POST"])
def render_example():
    dataset_name = request.args.get("dataset")
    split = request.args.get("split")
    example_idx = int(request.args.get("example_idx"))

    try:
        example_data = get_example_data(dataset_name, split, example_idx)
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error while getting example data: {e}")
        example_data = {}

    return jsonify(example_data)


@app.route("/llm_eval", methods=["GET", "POST"])
def llm_eval():
    logger.info(f"LLM eval page loaded")

    datasets = app.db["datasets_obj"]
    model_outs = {x: [] for x in ["datasets", "splits", "setup_ids", "valid_triplets"]}

    for dataset_name, dataset in datasets.items():
        splits = dataset.get_splits()
        model_outs["datasets"].append(dataset_name)

        for split in splits:
            output_setups = dataset.outputs[split].keys()
            model_outs["splits"].append(split)

            for setup_id in output_setups:
                model_outs["setup_ids"].append(setup_id)
                model_outs["valid_triplets"].append(
                    {
                        "dataset": dataset_name,
                        "split": split,
                        "setup_id": setup_id,
                        "example_count": dataset.get_example_count(split),
                    }
                )

    for key in ["datasets", "splits", "setup_ids"]:
        model_outs[key] = sorted(list(set(model_outs[key])))

    generate_campaign_index(source="model")

    campaign_index = app.db["campaign_index"]
    campaigns = defaultdict(dict)

    for campaign_id, campaign in campaign_index.items():
        campaigns[campaign_id]["metadata"] = campaign.metadata
        campaigns[campaign_id]["stats"] = campaign.get_stats()

    return render_template(
        "llm_eval.html",
        model_outs=model_outs,
        campaigns=campaigns,
        default_error_categories=app.config["default_error_categories"],
        host_prefix=app.config["host_prefix"],
    )


@app.route("/submit_annotations", methods=["POST"])
def submit_annotations():
    logger.info(f"Received annotations")
    data = request.get_json()
    campaign_id = data["campaign_id"]
    annotation_set = data["annotation_set"]
    annotator_id = data["annotator_id"]
    now = int(time.time())

    save_dir = os.path.join(ANNOTATIONS_DIR, campaign_id, "files")
    os.makedirs(save_dir, exist_ok=True)
    campaign = app.db["campaign_index"][campaign_id]

    with app.db["lock"]:
        db = campaign.db
        batch_idx = annotation_set[0]["batch_idx"]

        with open(os.path.join(save_dir, f"{batch_idx}-{annotator_id}-{now}.jsonl"), "w") as f:
            for row in annotation_set:
                f.write(json.dumps(row) + "\n")

        db.loc[batch_idx, "status"] = "finished"
        db.to_csv(os.path.join(ANNOTATIONS_DIR, campaign_id, "db.csv"), index=False)
        logger.info(f"Annotations for {campaign_id} (batch {batch_idx}) saved")

    return jsonify({"status": "success"})
