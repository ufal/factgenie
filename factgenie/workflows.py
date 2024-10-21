#!/usr/bin/env python3
import os
import datetime
import json
import time
import logging
import pandas as pd
import random
import time
import traceback
import yaml
import shutil
import importlib
import zipfile
import markdown
import traceback
import ast
import tempfile

import factgenie.utils as utils

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
    CampaignMode,
    CampaignStatus,
    ExampleStatus,
)
from jinja2 import Template

from factgenie import (
    CAMPAIGN_DIR,
    OUTPUT_DIR,
    INPUT_DIR,
    TEMPLATES_DIR,
    LLM_EVAL_CONFIG_DIR,
    LLM_GEN_CONFIG_DIR,
    CROWDSOURCING_CONFIG_DIR,
    PREVIEW_STUDY_ID,
)

file_handler = logging.FileHandler("error.log")
file_handler.setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


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

    if mode == CampaignMode.LLM_EVAL:
        config_dir = LLM_EVAL_CONFIG_DIR
    elif mode == CampaignMode.LLM_GEN:
        config_dir = LLM_GEN_CONFIG_DIR
    elif mode == CampaignMode.CROWDSOURCING:
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


def instantiate_campaign(app, campaign_id, mode):
    campaign = None

    if mode == CampaignMode.CROWDSOURCING:
        scheduler = app.db["scheduler"]
        campaign = HumanCampaign(campaign_id=campaign_id, scheduler=scheduler)
    elif mode == CampaignMode.LLM_EVAL:
        campaign = LLMCampaignEval(campaign_id=campaign_id)
    elif mode == CampaignMode.LLM_GEN:
        campaign = LLMCampaignGen(campaign_id=campaign_id)
    elif mode == CampaignMode.EXTERNAL:
        campaign = ExternalCampaign(campaign_id=campaign_id)
    elif mode == CampaignMode.HIDDEN:
        pass
    else:
        logger.warning(f"Unknown campaign mode: {mode}")

    return campaign


def load_campaign(app, campaign_id):
    campaign_index = generate_campaign_index(app, force_reload=False)

    if campaign_id not in campaign_index:
        logger.error(f"Unknown campaign {campaign_id}")
        return None

    campaign = campaign_index[campaign_id]
    return campaign


def generate_campaign_index(app, force_reload=True):
    if "campaign_index" in app.db:
        campaign_index = app.db["campaign_index"]
    else:
        campaign_index = defaultdict(dict)

    existing_campaign_ids = set()
    for campaign_dir in Path(CAMPAIGN_DIR).iterdir():
        try:
            metadata = json.load(open(campaign_dir / "metadata.json"))
            mode = metadata["mode"]
            campaign_id = metadata["id"]
            existing_campaign_ids.add(campaign_id)

            if campaign_id in campaign_index and not force_reload:
                continue

            campaign = instantiate_campaign(app=app, campaign_id=campaign_id, mode=mode)

            if (
                (mode == CampaignMode.LLM_EVAL or mode == CampaignMode.LLM_GEN)
                and campaign.metadata["status"] == CampaignStatus.RUNNING
                and campaign_id not in app.db["running_campaigns"]
            ):
                campaign.metadata["status"] = CampaignStatus.IDLE
                campaign.update_metadata()

            campaign_index[campaign_id] = campaign

        except:
            traceback.print_exc()
            logger.error(f"Error while loading campaign {campaign_dir}")

    # remove campaigns that are no longer in the directory
    campaign_index = {k: v for k, v in campaign_index.items() if k in existing_campaign_ids}

    app.db["campaign_index"] = campaign_index

    return app.db["campaign_index"]


def get_sorted_campaign_list(app, modes):
    campaign_index = generate_campaign_index(app, force_reload=True)

    campaigns = [c for c in campaign_index.values() if c.metadata["mode"] in modes]

    campaigns.sort(key=lambda x: x.metadata["created"], reverse=True)
    campaigns = {
        c.metadata["id"]: {"metadata": c.metadata, "stats": c.get_stats(), "data": c.db.to_dict(orient="records")}
        for c in campaigns
    }
    return campaigns


def load_annotations_for_campaign(subdir):
    annotations_campaign = defaultdict(list)

    # find metadata for the campaign
    metadata_path = CAMPAIGN_DIR / subdir / "metadata.json"
    if not metadata_path.exists():
        return None

    with open(metadata_path) as f:
        metadata = json.load(f)

    if metadata["mode"] == CampaignMode.HIDDEN:
        return None

    jsonl_files = (CAMPAIGN_DIR / subdir / "files").glob("*.jsonl")

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

    # for all subdirectories in CAMPAIGN_DIR, load content of all the jsonl files
    for subdir in os.listdir(CAMPAIGN_DIR):
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


def generate_output_index(app=None, force_reload=True):
    if app and app.db["output_index"] is not None and not force_reload:
        return app.db["output_index"]

    outputs = []

    # find recursively all JSONL files in the output directory
    outs = list(Path(OUTPUT_DIR).rglob("*.jsonl"))

    for out in outs:
        with open(out) as f:
            for line_num, line in enumerate(f):
                try:
                    j = json.loads(line)

                    for key in ["dataset", "split", "setup_id"]:
                        j[key] = slugify(j[key])

                    outputs.append(j)
                except Exception as e:
                    logger.error(
                        f"Error parsing output file {out} at line {line_num + 1}:\n\t{e.__class__.__name__}: {e}"
                    )

    # if no outputs, create an empty DataFrame with columns `dataset`, `split`, `setup_id`, `example_idx`, `in`, `out`, `metadata`
    if not outputs:
        output_index = pd.DataFrame(columns=["dataset", "split", "setup_id", "example_idx", "in", "out", "metadata"])
    else:
        output_index = pd.DataFrame.from_records(outputs)

    if app:
        app.db["output_index"] = output_index

    return output_index


def export_campaign_outputs(campaign_id):
    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for root, _dirs, files in os.walk(os.path.join(CAMPAIGN_DIR, campaign_id)):
            for file in files:
                zip_file.write(
                    os.path.join(root, file),
                    os.path.relpath(os.path.join(root, file), os.path.join(CAMPAIGN_DIR, campaign_id)),
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

    try:
        example = dataset.get_example(split=split, example_idx=example_idx)
    except:
        raise ValueError("Example cannot be retrieved from the dataset")

    try:
        html = dataset.render(example=example)
    except:
        raise ValueError("Example cannot be rendered")

    # temporary solution for external files
    # prefix all the "/files" calls with "app.config["host_prefix"]"
    html = html.replace('src="/files', f'src="{app.config["host_prefix"]}/files')

    generated_outputs = get_outputs(app, dataset_id, split, example_idx)

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
    outputs = generate_output_index(app).copy()

    # filter the df by datasets
    outputs = outputs[outputs["dataset"].isin(datasets)]

    # if non_empty, filter only the examples with outputs
    if non_empty:
        outputs = outputs[outputs["out"].notnull()]

    # aggregate `example_idx` to list, drop "in", "out", "metadata"
    outputs = (
        outputs.groupby(["dataset", "split", "setup_id"])
        .agg(example_idx=pd.NamedAgg(column="example_idx", aggfunc=list))
        .reset_index()
    )
    # rename "example_idx" to "output_ids"
    outputs = outputs.rename(columns={"example_idx": "output_ids"})
    outputs = outputs.to_dict(orient="records")

    return outputs


def get_output(dataset, split, setup_id, example_idx):
    output_index = generate_output_index()

    output = output_index[
        (output_index["dataset"] == dataset)
        & (output_index["split"] == split)
        & (output_index["setup_id"] == setup_id)
        & (output_index["example_idx"] == example_idx)
    ]

    if len(output) == 0:
        return None

    return str(output.iloc[0].out)


def get_outputs(app, dataset_id, split, example_idx):
    outputs = generate_output_index(app, force_reload=False)

    outputs = outputs[
        (outputs["dataset"] == dataset_id) & (outputs["split"] == split) & (outputs["example_idx"] == example_idx)
    ]

    outputs = outputs.to_dict(orient="records")

    return outputs


def get_output_ids(dataset, split, setup_id):
    output_index = generate_output_index()

    output_ids = output_index[
        (output_index["dataset"] == dataset) & (output_index["split"] == split) & (output_index["setup_id"] == setup_id)
    ]["example_idx"].tolist()

    return output_ids


def select_batch_idx(db, seed):
    free_examples = db[db["status"] == ExampleStatus.FREE]
    assigned_examples = db[db["status"] == ExampleStatus.ASSIGNED]

    if len(free_examples) == 0 and len(assigned_examples) == 0:
        raise ValueError("No examples available")

    # if no free examples but still assigned examples, take the oldest assigned example
    # if len(free_examples) == 0 and len(assigned_examples) > 0:
    #     free_examples = assigned_examples
    #     free_examples = free_examples.sort_values(by=["start"])
    #     free_examples = free_examples.head(1)

    #     logger.info(f"Annotating extra example {free_examples.index[0]}")

    example = free_examples.sample(random_state=seed)
    batch_idx = int(example.batch_idx.values[0])
    logger.info(f"Selecting batch {batch_idx}")

    return batch_idx


def get_annotator_batch(app, campaign, service_ids):
    db = campaign.db

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

        # for llm_gen, setup_id based on the generation model
        if mode == CampaignMode.LLM_EVAL:
            setup_id = c["setup_id"]
            ids = get_output_ids(dataset.id, c["split"], setup_id)
        elif mode == CampaignMode.LLM_GEN:
            setup_id = campaign_id
            ids = list(range(dataset.get_example_count(c["split"])))

        for i in ids:
            record = {
                "dataset": c["dataset"],
                "split": c["split"],
                "example_idx": i,
            }
            record["setup_id"] = setup_id

            all_examples.append(record)

    df = pd.DataFrame.from_records(all_examples)

    # create a column for batch index and assign each example to a batch
    df["annotator_group"] = 0
    df["annotator_id"] = ""
    df["status"] = ExampleStatus.FREE
    df["start"] = None
    df["end"] = None

    return df


def generate_crowdsourcing_campaign_db(app, campaign_data, config):
    # load all outputs
    all_examples = []

    examples_per_batch = config["examples_per_batch"]
    sort_order = config["sort_order"]

    for c in campaign_data:
        for i in get_output_ids(c["dataset"], c["split"], c["setup_id"]):
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
    df["start"] = None
    df["end"] = None

    return df


def parse_campaign_config(config):
    def parse_value(value):
        try:
            # Try to parse the value as a literal (int, list, dict, etc.)
            parsed_value = ast.literal_eval(value)
            return parsed_value
        except (ValueError, SyntaxError):
            # If parsing fails, return the value as a string
            return value

    parsed_config = {key: parse_value(value) for key, value in config.items()}
    return parsed_config


def set_dataset_enabled(app, dataset_id, enabled):
    config = utils.load_dataset_config()
    config[dataset_id]["enabled"] = enabled

    if enabled:
        dataset = instantiate_dataset(dataset_id, config[dataset_id])
        app.db["datasets_obj"][dataset_id] = dataset
    else:
        app.db["datasets_obj"].pop(dataset_id, None)

    utils.save_dataset_config(config)


def get_local_dataset_overview(app):
    config = utils.load_dataset_config()
    overview = {}

    for dataset_id, dataset_config in config.items():
        class_name = dataset_config["class"]
        params = dataset_config.get("params", {})
        is_enabled = dataset_config.get("enabled", True)
        name = dataset_config.get("name", dataset_id)
        description = dataset_config.get("description", "")
        splits = dataset_config.get("splits", [])
        dataset_type = dataset_config.get("type", "default")

        if is_enabled:
            dataset = app.db["datasets_obj"].get(dataset_id)

            if dataset is None:
                logger.warning(f"Dataset {dataset_id} is enabled but not loaded, loading...")
                try:
                    dataset = instantiate_dataset(dataset_id, dataset_config)
                    app.db["datasets_obj"][dataset_id] = dataset
                except Exception as e:
                    logger.error(f"Error while loading dataset {dataset_id}")
                    traceback.print_exc()
                    continue

            example_count = {split: dataset.get_example_count(split) for split in dataset.get_splits()}
        else:
            example_count = {}

        overview[dataset_id] = {
            "class": class_name,
            "params": params,
            "enabled": is_enabled,
            "splits": splits,
            "name": name,
            "description": description,
            "example_count": example_count,
            "type": dataset_type,
        }

    return overview


def get_resources(app):
    config = utils.load_resources_config()

    return config


def download_dataset(app, dataset_id):
    config = utils.load_resources_config()
    dataset_config = config.get(dataset_id)

    if dataset_config is None:
        raise ValueError(f"Dataset {dataset_id} not found in the download config")

    submodule, class_name = dataset_config["class"].split(".")

    dataset_cls = get_dataset_class(submodule, class_name)
    download_dir = INPUT_DIR / dataset_id
    output_dir = OUTPUT_DIR
    campaign_dir = CAMPAIGN_DIR

    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    dataset_cls.download(
        dataset_id=dataset_id,
        data_download_dir=download_dir,
        out_download_dir=output_dir,
        annotation_download_dir=campaign_dir,
        splits=dataset_config["splits"],
        outputs=dataset_config.get("outputs", []),
        dataset_config=dataset_config,
    )

    # add an entry in the dataset config
    config = utils.load_dataset_config()

    config[dataset_id] = {
        "class": dataset_config["class"],
        "name": dataset_config.get("name", dataset_id),
        "description": dataset_config.get("description", ""),
        "splits": dataset_config["splits"],
        "enabled": True,
    }

    dataset = instantiate_dataset(dataset_id, config[dataset_id])
    app.db["datasets_obj"][dataset_id] = dataset

    utils.save_dataset_config(config)

    return dataset


def delete_dataset(app, dataset_id):
    config = utils.load_dataset_config()
    config.pop(dataset_id, None)
    utils.save_dataset_config(config)

    # remove the data directory
    shutil.rmtree(f"factgenie/data/{dataset_id}", ignore_errors=True)

    delete_model_outputs(dataset_id, None, None)

    app.db["datasets_obj"].pop(dataset_id, None)


def export_dataset(app, dataset_id):
    zip_buffer = BytesIO()
    data_path = INPUT_DIR / dataset_id

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

    # assemble relevant outputs
    output_index = generate_output_index(app)
    outputs = output_index[
        (output_index["dataset"] == dataset_id)
        & (output_index["split"] == split)
        & (output_index["setup_id"] == setup_id)
    ]
    # write the outputs to a temporary JSONL file
    tmp_file_path = tempfile.mktemp()

    with open(tmp_file_path, "w") as f:
        for _, row in outputs.iterrows():
            j = row.to_dict()
            f.write(json.dumps(j) + "\n")

    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        zip_file.write(
            tmp_file_path,
            f"{dataset_id}-{split}-{setup_id}.jsonl",
        )

    # Set response headers for download
    response = make_response(zip_buffer.getvalue())
    response.headers["Content-Type"] = "application/zip"
    response.headers["Content-Disposition"] = f"attachment; filename={dataset_id}_{split}_{setup_id}.zip"

    return response


def get_dataset_class(submodule, class_name):
    # Dynamically import the class
    module = importlib.import_module("factgenie.datasets")
    submodule = getattr(module, submodule)
    dataset_class = getattr(submodule, class_name)

    return dataset_class


def instantiate_dataset(dataset_id, dataset_config):
    submodule, class_name = dataset_config["class"].split(".")

    dataset_class = get_dataset_class(submodule, class_name)

    return dataset_class(dataset_id, **dataset_config)


def instantiate_datasets():
    config = utils.load_dataset_config()
    datasets = {}

    for dataset_id, dataset_config in config.items():
        is_enabled = dataset_config.get("enabled", True)

        if not is_enabled:
            continue

        try:
            datasets[dataset_id] = instantiate_dataset(dataset_id, dataset_config)
        except Exception as e:
            logger.error(f"Error while loading dataset {dataset_id}")
            traceback.print_exc()

    return datasets


def upload_dataset(app, dataset_id, dataset_name, dataset_description, dataset_format, dataset_data):
    params = {
        "text": {"suffix": "txt", "class": "basic.PlainTextDataset", "type": "default"},
        "jsonl": {"suffix": "jsonl", "class": "basic.JSONLDataset", "type": "json"},
        "csv": {"suffix": "csv", "class": "basic.CSVDataset", "type": "table"},
        "html": {"suffix": "zip", "class": "basic.HTMLDataset", "type": "default"},
    }
    data_dir = INPUT_DIR / dataset_id
    os.makedirs(data_dir, exist_ok=True)

    # slugify all split names
    dataset_data = {slugify(k): v for k, v in dataset_data.items()}

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
    config = utils.load_dataset_config()
    config[dataset_id] = {
        "name": dataset_name,
        "class": params[dataset_format]["class"],
        "description": dataset_description,
        "splits": list(dataset_data.keys()),
        "enabled": True,
    }
    utils.save_dataset_config(config)

    app.db["datasets_obj"][dataset_id] = instantiate_dataset(dataset_id, config[dataset_id])


def upload_model_outputs(dataset, split, setup_id, model_outputs):
    path = Path(OUTPUT_DIR) / dataset.id
    path.mkdir(parents=True, exist_ok=True)

    generated = model_outputs.strip().split("\n")
    setup_id = slugify(setup_id)

    if len(generated) != len(dataset.examples[split]):
        raise ValueError(
            f"Output count mismatch for {setup_id} in {split}: {len(generated)} vs {len(dataset.examples[split])}"
        )

    with open(f"{path}/{split}-{setup_id}.jsonl", "w") as f:
        for i, out in enumerate(generated):
            j = {
                "dataset": dataset.id,
                "split": split,
                "setup_id": setup_id,
                "example_idx": i,
                "out": out,
            }
            f.write(json.dumps(j) + "\n")

    with open(f"{path.parent}/metadata.json", "w") as f:
        json.dump(
            {
                "id": setup_id,
                "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=4,
        )


def delete_model_outputs(dataset, split=None, setup_id=None):
    path = Path(OUTPUT_DIR)

    # look through all JSON files in the output directory
    for file in path.glob("*.jsonl"):
        new_lines = []

        with open(file) as f:
            for line in f:
                j = json.loads(line)

                # None means all
                if (split is None or j["split"] == split) and (setup_id is None or j["setup_id"] == setup_id):
                    continue

                new_lines.append(line)

        if len(new_lines) == 0:
            os.remove(file)
        else:
            with open(file, "w") as f:
                f.writelines(new_lines)


def create_llm_campaign(mode, campaign_id, config, campaign_data, datasets, overwrite=False):
    campaign_id = slugify(campaign_id)

    # create a new directory
    if os.path.exists(os.path.join(CAMPAIGN_DIR, campaign_id)):
        if not overwrite:
            raise ValueError(f"Campaign {campaign_id} already exists")
        else:
            shutil.rmtree(os.path.join(CAMPAIGN_DIR, campaign_id))

    try:
        os.makedirs(os.path.join(CAMPAIGN_DIR, campaign_id, "files"), exist_ok=True)

        # create the annotation CSV
        db = generate_llm_campaign_db(mode, datasets, campaign_id, campaign_data)
        db_path = os.path.join(CAMPAIGN_DIR, campaign_id, "db.csv")
        logger.info(f"DB with {len(db)} free examples created for {campaign_id} at {db_path}")
        db.to_csv(db_path, index=False)

        # save metadata
        metadata_path = os.path.join(CAMPAIGN_DIR, campaign_id, "metadata.json")
        logger.info(f"Metadata for {campaign_id} saved at {metadata_path}")
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "id": campaign_id,
                    "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "mode": mode,
                    "status": CampaignStatus.IDLE,
                    "config": config,
                },
                f,
                indent=4,
            )
    except Exception as e:
        # cleanup
        shutil.rmtree(os.path.join(CAMPAIGN_DIR, campaign_id))
        os.remove(os.path.join(OUTPUT_DIR, campaign_id))
        raise e


def generate_default_id(app, mode, prefix):
    campaign_list = get_sorted_campaign_list(app, modes=[mode])

    i = 1
    default_campaign_id = f"{prefix}-{i}"
    while default_campaign_id in campaign_list.keys():
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
    save_dir = os.path.join(CAMPAIGN_DIR, campaign_id, "files")
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
    save_dir = os.path.join(CAMPAIGN_DIR, campaign_id, "files")
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


def duplicate_llm_campaign(app, mode, campaign_id, new_campaign_id):
    # copy the directory except for the annotations
    old_campaign_dir = os.path.join(CAMPAIGN_DIR, campaign_id)
    new_campaign_dir = os.path.join(CAMPAIGN_DIR, new_campaign_id)

    # if new campaign dir exists, return error
    if os.path.exists(new_campaign_dir):
        return utils.error("Campaign already exists")

    shutil.copytree(old_campaign_dir, new_campaign_dir, ignore=shutil.ignore_patterns("files"))

    # copy the db
    old_db = pd.read_csv(os.path.join(old_campaign_dir, "db.csv"))
    new_db = old_db.copy()
    new_db["status"] = ExampleStatus.FREE

    # clean the columns
    new_db["annotator_id"] = ""
    new_db["start"] = None
    new_db["end"] = None

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

    # if it is a human campaign, copy also the symlink to the annotate page
    if metadata["mode"] == CampaignMode.CROWDSOURCING:
        shutil.copy(
            os.path.join(TEMPLATES_DIR, "campaigns", campaign_id, "annotate.html"),
            os.path.join(TEMPLATES_DIR, "campaigns", new_campaign_id, "annotate.html"),
        )

    return utils.success()


def save_config(filename, config, mode):
    # https://github.com/yaml/pyyaml/issues/121#issuecomment-1018117110
    def yaml_multiline_string_pipe(dumper, data):
        text_list = [line.rstrip() for line in data.splitlines()]
        fixed_data = "\n".join(text_list)
        if len(text_list) > 1:
            return dumper.represent_scalar("tag:yaml.org,2002:str", fixed_data, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", fixed_data)

    yaml.add_representer(str, yaml_multiline_string_pipe)

    if mode == CampaignMode.LLM_EVAL:
        save_dir = LLM_EVAL_CONFIG_DIR
    elif mode == CampaignMode.LLM_GEN:
        save_dir = LLM_GEN_CONFIG_DIR
    else:
        save_dir = CROWDSOURCING_CONFIG_DIR

    with open(os.path.join(save_dir, filename), "w") as f:
        yaml.dump(config, f, indent=2, allow_unicode=True)


def save_annotations(app, campaign_id, annotation_set, annotator_id):
    now = int(time.time())

    save_dir = os.path.join(CAMPAIGN_DIR, campaign_id, "files")
    os.makedirs(save_dir, exist_ok=True)
    campaign = load_campaign(app, campaign_id=campaign_id)

    with app.db["lock"]:
        db = campaign.db
        batch_idx = annotation_set[0]["batch_idx"]

        # if the batch is not assigned to this annotator, return an error
        batch_annotator_id = db.loc[db["batch_idx"] == batch_idx, "annotator_id"].iloc[0]

        if batch_annotator_id != annotator_id and annotator_id != PREVIEW_STUDY_ID:
            logger.info(
                f"Annotations rejected: batch {batch_idx} in {campaign_id} not assigned to annotator {annotator_id}"
            )
            return utils.error(f"Batch not assigned to annotator {annotator_id}")

        with open(os.path.join(save_dir, f"{batch_idx}-{annotator_id}-{now}.jsonl"), "w") as f:
            for row in annotation_set:
                f.write(json.dumps(row) + "\n")

        db.loc[db["batch_idx"] == batch_idx, "status"] = ExampleStatus.FINISHED
        db.loc[db["batch_idx"] == batch_idx, "end"] = now

        campaign.update_db(db)
        logger.info(f"Annotations for {campaign_id} (batch {batch_idx}) saved")

    final_message_html = markdown.markdown(campaign.metadata["config"]["final_message"])

    if annotator_id == PREVIEW_STUDY_ID:
        preview_message = f'<div class="alert alert-info" role="alert"><p>You are in a preview mode. Click <a href="{app.config["host_prefix"]}/crowdsourcing"><b>here</b></a> to go back to the campaign view.</p><p><i>This message will not be displayed to the annotators.</i></p></div>'

        return utils.success(message=final_message_html + preview_message)

    return utils.success(message=final_message_html)


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
        "final_message": config.get("finalMessage"),
        "examples_per_batch": int(config.get("examplesPerBatch")),
        "idle_time": int(config.get("idleTime")),
        "annotation_granularity": config.get("annotationGranularity"),
        "service": config.get("service"),
        "sort_order": config.get("sortOrder"),
        "annotation_span_categories": config.get("annotationSpanCategories"),
        "flags": config.get("flags"),
        "options": config.get("options"),
        "text_fields": config.get("textFields"),
    }

    return config


def create_crowdsourcing_campaign(app, campaign_id, config, campaign_data):
    # create a new directory
    if os.path.exists(os.path.join(CAMPAIGN_DIR, campaign_id)):
        return jsonify({"error": "Campaign already exists"})

    try:
        os.makedirs(os.path.join(CAMPAIGN_DIR, campaign_id, "files"), exist_ok=True)

        # create the annotation CSV
        db = generate_crowdsourcing_campaign_db(app, campaign_data, config=config)
        db.to_csv(os.path.join(CAMPAIGN_DIR, campaign_id, "db.csv"), index=False)

        # save metadata
        with open(os.path.join(CAMPAIGN_DIR, campaign_id, "metadata.json"), "w") as f:
            json.dump(
                {
                    "id": campaign_id,
                    "mode": CampaignMode.CROWDSOURCING,
                    "config": config,
                    "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                f,
                indent=4,
            )

        # prepare the crowdsourcing HTML page
        create_crowdsourcing_page(campaign_id, config)

        load_campaign(app, campaign_id)
    except Exception as e:
        # cleanup
        shutil.rmtree(os.path.join(CAMPAIGN_DIR, campaign_id))
        raise e


def generate_flags(flags):
    if not flags:
        return ""

    flags_segment = "<div class='mb-4'><p><b>Please check if you agree with any of the following statements:</b></p>"
    for i, flag in enumerate(flags):
        flags_segment += f"""
            <div class="form-check crowdsourcing-flag">
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
                    <div><label for="select-{i}"><b>{option["label"]}</b></label></div>
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
                    <div><label for="slider-{i}"><b>{option["label"]}</b></label></div>
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


def generate_text_fields(text_fields):
    if not text_fields:
        return ""

    text_fields_segment = "<div class='mt-2 mb-3'>"
    for i, text_field in enumerate(text_fields):
        text_fields_segment += f"""
            <div class="form-group crowdsourcing-text mb-4">
                <label for="textbox-{i}"><b>{text_field}</b></label>
                <input type="text" class="form-control textbox-crowdsourcing" id="textbox-crowdsourcing-{i}">
            </div>
        """
    text_fields_segment += "</div>"
    return text_fields_segment


def create_crowdsourcing_page(campaign_id, config):
    final_page_path = os.path.join(CAMPAIGN_DIR, campaign_id, "pages", "annotate.html")
    symlink_path = os.path.join(TEMPLATES_DIR, "campaigns", campaign_id, "annotate.html")

    os.makedirs(os.path.dirname(final_page_path), exist_ok=True)
    os.makedirs(os.path.dirname(symlink_path), exist_ok=True)

    # assemble the crowdsourcing page
    parts = []
    for part in ["header", "body", "footer"]:
        part_path = os.path.join(TEMPLATES_DIR, CampaignMode.CROWDSOURCING, "annotate_{}.html".format(part))

        with open(part_path, "r") as f:
            parts.append(f.read())

    instructions_html = markdown.markdown(config["annotator_instructions"])

    # format only the body, keeping the unfilled templates in header and footer
    template = Template(parts[1])

    rendered_content = template.render(
        instructions=instructions_html,
        annotation_span_categories=config.get("annotation_span_categories", []),
        flags=generate_flags(config.get("flags", [])),
        options=generate_options(config.get("options", [])),
        text_fields=generate_text_fields(config.get("text_fields", [])),
    )

    # concatenate with header and footer
    content = parts[0] + rendered_content + parts[2]

    with open(final_page_path, "w") as f:
        f.write(content)

    # create a symlink to the page in the templates folder
    os.symlink(final_page_path, symlink_path)


def pause_llm_campaign(app, campaign_id):
    if campaign_id in app.db["running_campaigns"]:
        app.db["running_campaigns"].remove(campaign_id)

    campaign = load_campaign(app, campaign_id=campaign_id)
    campaign.metadata["status"] = CampaignStatus.IDLE
    campaign.update_metadata()


def announce(announcer, payload):
    msg = utils.format_sse(data=json.dumps(payload))
    if announcer is not None:
        announcer.announce(msg=msg)


def run_llm_campaign(mode, campaign_id, announcer, campaign, datasets, model, running_campaigns):
    start_time = int(time.time())

    # set metadata status
    campaign.metadata["status"] = CampaignStatus.RUNNING
    campaign.update_metadata()
    db = campaign.db

    logger.info(f"Starting LLM campaign {campaign_id}")

    for i, row in db.iterrows():
        # campaign was paused
        if campaign_id not in running_campaigns:
            break

        if row["status"] == ExampleStatus.FINISHED:
            continue

        db.loc[i, "start"] = float(time.time())

        dataset_id = row["dataset"]
        split = row["split"]
        setup_id = row.get("setup_id")
        example_idx = row["example_idx"]

        dataset = datasets[dataset_id]
        example = dataset.get_example(split, example_idx)

        announce(
            announcer,
            {
                "campaign_id": campaign_id,
                "type": "status",
                "message": f"Example {example_idx}: Waiting for model response",
            },
        )

        if mode == CampaignMode.LLM_EVAL:
            generated_output = get_output(dataset_id, split, setup_id, example_idx)

            if not generated_output:
                return utils.error(
                    f"Model output not found for dataset {dataset_id}, split {split}, example {example_idx}, setup {setup_id}"
                )
            output = model.annotate_example(example, generated_output)

        elif mode == CampaignMode.LLM_GEN:
            output = model.generate_output(example)

        if "error" in output:
            return utils.error(output["error"])

        if mode == CampaignMode.LLM_EVAL:
            annotator_id = model.get_annotator_id()

            record = save_annotation(
                annotator_id, campaign_id, dataset_id, split, setup_id, example_idx, output, start_time
            )
            # frontend adjustments
            record["output"] = record.pop("annotations")
        elif mode == CampaignMode.LLM_GEN:
            record = save_output(campaign_id, dataset_id, split, example_idx, output, start_time)

            # frontend adjustments
            record["setup_id"] = setup_id
            record["output"] = record.pop("out")

        db.loc[i, "status"] = ExampleStatus.FINISHED
        db.loc[i, "end"] = float(time.time())
        campaign.update_db(db)

        stats = campaign.get_stats()
        finished_examples_cnt = stats["finished"]
        payload = {"campaign_id": campaign_id, "stats": stats, "type": "result", "annotation": record}

        announce(announcer, payload)

        logger.info(f"{campaign_id}: {finished_examples_cnt}/{len(db)} examples")

    final_message = ""
    # if all fields are finished, set the metadata to finished
    if len(db.status.unique()) == 1 and db.status.unique()[0] == ExampleStatus.FINISHED:
        campaign.metadata["status"] = CampaignStatus.FINISHED
        campaign.update_metadata()

        if campaign_id in running_campaigns:
            running_campaigns.remove(campaign_id)

        if mode == CampaignMode.LLM_EVAL:
            final_message = (
                f"All examples have been annotated. You can find the annotations in {CAMPAIGN_DIR}/{campaign_id}/files."
            )
        elif mode == CampaignMode.LLM_GEN:
            final_message = (
                f"All examples have been generated. You can find the outputs in {CAMPAIGN_DIR}/{campaign_id}/files."
            )
    else:
        logger.warning("Spurious exit from the loop")

    return jsonify(success=True, status=campaign.metadata["status"], final_message=final_message)


def save_generation_outputs(app, campaign_id, setup_id):
    """
    Load the files from the `GENERATIONS_DIR` and save them in the appropriate subdirectory in `OUTPUT_DIR`.
    """

    campaign = load_campaign(app, campaign_id)
    metadata = campaign.metadata

    # load the metadata
    with open(CAMPAIGN_DIR / campaign_id / "metadata.json") as f:
        metadata = json.load(f)

    # load the outputs
    outputs = []

    for file in os.listdir(CAMPAIGN_DIR / campaign_id / "files"):
        if file.endswith(".jsonl"):
            with open(CAMPAIGN_DIR / campaign_id / "files" / file) as f:
                for line in f:
                    record = json.loads(line)
                    # replace the campaign_id with the desired setup_id
                    record["setup_id"] = setup_id
                    record["metadata"] = metadata

                    outputs.append(record)

    # save the outputs
    path = OUTPUT_DIR / setup_id
    os.makedirs(path, exist_ok=True)

    with open(path / f"{setup_id}.jsonl", "w") as f:
        for example in outputs:
            f.write(json.dumps(example) + "\n")

    return utils.success()
