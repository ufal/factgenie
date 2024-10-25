#!/usr/bin/env python3

import pandas as pd
import datetime
import shutil
import time
import json
import os
import ast
import logging
import traceback

from slugify import slugify
from factgenie.campaigns import CampaignMode, CampaignStatus, ExampleStatus
from flask import jsonify
from factgenie.crowdsourcing import save_annotation
import factgenie.utils as utils
import factgenie.workflows as workflows

from factgenie import CAMPAIGN_DIR, OUTPUT_DIR, TEMPLATES_DIR

logger = logging.getLogger(__name__)


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
        os.makedirs(os.path.join(TEMPLATES_DIR, "campaigns", new_campaign_id), exist_ok=True)
        shutil.copy(
            os.path.join(TEMPLATES_DIR, "campaigns", campaign_id, "annotate.html"),
            os.path.join(TEMPLATES_DIR, "campaigns", new_campaign_id, "annotate.html"),
        )

    return utils.success()


def generate_llm_campaign_db(mode, datasets, campaign_id, campaign_data):
    # load all outputs
    all_examples = []

    for c in campaign_data:
        dataset = datasets[c["dataset"]]

        # for llm_gen, setup_id based on the generation model
        if mode == CampaignMode.LLM_EVAL:
            setup_id = c["setup_id"]
            ids = workflows.get_output_ids(dataset.id, c["split"], setup_id)
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

        utils.announce(
            announcer,
            {
                "campaign_id": campaign_id,
                "type": "status",
                "message": f"Example {example_idx}: Waiting for model response",
            },
        )

        if mode == CampaignMode.LLM_EVAL:
            generated_output = workflows.get_output_for_setup(dataset_id, split, example_idx, setup_id)

            generated_output = str(generated_output["out"]) if generated_output else None

            if generated_output is None:
                return utils.error(
                    f"Model output not found for dataset {dataset_id}, split {split}, example {example_idx}, setup {setup_id}"
                )
            try:
                output = model.annotate_example(example, generated_output)
            except Exception as e:
                traceback.print_exc()
                return utils.error(str(e))

        elif mode == CampaignMode.LLM_GEN:
            try:
                output = model.generate_output(example)
            except Exception as e:
                traceback.print_exc()
                return utils.error(str(e))

        if mode == CampaignMode.LLM_EVAL:
            annotator_id = model.get_annotator_id()

            record = save_annotation(
                annotator_id, campaign_id, dataset_id, split, setup_id, example_idx, output, start_time
            )
            # frontend adjustments
            record["output"] = record.pop("annotations")
        elif mode == CampaignMode.LLM_GEN:
            record = save_generated_output(campaign_id, dataset_id, split, example_idx, output, start_time)

            # frontend adjustments
            record["setup_id"] = setup_id
            record["output"] = record.pop("out")

        db.loc[i, "status"] = ExampleStatus.FINISHED
        db.loc[i, "end"] = float(time.time())
        campaign.update_db(db)

        stats = campaign.get_stats()
        finished_examples_cnt = stats["finished"]
        payload = {"campaign_id": campaign_id, "stats": stats, "type": "result", "annotation": record}

        utils.announce(announcer, payload)

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


def pause_llm_campaign(app, campaign_id):
    if campaign_id in app.db["running_campaigns"]:
        app.db["running_campaigns"].remove(campaign_id)

    campaign = workflows.load_campaign(app, campaign_id=campaign_id)
    campaign.metadata["status"] = CampaignStatus.IDLE
    campaign.update_metadata()


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


def save_generation_outputs(app, campaign_id, setup_id):
    """
    Load the files from the `GENERATIONS_DIR` and save them in the appropriate subdirectory in `OUTPUT_DIR`.
    """

    campaign = workflows.load_campaign(app, campaign_id)
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
                    # imprint campaign configuration
                    record["metadata"] = metadata["config"]

                    outputs.append(record)

    # save the outputs
    path = OUTPUT_DIR / setup_id
    os.makedirs(path, exist_ok=True)

    with open(path / f"{setup_id}.jsonl", "w") as f:
        for example in outputs:
            f.write(json.dumps(example) + "\n")

    return utils.success()


def save_generated_output(campaign_id, dataset_id, split, example_idx, output, start_time):
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
