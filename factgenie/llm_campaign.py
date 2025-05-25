#!/usr/bin/env python3

import ast
import datetime
import json
import logging
import os
import shutil
import time
import traceback

import pandas as pd
import requests
import urllib3
from flask import jsonify
from slugify import slugify

import factgenie.utils as utils
import factgenie.workflows as workflows
from factgenie import CAMPAIGN_DIR, OUTPUT_DIR, TEMPLATES_DIR
from factgenie.campaign import CampaignMode, CampaignStatus, ExampleStatus

logger = logging.getLogger("factgenie")


def create_llm_campaign(app, mode, campaign_id, config, campaign_data, datasets, overwrite=False):
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
        db = generate_llm_campaign_db(app, mode, datasets, campaign_id, campaign_data)
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

    return utils.success()


def generate_llm_campaign_db(app, mode, datasets, campaign_id, campaign_data):
    # load all outputs
    all_examples = []

    for c in campaign_data:
        dataset = datasets[c["dataset"]]

        # for llm_gen, setup_id based on the generation model
        if mode == CampaignMode.LLM_EVAL:
            setup_id = c["setup_id"]
            ids = workflows.get_output_ids(app, dataset.id, c["split"], setup_id)
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
    df["annotator_id"] = ""
    df["annotator_group"] = 0
    df["status"] = ExampleStatus.FREE
    df["start"] = None
    df["end"] = None

    return df


def run_llm_campaign(app, mode, campaign_id, announcer, campaign, datasets, model, running_campaigns):
    db = campaign.db

    # set campaign status to running
    campaign.metadata["status"] = CampaignStatus.RUNNING
    campaign.metadata["last_run"] = int(time.time())
    campaign.update_metadata()

    provider = campaign.metadata["config"].get("api_provider", None)

    logger.info(f"Starting LLM campaign \033[1m{campaign_id}\033[0m | {provider}")

    logger.info(f"=" * 50)
    logger.info(f"\033[1mModel\033[0m: {campaign.metadata['config']['model']}")
    logger.info(f"\033[1mAPI URL\033[0m: {campaign.metadata['config'].get('api_url')}")
    logger.info(f"\033[1mModel args\033[0m: {campaign.metadata['config'].get('model_args')}")
    logger.info(f"\033[1mSystem message\033[0m: \"{campaign.metadata['config'].get('system_msg')}\"")
    logger.info(f"\033[1mAnnotation span categories\033[0m:")

    for cat in campaign.metadata["config"].get("annotation_span_categories", []):
        logger.info(f"[{cat['name']}] {cat['description']}")

    logger.info(f"=" * 50)

    # regenerate output index
    workflows.get_output_index(app, force_reload=True)

    # generate outputs / annotations for all free examples in the db
    for i, row in db[db.status == ExampleStatus.FREE].iterrows():
        # campaign was paused
        if campaign_id not in running_campaigns:
            break

        dataset_id = row["dataset"]
        split = row["split"]
        example_idx = row["example_idx"]
        example = datasets[dataset_id].get_example(split, example_idx)
        # only for llm_eval
        setup_id = row.get("setup_id")

        db.loc[i, "start"] = float(time.time())
        db.loc[i, "annotator_id"] = campaign.metadata["config"]["model"] + "-" + campaign_id

        # generate output or annotate example
        try:
            if mode == CampaignMode.LLM_EVAL:
                generated_output = workflows.get_output_for_setup(
                    dataset_id, split, example_idx, setup_id, app=app, force_reload=False
                )
                res = model.generate_output(data=example, text=generated_output["output"])
                # keep the annotated text in the object
                res["output"] = generated_output["output"]
            elif mode == CampaignMode.LLM_GEN:
                res = model.generate_output(data=example)
        except requests.exceptions.ConnectionError as e:
            traceback.print_exc()
            return utils.error(
                f"Error processing example {dataset_id}-{split}-{example_idx}: {e.__class__.__name__}: {str(e)}\n"
            )

        except Exception as e:
            traceback.print_exc()
            return utils.error(
                f"Error processing example {dataset_id}-{split}-{example_idx}: {e.__class__.__name__}: {str(e)}"
            )

        # update the DB
        db.loc[i, "end"] = float(time.time())
        db.loc[i, "status"] = ExampleStatus.FINISHED

        campaign.update_db(db)

        # save the record to a JSONL file
        response = workflows.save_record(
            mode=mode,
            campaign=campaign,
            row=db.loc[i],
            result=res,
        )

        # send a response to the frontend
        stats = campaign.get_stats()
        payload = {"campaign_id": campaign_id, "stats": stats, "type": "result", "response": response}

        utils.announce(announcer, payload)
        logger.info(f"-" * 50)
        logger.info(f"{campaign_id}: {stats['finished']}/{stats['total']} examples")
        logger.info(f"-" * 50)

    # if all examples are finished, set the campaign status to finished
    if len(db.status.unique()) == 1 and db.status.unique()[0] == ExampleStatus.FINISHED:
        campaign.metadata["status"] = CampaignStatus.FINISHED
        campaign.update_metadata()

        if campaign_id in running_campaigns:
            running_campaigns.remove(campaign_id)

    return jsonify(success=True, status=campaign.metadata["status"])


def pause_llm_campaign(app, campaign_id):
    if campaign_id in app.db["running_campaigns"]:
        app.db["running_campaigns"].remove(campaign_id)

    campaign = workflows.load_campaign(app, campaign_id=campaign_id)
    campaign.metadata["status"] = CampaignStatus.IDLE
    campaign.update_metadata()


def parse_llm_gen_config(config):
    config = {
        "api_provider": config.get("apiProvider"),
        "model": config.get("modelName"),
        "prompt_strat": config.get("promptStrat"),
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
        "api_provider": config.get("apiProvider"),
        "model": config.get("modelName"),
        "prompt_strat": config.get("promptStrat"),
        "prompt_template": config.get("promptTemplate"),
        "system_msg": config.get("systemMessage"),
        "annotation_overlap_allowed": config.get("annotationOverlapAllowed", False),
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

                    outputs.append(record)

    # save the outputs
    path = OUTPUT_DIR / setup_id
    os.makedirs(path, exist_ok=True)

    with open(path / f"{campaign_id}.jsonl", "w") as f:
        for example in outputs:
            f.write(json.dumps(example) + "\n")

    return utils.success()
