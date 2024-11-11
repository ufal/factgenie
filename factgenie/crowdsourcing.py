#!/usr/bin/env python3

import datetime
import json
import shutil
import random
import time
import logging
import pandas as pd
import os
import markdown

from flask import jsonify
from jinja2 import Template
import factgenie.utils as utils
import factgenie.workflows as workflows
from factgenie import CAMPAIGN_DIR, PREVIEW_STUDY_ID, TEMPLATES_DIR
from factgenie.campaigns import CampaignMode, ExampleStatus

logger = logging.getLogger(__name__)


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

        workflows.load_campaign(app, campaign_id)
    except Exception as e:
        # cleanup
        shutil.rmtree(os.path.join(CAMPAIGN_DIR, campaign_id))
        raise e


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
    if os.path.exists(symlink_path):
        os.remove(symlink_path)

    os.symlink(final_page_path, symlink_path)


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


def generate_crowdsourcing_campaign_db(app, campaign_data, config):
    # load all outputs
    all_examples = []

    examples_per_batch = config["examples_per_batch"]
    annotators_per_example = config["annotators_per_example"]
    sort_order = config["sort_order"]

    for c in campaign_data:
        for i in workflows.get_output_ids(app, c["dataset"], c["split"], c["setup_id"]):
            all_examples.append(
                {
                    "dataset": c["dataset"],
                    "split": c["split"],
                    "example_idx": i,
                    "setup_id": c["setup_id"],
                }
            )

    # deterministic shuffling
    random.seed(42)

    # shuffle all examples and setups
    if sort_order == "shuffle-all":
        random.shuffle(all_examples)
    # sort examples by example_idx, shuffle setups
    elif sort_order == "sort-example-ids-shuffle-setups":
        random.shuffle(all_examples)
        all_examples = sorted(all_examples, key=lambda x: (x["example_idx"], x["dataset"], x["split"]))
    # sort examples by example_idx, keep the setup order
    elif sort_order == "sort-example-ids-keep-setups":
        all_examples = sorted(all_examples, key=lambda x: (x["example_idx"], x["dataset"], x["split"]))
    # keep all examples and setups in the default order
    elif sort_order == "keep-all":
        pass
    else:
        raise ValueError(f"Unknown sort order {sort_order}")

    df = pd.DataFrame.from_records(all_examples)

    # create a column for batch index and assign each example to a batch
    df["batch_idx"] = df.index // examples_per_batch
    df["annotator_id"] = ""
    df["status"] = ExampleStatus.FREE
    df["start"] = None
    df["end"] = None

    # if we have multiple `annotators_per_example`, repeat the dataframe `annotators_per_example` times, assigning an increasing index of `annotator_group` to each repeating dataframe
    if annotators_per_example > 1:
        df = pd.concat([df] * annotators_per_example, ignore_index=True)

    df["annotator_group"] = df.index // len(all_examples)

    return df


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


def parse_crowdsourcing_config(config):
    # parse Nones or empty strings
    examples_per_batch = config.get("examplesPerBatch")
    examples_per_batch = int(examples_per_batch) if examples_per_batch else 10
    annotators_per_example = config.get("annotatorsPerExample")
    annotators_per_example = int(annotators_per_example) if annotators_per_example else 1
    idle_time = config.get("idleTime")
    idle_time = int(idle_time) if idle_time else 120
    config = {
        "annotator_instructions": config.get("annotatorInstructions", "No instructions needed :)"),
        "final_message": config.get("finalMessage"),
        "examples_per_batch": int(examples_per_batch),
        "annotators_per_example": int(annotators_per_example),
        "idle_time": int(idle_time),
        "annotation_granularity": config.get("annotationGranularity"),
        "service": config.get("service"),
        "sort_order": config.get("sortOrder"),
        "annotation_span_categories": config.get("annotationSpanCategories"),
        "flags": config.get("flags"),
        "options": config.get("options"),
        "text_fields": config.get("textFields"),
    }

    return config


def select_batch(db, seed):
    # Choose from the batches with the most available annotators

    free_batches = db[db["status"] == ExampleStatus.FREE]
    annotator_groups_per_batch = free_batches.groupby("batch_idx")["annotator_group"].nunique()
    eligible_batches = annotator_groups_per_batch[annotator_groups_per_batch == annotator_groups_per_batch.max()]

    eligible_examples = free_batches[free_batches["batch_idx"].isin(eligible_batches.index)]

    # Randomly select an example (with its batch) from the eligible ones
    if not eligible_examples.empty:
        selected_example = eligible_examples.sample(n=1, random_state=seed).iloc[0]
        selected_batch_idx = selected_example["batch_idx"]

        # Get the lowest annotator group for the selected batch which is free
        selected_annotator_group = db[(db["batch_idx"] == selected_batch_idx) & (db["status"] == ExampleStatus.FREE)][
            "annotator_group"
        ].min()

        logging.info(f"Selected batch {selected_batch_idx} (annotator group {selected_annotator_group})")
        return selected_batch_idx, selected_annotator_group
    else:
        raise ValueError("No available batches")


def get_examples_for_batch(db, batch_idx, annotator_group):
    annotator_batch = []

    # find all examples for this batch and annotator group
    batch_examples = db[(db["batch_idx"] == batch_idx) & (db["annotator_group"] == annotator_group)]

    for _, row in batch_examples.iterrows():
        annotator_batch.append(
            {
                "dataset": row["dataset"],
                "split": row["split"],
                "setup_id": row["setup_id"],
                "example_idx": row["example_idx"],
                "batch_idx": row["batch_idx"],
                "annotator_group": row["annotator_group"],
            }
        )

    return annotator_batch


def get_annotator_batch(app, campaign, service_ids, batch_idx=None):
    db = campaign.db

    # simple locking over the CSV file to prevent double writes
    with app.db["lock"]:
        annotator_id = service_ids["annotator_id"]

        logging.info(f"Acquiring lock for {annotator_id}")
        start = int(time.time())
        seed = random.seed(str(start) + str(service_ids.values()))

        if not batch_idx:
            # usual case: an annotator opened the annotation page, we need to select the batch
            try:
                batch_idx, annotator_group = select_batch(db, seed)
            except ValueError:
                # no available batches
                return []
        else:
            # preview mode
            batch_idx = int(batch_idx)
            annotator_group = 0

        mask = (db["batch_idx"] == batch_idx) & (db["annotator_group"] == annotator_group)

        # we do not block the example if we are in preview mode
        if annotator_id != PREVIEW_STUDY_ID:
            db.loc[mask, "status"] = ExampleStatus.ASSIGNED

        db.loc[mask, "start"] = start
        db.loc[mask, "annotator_id"] = annotator_id
        campaign.update_db(db)

        annotator_batch = get_examples_for_batch(db, batch_idx, annotator_group)
        logging.info(f"Releasing lock for {annotator_id}")

    return annotator_batch


def save_annotations(app, campaign_id, annotation_set, annotator_id):
    now = int(time.time())

    save_dir = os.path.join(CAMPAIGN_DIR, campaign_id, "files")
    os.makedirs(save_dir, exist_ok=True)
    campaign = workflows.load_campaign(app, campaign_id=campaign_id)

    with app.db["lock"]:
        db = campaign.db
        batch_idx = annotation_set[0]["batch_idx"]
        annotator_group = annotation_set[0]["annotator_group"]

        # select the examples for this batch and annotator group
        mask = (db["batch_idx"] == batch_idx) & (db["annotator_group"] == annotator_group)

        # if the batch is not assigned to this annotator, return an error
        batch_annotator_id = db.loc[mask].iloc[0]["annotator_id"]

        if batch_annotator_id != annotator_id and annotator_id != PREVIEW_STUDY_ID:
            logger.info(
                f"Annotations rejected: batch {batch_idx} in {campaign_id} not assigned to annotator {annotator_id}"
            )
            return utils.error(f"Batch not assigned to annotator {annotator_id}")

        # update the db
        db.loc[mask, "status"] = ExampleStatus.FINISHED
        db.loc[mask, "end"] = now
        campaign.update_db(db)

        # save the annotations
        for i, ann in enumerate(annotation_set):
            row = db.loc[mask].iloc[i]

            # retrieve the related model output to save it with the annotations
            output = workflows.get_output_for_setup(
                dataset=row["dataset"],
                split=row["split"],
                setup_id=row["setup_id"],
                example_idx=row["example_idx"],
                app=app,
                force_reload=False,
            )["output"]

            res = {
                "annotations": ann["annotations"],
                "flags": ann["flags"],
                "options": ann["options"],
                "text_fields": ann["textFields"],
                "output": output,
            }
            # save the record to a JSONL file
            workflows.save_record(
                mode=CampaignMode.CROWDSOURCING,
                campaign=campaign,
                row=row,
                result=res,
            )
        logger.info(
            f"Annotations for {campaign_id} (batch {batch_idx}, annotator group {annotator_group}, annotator {annotator_id}) saved."
        )

    final_message_html = markdown.markdown(campaign.metadata["config"]["final_message"])

    if annotator_id == PREVIEW_STUDY_ID:
        preview_message = f'<div class="alert alert-info" role="alert"><p>You are in a preview mode. Click <a href="{app.config["host_prefix"]}/crowdsourcing"><b>here</b></a> to go back to the campaign view.</p><p><i>This message will not be displayed to the annotators.</i></p></div>'

        return utils.success(message=final_message_html + preview_message)

    return utils.success(message=final_message_html)
