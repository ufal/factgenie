#!/usr/bin/env python3

import datetime
import json
import logging
import os
import random
import shutil
import time

import markdown
import pandas as pd
from flask import jsonify
from jinja2 import Template

import factgenie.utils as utils
import factgenie.workflows as workflows
from factgenie import CAMPAIGN_DIR, PREVIEW_STUDY_ID, TEMPLATES_DIR
from factgenie.campaign import CampaignMode, ExampleStatus

logger = logging.getLogger("factgenie")


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

    os.makedirs(os.path.dirname(final_page_path), exist_ok=True)

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
        sliders=generate_sliders(config.get("sliders", [])),
        text_fields=generate_text_fields(config.get("text_fields", [])),
    )

    # concatenate with header and footer
    content = parts[0] + rendered_content + parts[2]

    with open(final_page_path, "w") as f:
        f.write(content)


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
        options_segment += f"""
            <div class="form-group crowdsourcing-option option-select mb-4">
                <div><label for="select-{i}"><b>{option["label"]}</b></label></div>
                <select class="form-select select-crowdsourcing mb-1" id="select-crowdsourcing-{i}">
                    <option value="" selected disabled>Select an option...</option>
        """
        for j, value in enumerate(option["values"]):
            options_segment += f"""<option class="select-crowdsourcing-{i}-value" value="{j}">{value}</option>
            """
        options_segment += """
                </select>
            </div>
        """

    return options_segment


def generate_sliders(sliders):
    if not sliders:
        return ""

    sliders_segment = "<div class='mt-2 mb-3'>"
    for i, slider in enumerate(sliders):
        sliders_segment += f"""
            <div class="form-group crowdsourcing-slider mb-4">
                <label for="slider-{i}"><b>{slider["label"]}</b></label>
                <input type="range" class="form-range slider-crowdsourcing" id="slider-crowdsourcing-{i}" min="{slider["min"]}" max="{slider["max"]}" step="{slider["step"]}">
                <div class="d-flex justify-content-between">
                    <div class="text-muted small"><span>{slider["min"]}</span></div>
                    <div><span id="slider-crowdsourcing-{i}-value" class="slider-crowdsourcing-value" data-default-value="?"></span></div>
                    <div class="text-muted small"><span>{slider["max"]}</span></div>
                </div>
            </div>
        """
    sliders_segment += "</div>"
    sliders_segment += """<script src="{{ host_prefix }}/static/js/render-sliders.js"></script>"""
    return sliders_segment


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

    # Create multiple copies of the dataframe for each annotator group
    dfs = []
    for annotator_group in range(annotators_per_example):
        df_copy = df.copy()
        df_copy["annotator_group"] = annotator_group
        # Adjust batch_idx for subsequent groups by adding offset
        df_copy["batch_idx"] += annotator_group * (len(df) // examples_per_batch + 1)
        dfs.append(df_copy)

    # Combine all dataframes
    df = pd.concat(dfs, ignore_index=True)

    df["annotator_id"] = ""
    df["status"] = ExampleStatus.FREE
    df["start"] = None
    df["end"] = None

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
        "annotation_overlap_allowed": config.get("annotationOverlapAllowed", False),
        "service": config.get("service"),
        "sort_order": config.get("sortOrder"),
        "annotation_span_categories": config.get("annotationSpanCategories"),
        "flags": config.get("flags"),
        "options": config.get("options"),
        "sliders": config.get("sliders"),
        "text_fields": config.get("textFields"),
    }

    return config


def select_batch(db, seed, annotator_id):
    # If the annotator already has any examples with ExampleStatus.ASSIGNED, return that batch
    if annotator_id != PREVIEW_STUDY_ID:
        if not db.loc[(db["annotator_id"] == annotator_id) & (db["status"] == ExampleStatus.ASSIGNED)].empty:
            assigned_batch = db.loc[
                (db["annotator_id"] == annotator_id) & (db["status"] == ExampleStatus.ASSIGNED)
            ].iloc[0]
            logging.info(f"Reusing batch {assigned_batch['batch_idx']}")
            return assigned_batch["batch_idx"]

    # Choose from the batches with the lowest annotator group
    free_batches = db[db["status"] == ExampleStatus.FREE]
    eligible_batches = free_batches.groupby("batch_idx")["annotator_group"].min()

    eligible_batches = eligible_batches[eligible_batches == eligible_batches.min()]

    eligible_examples = free_batches[free_batches["batch_idx"].isin(eligible_batches.index)]

    # Randomly select an example (with its batch) from the eligible ones
    if not eligible_examples.empty:
        selected_example = eligible_examples.sample(n=1, random_state=seed).iloc[0]
        selected_batch_idx = selected_example["batch_idx"]

        logging.info(f"Selected batch {selected_batch_idx}")
        return selected_batch_idx
    else:
        raise ValueError("No available batches")


def get_examples_for_batch(db, batch_idx):
    annotator_batch = []

    # find all examples for this batch and annotator group
    batch_examples = db[db["batch_idx"] == batch_idx]

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

        logger.info(f"Acquiring lock for {annotator_id}")
        start = int(time.time())
        seed = random.seed(str(start) + str(service_ids.values()))

        if not batch_idx:
            # usual case: an annotator opened the annotation page, we need to select the batch
            try:
                batch_idx = select_batch(db, seed, annotator_id)
            except ValueError as e:
                logger.info(str(e))
                # no available batches
                return []
        else:
            # preview mode with the specific batch
            batch_idx = int(batch_idx)

        mask = db["batch_idx"] == batch_idx

        # we do not block the example if we are in preview mode
        if annotator_id != PREVIEW_STUDY_ID:
            db.loc[mask, "status"] = ExampleStatus.ASSIGNED
            db.loc[mask, "start"] = start
            db.loc[mask, "annotator_id"] = annotator_id

            campaign.update_db(db)

        annotator_batch = get_examples_for_batch(db, batch_idx)
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

        # select the examples for this batch and annotator group
        mask = db["batch_idx"] == batch_idx

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

            annotations = ann["annotations"]
            # remove empty annotations
            annotations = [a for a in annotations if a["text"]]

            res = {
                "annotations": annotations,
                "flags": ann["flags"],
                "options": ann["options"],
                "sliders": ann["sliders"],
                "text_fields": ann["textFields"],
                "time_last_saved": ann.get("timeLastSaved"),
                "time_last_accessed": ann.get("timeLastAccessed"),
                "output": output,
            }
            # save the record to a JSONL file
            workflows.save_record(
                mode=CampaignMode.CROWDSOURCING,
                campaign=campaign,
                row=row,
                result=res,
            )
        logger.info(f"Annotations for {campaign_id} (batch {batch_idx}, annotator {annotator_id}) saved.")

    final_message_html = markdown.markdown(campaign.metadata["config"]["final_message"])

    if annotator_id == PREVIEW_STUDY_ID:
        preview_message = f'<div class="alert alert-info" role="alert"><p>You are in a preview mode. Click <a href="{app.config["host_prefix"]}/crowdsourcing"><b>here</b></a> to go back to the campaign view.</p><p><i>This message will not be displayed to the annotators.</i></p></div>'

        return utils.success(message=final_message_html + preview_message)

    return utils.success(message=final_message_html)
