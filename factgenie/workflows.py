#!/usr/bin/env python3
import datetime
import importlib
import json
import logging
import os
import shutil
import tempfile
import time
import traceback
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import pandas as pd
import yaml
from flask import make_response
from slugify import slugify

import factgenie.utils as utils
from factgenie import (
    CAMPAIGN_DIR,
    CROWDSOURCING_CONFIG_DIR,
    INPUT_DIR,
    LLM_EVAL_CONFIG_DIR,
    LLM_GEN_CONFIG_DIR,
    OUTPUT_DIR,
)
from factgenie.campaign import (
    CampaignMode,
    CampaignStatus,
    ExampleStatus,
    ExternalCampaign,
    HumanCampaign,
    LLMCampaignEval,
    LLMCampaignGen,
)

logger = logging.getLogger("factgenie")


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


def get_example_data(app, dataset_id, split, example_idx, setup_id=None):
    dataset = get_dataset(app=app, dataset_id=dataset_id)

    try:
        example = dataset.get_example(split=split, example_idx=example_idx)
    except:
        raise ValueError("Example cannot be retrieved from the dataset")

    try:
        html = dataset.render(example=example)

        if html is not None:
            # temporary solution for external files
            # prefix all the "/files" calls with "app.config["host_prefix"]"
            html = html.replace('src="/files', f'src="{app.config["host_prefix"]}/files')
    except:
        raise ValueError("Example cannot be rendered")

    if setup_id:
        generated_outputs = [
            get_output_for_setup(dataset_id, split, example_idx, setup_id, app=app, force_reload=False)
        ]
    else:
        generated_outputs = get_outputs(dataset_id, split, example_idx, app=app, force_reload=False)

    for i, output in enumerate(generated_outputs):
        setup_id = output["setup_id"]
        annotations = get_annotations(app, dataset_id, split, example_idx, setup_id)
        generated_outputs[i]["annotations"] = annotations

    return {
        "html": html,
        "raw_data": example,
        "generated_outputs": generated_outputs,
    }


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
        if not campaign_dir.is_dir():
            continue
        try:
            with open(campaign_dir / "metadata.json") as f:
                metadata = json.load(f)
            mode = metadata["mode"]
            campaign_id = slugify(metadata["id"])
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


def load_annotations_from_file(file_path, metadata):
    annotations_campaign = []

    with open(file_path) as f:
        for line in f:
            annotation_records = load_annotations_from_record(line, jsonl_file=file_path, metadata=metadata)
            annotations_campaign.append(annotation_records[0])

    return annotations_campaign


def create_annotation_example_record(jsonl_line, jsonl_file, metadata):
    # campaign metadata can be overwritten by annotation metadata
    ann_metadata = metadata["config"].copy()
    ann_metadata["campaign_id"] = metadata["id"]
    if "metadata" in jsonl_line:
        ann_metadata.update(jsonl_line["metadata"])

    return {
        "annotation_span_categories": ann_metadata["annotation_span_categories"],
        "annotator_id": ann_metadata.get("annotator_id"),
        "annotator_group": ann_metadata.get("annotator_group"),
        "annotation_granularity": ann_metadata.get("annotation_granularity"),
        "annotation_overlap_allowed": ann_metadata.get("annotation_overlap_allowed"),
        "campaign_id": slugify(ann_metadata["campaign_id"]),
        "dataset": slugify(jsonl_line["dataset"]),
        "example_idx": int(jsonl_line["example_idx"]),
        "setup_id": slugify(jsonl_line["setup_id"]),
        "split": slugify(jsonl_line["split"]),
        "flags": jsonl_line.get("flags", []),
        "options": jsonl_line.get("options", []),
        "sliders": jsonl_line.get("sliders", []),
        "text_fields": jsonl_line.get("text_fields", []),
        "jsonl_file": jsonl_file,
    }


def load_annotations_from_record(line, jsonl_file, metadata, split_spans=False):
    jsonl_line = json.loads(line)
    annotation_records = []

    record = create_annotation_example_record(jsonl_line, jsonl_file, metadata)

    if split_spans:
        for annotation in jsonl_line["annotations"]:
            record["annotation_type"] = int(annotation["type"])
            record["annotation_start"] = annotation["start"]
            record["annotation_text"] = annotation["text"]

            annotation_records.append(record.copy())
    else:
        record["annotations"] = jsonl_line["annotations"]
        annotation_records.append(record)

    return annotation_records


def get_annotation_files():
    """Get dictionary of annotation JSONL files and their modification times"""
    files_dict = {}
    for jsonl_file in Path(CAMPAIGN_DIR).rglob("*.jsonl"):
        campaign_dir = jsonl_file.parent.parent

        # find metadata for the campaign
        metadata_path = campaign_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        with open(metadata_path) as f:
            metadata = json.load(f)

        if metadata["mode"] == CampaignMode.HIDDEN or metadata["mode"] == CampaignMode.LLM_GEN:
            continue

        files_dict[str(jsonl_file)] = {
            "mtime": jsonl_file.stat().st_mtime,
            "metadata": metadata,
        }

    return files_dict


def remove_annotations(app, file_path):
    """Remove annotations from the annotation index for a specific file"""
    if app.db["annotation_index"] is not None:
        # Filter out annotations from the specified file
        app.db["annotation_index"] = app.db["annotation_index"][app.db["annotation_index"]["jsonl_file"] != file_path]


def get_annotation_index(app, force_reload=True):
    if app and app.db["annotation_index"] is not None and not force_reload:
        return app.db["annotation_index"]

    logger.debug("Reloading annotation index")

    # Get current files and their modification times
    current_files = get_annotation_files()
    cached_files = app.db.get("annotation_index_cache", {})
    new_annotations = []

    # Handle modified files
    for file_path, file_info in current_files.items():
        mod_time = file_info["mtime"]
        metadata = file_info["metadata"]

        if file_path not in cached_files or cached_files[file_path]["mtime"] < mod_time:
            remove_annotations(app, file_path)
            new_annotations.extend(load_annotations_from_file(file_path, metadata))

    # Handle deleted files
    for file_path in set(cached_files.keys()) - set(current_files.keys()):
        remove_annotations(app, file_path)

    # Update the cache
    app.db["annotation_index_cache"] = current_files

    if app.db["annotation_index"] is None:
        app.db["annotation_index"] = pd.DataFrame.from_records(new_annotations)
    else:
        app.db["annotation_index"] = pd.concat([app.db["annotation_index"], pd.DataFrame.from_records(new_annotations)])

    return app.db["annotation_index"]


def get_annotations(app, dataset_id, split, example_idx, setup_id):
    annotation_index = get_annotation_index(app, force_reload=False)

    if annotation_index.empty:
        return []

    annotations = annotation_index[
        (annotation_index["dataset"] == dataset_id)
        & (annotation_index["split"] == split)
        & (annotation_index["example_idx"] == example_idx)
        & (annotation_index["setup_id"] == setup_id)
    ]

    if annotations.empty:
        return []

    return annotations.to_dict(orient="records")


def get_output_files():
    """Get dictionary of annotation JSONL files and their modification times"""
    files_dict = {}
    for jsonl_file in Path(OUTPUT_DIR).rglob("*.jsonl"):
        files_dict[str(jsonl_file)] = jsonl_file.stat().st_mtime

    return files_dict


def load_outputs_from_file(file_path, cols):
    outputs = []

    with open(file_path) as f:
        for line_num, line in enumerate(f):
            try:
                j = json.loads(line)

                for key in ["dataset", "split", "setup_id"]:
                    j[key] = slugify(j[key])

                if "output" not in j:
                    logger.warning(
                        f"The output record in {file_path} at line {line_num + 1} is missing the 'output' key, skipping. Available keys: {list(j.keys())}"
                    )
                    continue

                # drop any keys that are not in the key set
                j = {k: v for k, v in j.items() if k in cols}
                j["jsonl_file"] = file_path
                outputs.append(j)
            except Exception as e:
                logger.error(
                    f"Error parsing output file {file_path} at line {line_num + 1}:\n\t{e.__class__.__name__}: {e}"
                )

    return outputs


def remove_outputs(app, file_path):
    """Remove outputs from the output index for a specific file"""
    if app.db["output_index"] is not None:
        # Filter out outputs from the specified file
        app.db["output_index"] = app.db["output_index"][app.db["output_index"].get("jsonl_file") != file_path]


def get_output_index(app, force_reload=True):
    if hasattr(app, "db") and app.db["output_index"] is not None and not force_reload:
        return app.db["output_index"]

    logger.debug("Reloading output index")

    cols = ["dataset", "split", "setup_id", "example_idx", "output"]

    current_outs = get_output_files()
    cached_outs = app.db.get("output_index_cache", {})
    new_outputs = []

    # Handle modified files
    for file_path, mod_time in current_outs.items():
        if file_path not in cached_outs or cached_outs[file_path] < mod_time:
            remove_outputs(app, file_path)
            new_outputs.extend(load_outputs_from_file(file_path, cols))

    # Handle deleted files
    for file_path in set(cached_outs.keys()) - set(current_outs.keys()):
        # Remove outputs for deleted files from the index
        if app.db["output_index"] is not None:
            remove_outputs(app, file_path)

    # Update the cache
    app.db["output_index_cache"] = current_outs

    if new_outputs:
        app.db["output_index"] = pd.concat([app.db["output_index"], pd.DataFrame.from_records(new_outputs)])
    elif app.db["output_index"] is None:
        app.db["output_index"] = pd.DataFrame(columns=cols)

    # Hotfix to prevent duplicate outputs after some updates
    # Probably a caching issue, should be fixed more properly
    app.db["output_index"] = (
        app.db["output_index"]
        .drop_duplicates(subset=["dataset", "split", "setup_id", "example_idx"], keep="last")
        .reset_index(drop=True)
    )

    return app.db["output_index"]


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
        }

    return overview


def get_dataset_class(submodule, class_name):
    # Dynamically import the class
    module = importlib.import_module("factgenie.datasets")
    submodule = getattr(module, submodule)
    dataset_class = getattr(submodule, class_name)

    return dataset_class


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
    shutil.rmtree(INPUT_DIR / dataset_id, ignore_errors=True)

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


def set_dataset_enabled(app, dataset_id, enabled):
    config = utils.load_dataset_config()
    config[dataset_id]["enabled"] = enabled

    if enabled:
        dataset = instantiate_dataset(dataset_id, config[dataset_id])
        app.db["datasets_obj"][dataset_id] = dataset
    else:
        app.db["datasets_obj"].pop(dataset_id, None)

    utils.save_dataset_config(config)


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
    if dataset_id in config:
        if config[dataset_id]["class"] != params[dataset_format]["class"]:
            raise ValueError(f"Dataset {dataset_id} already exists with a different class")

        elif any([split in config[dataset_id]["splits"] for split in dataset_data.keys()]):
            raise ValueError(f"Dataset {dataset_id} already exists with the same split")
        else:
            # if the user uploads a new split, add it to the existing dataset
            config[dataset_id]["splits"] = list(set(config[dataset_id]["splits"] + list(dataset_data.keys())))

            # update description:
            config[dataset_id]["description"] = dataset_description
    else:
        config[dataset_id] = {
            "name": dataset_name,
            "class": params[dataset_format]["class"],
            "description": dataset_description,
            "splits": list(dataset_data.keys()),
            "enabled": True,
        }
    utils.save_dataset_config(config)

    app.db["datasets_obj"][dataset_id] = instantiate_dataset(dataset_id, config[dataset_id])


def delete_model_outputs(dataset, split=None, setup_id=None):
    path = Path(OUTPUT_DIR)

    # look through all JSON files in the output directory
    for file in path.rglob("*.jsonl"):
        new_lines = []

        with open(file) as f:
            for line in f:
                j = json.loads(line)

                # None means all
                if (
                    (j["dataset"] == dataset)
                    and (split is None or j["split"] == split)
                    and (setup_id is None or j["setup_id"] == setup_id)
                ):
                    # delete the line
                    continue

                new_lines.append(line)

        if len(new_lines) == 0:
            os.remove(file)
        else:
            with open(file, "w") as f:
                f.writelines(new_lines)

    # remove any empty directories in the output directory
    for directory in path.rglob("*"):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()


def export_outputs(app, dataset_id, split, setup_id):
    zip_buffer = BytesIO()

    # assemble relevant outputs
    output_index = get_output_index(app)

    if output_index.empty:
        raise ValueError("No outputs found")

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


def get_available_data(app, datasets):
    data = []

    for dataset_id in datasets:
        splits = datasets[dataset_id]["splits"]

        for split in splits:
            data.append(
                {
                    "dataset": dataset_id,
                    "split": split,
                    "output_ids": list(range(datasets[dataset_id]["example_count"][split])),
                }
            )

    return data


def get_model_outputs_overview(app, datasets):
    output_index = get_output_index(app)

    if output_index.empty:
        return []

    # filter the df by datasets
    outputs = output_index.copy()

    if datasets:
        outputs = outputs[outputs["dataset"].isin(datasets)]

    # aggregate output ids into a list
    outputs = (
        outputs.groupby(["dataset", "split", "setup_id"])
        .agg(example_idx=pd.NamedAgg(column="example_idx", aggfunc=list))
        .reset_index()
    )
    # rename "example_idx" to "output_ids"
    outputs = outputs.rename(columns={"example_idx": "output_ids"})
    outputs = outputs.to_dict(orient="records")

    return outputs


def get_output_for_setup(dataset, split, example_idx, setup_id, app=None, force_reload=True):
    output_index = get_output_index(app=app, force_reload=force_reload)

    if output_index.empty:
        return None

    output = output_index[
        (output_index["dataset"] == dataset)
        & (output_index["split"] == split)
        & (output_index["setup_id"] == setup_id)
        & (output_index["example_idx"] == example_idx)
    ]

    if output.empty:
        return None

    return output.to_dict(orient="records")[0]


def get_outputs(dataset_id, split, example_idx, app=None, force_reload=True):
    outputs = get_output_index(app, force_reload=force_reload)

    if outputs.empty:
        return []

    outputs = outputs[
        (outputs["dataset"] == dataset_id) & (outputs["split"] == split) & (outputs["example_idx"] == example_idx)
    ]

    outputs = outputs.to_dict(orient="records")

    return outputs


def get_output_ids(app, dataset, split, setup_id):
    output_index = get_output_index(app)

    if output_index.empty:
        return []

    output_ids = output_index[
        (output_index["dataset"] == dataset) & (output_index["split"] == split) & (output_index["setup_id"] == setup_id)
    ]["example_idx"].tolist()

    return output_ids


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
                "output": out,
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


def get_campaign_data(campaign):
    campaign_data = campaign.db.to_dict(orient="records")

    # external campaigns do not have a db, we need to compute the equivalent from the JSONL files
    if not campaign_data:
        finished_examples = campaign.get_finished_examples()
        campaign_data = []

        for example in finished_examples:
            campaign_data.append(
                {
                    "dataset": example["dataset"],
                    "split": example["split"],
                    "setup_id": example["setup_id"],
                    "example_idx": example["example_idx"],
                    "annotator_id": example["metadata"]["annotator_id"],
                    "annotator_group": example["metadata"]["annotator_group"],
                    "status": ExampleStatus.FINISHED,
                }
            )

    return campaign_data


def get_sorted_campaign_list(app, modes):
    campaign_index = generate_campaign_index(app, force_reload=True)

    campaigns = [c for c in campaign_index.values() if c.metadata["mode"] in modes]

    campaigns.sort(key=lambda x: x.metadata["created"], reverse=True)

    campaigns = {
        c.metadata["id"]: {"metadata": c.metadata, "stats": c.get_stats(), "data": get_campaign_data(c)}
        for c in campaigns
    }
    return campaigns


def generate_default_id(app, mode, prefix):
    campaign_list = get_sorted_campaign_list(app, modes=[mode])

    i = 1
    default_campaign_id = f"{prefix}-{i}"
    while default_campaign_id in campaign_list.keys():
        default_campaign_id = f"{prefix}-{i}"
        i += 1

    return default_campaign_id


def refresh_indexes(app):
    # force reload the annotation and output index
    get_annotation_index(app, force_reload=True)
    get_output_index(app=app, force_reload=True)


def save_record(mode, campaign, row, result):
    campaign_id = campaign.metadata["id"]

    save_dir = os.path.join(CAMPAIGN_DIR, campaign_id, "files")
    os.makedirs(save_dir, exist_ok=True)

    dataset_id = str(row["dataset"])
    split = str(row["split"])
    example_idx = int(row["example_idx"])
    annotator_id = str(row["annotator_id"])

    # save the output
    record = {
        "dataset": dataset_id,
        "split": split,
        "example_idx": example_idx,
        "output": result["output"],
        "metadata": campaign.metadata["config"].copy(),
    }

    if result.get("thinking_trace"):
        record["thinking_trace"] = result["thinking_trace"]

    record["metadata"].pop("annotator_instructions", None)
    record["metadata"].pop("final_message", None)

    if mode == CampaignMode.LLM_EVAL or mode == CampaignMode.CROWDSOURCING:
        setup_id = str(row["setup_id"])
        record["setup_id"] = setup_id
        record["annotations"] = result["annotations"]
        record["flags"] = result.get("flags", [])
        record["options"] = result.get("options", [])
        record["sliders"] = result.get("sliders", [])
        record["text_fields"] = result.get("text_fields", [])

    if mode == CampaignMode.LLM_EVAL:
        record["metadata"]["prompt"] = result["prompt"]
        last_run = campaign.metadata.get("last_run", int(time.time()))
        filename = f"{dataset_id}-{split}-{setup_id}-{last_run}.jsonl"
    elif mode == CampaignMode.CROWDSOURCING:
        if result.get("time_last_saved", None):
            record["time_last_saved"] = result.get("time_last_saved", None)
        if result.get("time_last_accessed", None):
            record["time_last_accessed"] = result.get("time_last_accessed", None)

        batch_idx = row["batch_idx"]
        annotator_group = row.get("annotator_group", 0)
        batch_end = int(row["end"])
        filename = f"{batch_idx}-{annotator_group}-{annotator_id}-{batch_end}.jsonl"
    elif mode == CampaignMode.LLM_GEN:
        last_run = campaign.metadata.get("last_run", int(time.time()))
        filename = f"{dataset_id}-{split}-{last_run}.jsonl"
        record["metadata"]["prompt"] = result["prompt"]

    record["metadata"]["annotator_id"] = str(annotator_id)
    record["metadata"]["annotator_group"] = int(row["annotator_group"])
    record["metadata"]["campaign_id"] = str(campaign_id)

    record["metadata"]["start_timestamp"] = row.get("start", int(time.time()))
    record["metadata"]["end_timestamp"] = row.get("end", int(time.time()))

    # append the record to the file from the current run
    with open(os.path.join(save_dir, filename), "a") as f:
        f.write(json.dumps(record, allow_nan=True) + "\n")

    return record
