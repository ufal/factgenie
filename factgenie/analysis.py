#!/usr/bin/env python3

import re
import glob
import json
import random
import os
import argparse
import pandas as pd
from collections import defaultdict
from scipy.stats import pearsonr
import sys
from pathlib import Path
import logging
import coloredlogs
import factgenie.utils as utils

from factgenie.campaigns import ANNOTATIONS_DIR

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO", logger=logger, fmt="%(asctime)s %(levelname)s %(message)s")


def load_annotation(line, campaign_id):
    j = json.loads(line)
    annotation_records = []

    r = {
        "annotator_id": j["annotator_id"],
        "campaign_id": campaign_id,
        "dataset": j["dataset"],
        "example_idx": j["example_idx"],
        "setup_id": j["setup"]["id"],
        "split": j["split"],
    }
    for annotation in j["annotations"]:
        r["annotation_type"] = annotation["type"]
        r["annotation_start"] = annotation["start"]
        r["annotation_text"] = annotation["text"]

        annotation_records.append(r.copy())

    return annotation_records


def load_annotations_for_campaigns(campaign_ids):
    annotation_index = []

    for campaign_id in campaign_ids:
        jsonl_files = glob.glob(os.path.join(ANNOTATIONS_DIR, campaign_id, "files", "*.jsonl"))

        for jsonl_file in jsonl_files:
            with open(jsonl_file) as f:
                lines = f.readlines()

            for line in lines:
                try:
                    annotation_records = load_annotation(line, campaign_id)
                    annotation_index += annotation_records
                except Exception as e:
                    logger.error(f"Error while processing line: {line}")
                    logger.error(e)

    df = pd.DataFrame(annotation_index)

    return df


def clean_annotations(df, campaign):
    # remove lines with nans
    df = df.dropna()

    # remove annotations with type that is not in the correct range (0 - len(annotation_span_categories))
    annotation_span_categories = campaign.metadata["config"]["annotation_span_categories"]
    category_cnt = len(annotation_span_categories)
    df = df[df["annotation_type"].apply(lambda x: x in range(category_cnt))]

    # make annotation_type an integer
    df["annotation_type"] = df["annotation_type"].astype(int)

    return df


def compute_err_counts(df):
    """
    Compute error counts for each annotation type (separately for each dataset, split, setup_id).
    """
    results = []

    for dataset in df["dataset"].unique():
        for split in df["split"].unique():
            for setup_id in df["setup_id"].unique():
                # filter the dataframe
                df_filtered = df[(df["dataset"] == dataset) & (df["split"] == split) & (df["setup_id"] == setup_id)]

                # compute the error counts
                errors_counts = df_filtered.groupby("annotation_type").size().reset_index(name="error_count")

                errors_counts["dataset"] = dataset
                errors_counts["split"] = split
                errors_counts["setup_id"] = setup_id

                results.append(errors_counts)

    # concatenate all results into a single dataframe
    results = pd.concat(results, ignore_index=True)

    return results


def compute_avg_error_counts(error_counts, datasets):
    # for each line in error_counts, find the corresponding dataset in datasets and add the number of examples
    # then compute the average error count

    # add a column with the number of examples for each dataset, split
    error_counts["example_count"] = 0

    for i, row in error_counts.iterrows():
        dataset = row["dataset"]
        split = row["split"]
        dataset_info = datasets[dataset]
        error_counts.loc[i, "example_count"] = int(dataset_info["example_count"][split])

    error_counts["avg_count"] = error_counts["error_count"] / error_counts["example_count"]

    return error_counts


def aggregate_error_counts(error_counts, groupby):
    if groupby == "annotation_type":
        aggregated = error_counts.groupby("annotation_type").agg({"avg_count": "mean", "error_count": "sum"}).to_dict()

    elif groupby == "setup_id":
        # keep individual error categories, but aggregate setup_ids for each dataset, split
        aggregated = (
            error_counts.groupby(["dataset", "split", "annotation_type"])
            .agg({"avg_count": "mean", "error_count": "sum"})
            .to_dict()
        )

    elif groupby == "dataset":
        # keep individual error categories, but aggregate datasets for each split, setup_id
        aggregated = (
            error_counts.groupby(["split", "setup_id", "annotation_type"])
            .agg({"avg_count": "mean", "error_count": "sum"})
            .to_dict()
        )

    return aggregated


def compute_statistics(app, campaign, datasets):
    campaign_id = campaign.metadata["id"]

    statistics = {}

    df = load_annotations_for_campaigns([campaign_id])
    df = clean_annotations(df, campaign)

    error_counts = compute_err_counts(df)
    error_counts = compute_avg_error_counts(error_counts, datasets)

    statistics["error_counts"] = {
        "full": error_counts.to_dict(orient="records"),
        "aggregated": {
            "annotation_type": aggregate_error_counts(error_counts, "annotation_type"),
            "setup_id": aggregate_error_counts(error_counts, "setup_id"),
        },
    }

    return statistics
