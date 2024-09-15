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


def load_annotations(line, campaign_id):
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


def create_example_record(line, campaign_id, annotation_span_categories):
    # a record is created even if there are no annotations
    j = json.loads(line)

    example_record = {
        "annotator_id": j["annotator_id"],
        "campaign_id": campaign_id,
        "dataset": j["dataset"],
        "example_idx": j["example_idx"],
        "setup_id": j["setup"]["id"],
        "split": j["split"],
    }

    for i, category in enumerate(annotation_span_categories):
        example_record["cat_" + str(i)] = 0

        for annotation in j["annotations"]:
            if int(annotation["type"]) == i:
                example_record["cat_" + str(i)] += 1

    return example_record


def load_annotations_for_campaign(campaign):
    annotation_index = []
    example_index = []

    campaign_id = campaign.metadata["id"]
    annotation_span_categories = campaign.metadata["config"]["annotation_span_categories"]

    jsonl_files = glob.glob(os.path.join(ANNOTATIONS_DIR, campaign_id, "files", "*.jsonl"))

    for jsonl_file in jsonl_files:
        with open(jsonl_file) as f:
            lines = f.readlines()
        for line in lines:
            try:
                annotation_records = load_annotations(line, campaign_id)
                annotation_index += annotation_records

                example_record = create_example_record(line, campaign_id, annotation_span_categories)
                example_index.append(example_record)
            except Exception as e:
                logger.error(f"Error while processing line: {line}")
                logger.error(e)

    annotation_index = pd.DataFrame(annotation_index)
    example_index = pd.DataFrame(example_index)

    return annotation_index, example_index


def preprocess_annotations(df, campaign):
    # remove lines with nans
    df = df.dropna()

    # remove annotations with type that is not in the correct range (0 - len(annotation_span_categories))
    annotation_span_categories = campaign.metadata["config"]["annotation_span_categories"]
    category_cnt = len(annotation_span_categories)
    df = df[df["annotation_type"].apply(lambda x: x in range(category_cnt))]

    # make annotation_type an integer
    df["annotation_type"] = df["annotation_type"].astype(int)

    return df


def compute_ann_counts(df):
    """
    Compute annotation counts for each annotation type (separately for each dataset, split, setup_id).
    """
    results = []

    all_annotation_types = df["annotation_type"].unique()
    all_annotation_types.sort()

    for dataset in df["dataset"].unique():
        for split in df["split"].unique():
            for setup_id in df["setup_id"].unique():
                # filter the dataframe
                df_filtered = df[(df["dataset"] == dataset) & (df["split"] == split) & (df["setup_id"] == setup_id)]

                # make sure that all annotation types are present in the dataframe, even with zero counts
                ann_counts = (
                    df_filtered.groupby("annotation_type")
                    .size()
                    .reindex(all_annotation_types, fill_value=0)
                    .reset_index(name="ann_count")
                )

                ann_counts["dataset"] = dataset
                ann_counts["split"] = split
                ann_counts["setup_id"] = setup_id

                results.append(ann_counts)

    # concatenate all results into a single dataframe
    results = pd.concat(results, ignore_index=True)

    return results


def compute_avg_ann_counts(ann_counts, example_index):
    # for each line in ann_counts, find the corresponding dataset in datasets and add the number of examples
    # then compute the average annotation count

    # add a column with the number of examples for each dataset, split
    ann_counts["example_count"] = 0

    for i, row in ann_counts.iterrows():
        dataset = row["dataset"]
        split = row["split"]
        setup_id = row["setup_id"]
        ann_counts.loc[i, "example_count"] = (
            example_index[
                (example_index["dataset"] == dataset)
                & (example_index["split"] == split)
                & (example_index["setup_id"] == setup_id)
            ]
            .example_idx.unique()
            .shape[0]
        )

    ann_counts["avg_count"] = ann_counts["ann_count"] / ann_counts["example_count"]

    # round to three decimal places
    ann_counts["avg_count"] = ann_counts["avg_count"].round(3)

    return ann_counts


def compute_prevalence(ann_counts, example_index):
    # for each combination of dataset, split, setup_id, annotation_type, compute the percentage of examples that are affected by the annotation type and add it to the `ann_counts` dataframe
    for i, row in ann_counts.iterrows():
        dataset = row["dataset"]
        split = row["split"]
        setup_id = row["setup_id"]
        annotation_type = row["annotation_type"]

        examples = example_index[
            (example_index["dataset"] == dataset)
            & (example_index["split"] == split)
            & (example_index["setup_id"] == setup_id)
            & (example_index["cat_" + str(annotation_type)] > 0)
        ]

        ann_counts.loc[i, "prevalence"] = examples.shape[0] / row["example_count"]

        # round to three decimal places
        ann_counts["prevalence"] = ann_counts["prevalence"].round(3)

    return ann_counts


def aggregate_ann_counts(ann_counts, groupby):
    if groupby == "span":
        aggregated = (
            ann_counts.groupby("annotation_type")
            .agg({"avg_count": "mean", "ann_count": "sum", "example_count": "sum", "prevalence": "mean"})
            .reset_index()
            .to_dict(orient="records")
        )

    elif groupby == "setup":
        # keep individual annotation categories, but aggregate setup_ids for each dataset, split
        aggregated = (
            ann_counts.groupby(["setup_id", "annotation_type"])
            .agg({"avg_count": "mean", "ann_count": "sum", "example_count": "sum", "prevalence": "mean"})
            .reset_index()
            .to_dict(orient="records")
        )

    elif groupby == "dataset":
        # keep individual annotation categories, but aggregate datasets for each split, setup_id
        aggregated = (
            ann_counts.groupby(["dataset", "split", "annotation_type"])
            .agg({"avg_count": "mean", "ann_count": "sum", "example_count": "sum", "prevalence": "mean"})
            .reset_index()
            .to_dict(orient="records")
        )

    # round to three decimal places
    for a in aggregated:
        a["avg_count"] = round(a["avg_count"], 3)
        a["prevalence"] = round(a["prevalence"], 3)

    return aggregated


def compute_statistics(app, campaign, datasets):
    statistics = {}

    annotation_index, example_index = load_annotations_for_campaign(campaign)
    annotation_index = preprocess_annotations(annotation_index, campaign)

    annotation_counts = compute_ann_counts(annotation_index)
    annotation_counts = compute_avg_ann_counts(annotation_counts, example_index)

    annotation_counts = compute_prevalence(annotation_counts, example_index)

    statistics["ann_counts"] = {
        "full": annotation_counts.to_dict(orient="records"),
        "span": aggregate_ann_counts(annotation_counts, "span"),
        "setup": aggregate_ann_counts(annotation_counts, "setup"),
        "dataset": aggregate_ann_counts(annotation_counts, "dataset"),
    }

    return statistics
