#!/usr/bin/env python3

import os
import pandas as pd
from collections import defaultdict
import sys
import logging
import traceback
import zipfile
import factgenie.workflows as workflows

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)


def generate_example_index(app, campaign):
    logger.info(f"Preparing example index for campaign {campaign.campaign_id}")

    annotation_span_categories = campaign.metadata["config"]["annotation_span_categories"]
    example_index = workflows.get_annotation_index(app, force_reload=True).copy()

    # get the examples for a specific campaign
    example_index = example_index[example_index["campaign_id"] == campaign.campaign_id]

    # Add category count columns to example index
    for i in range(len(annotation_span_categories)):
        col_name = f"cat_{i}"
        example_index[col_name] = example_index["annotations"].apply(
            lambda anns: sum(1 for a in anns if a["type"] == i)
        )

    return example_index


def generate_span_index(app, campaign):
    logger.info(f"Preparing span index for campaign {campaign.campaign_id}")

    span_index = workflows.get_annotation_index(app).copy()

    # get the examples for a specific campaign
    span_index = span_index[span_index["campaign_id"] == campaign.campaign_id]

    # Remove examples with no annotations
    span_index = span_index[span_index["annotations"].apply(lambda x: len(x) > 0)]

    if not span_index.empty:
        # Create a separate row for each annotation
        span_index = span_index.explode("annotations").reset_index(drop=True)

        # Extract annotation fields into separate columns
        span_index["annotation_type"] = span_index["annotations"].apply(lambda x: x["type"])
        span_index["annotation_start"] = span_index["annotations"].apply(lambda x: x["start"])
        span_index["annotation_text"] = span_index["annotations"].apply(lambda x: x["text"])

        # Drop the original annotations column
        span_index = span_index.drop("annotations", axis=1)

    return span_index


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
    logger.info("Computing annotation counts")

    # Create multi-index groupby once
    grouped = df.groupby(["dataset", "split", "setup_id", "annotation_type"]).size().reset_index(name="ann_count")

    # Create complete multi-index for all combinations
    idx = pd.MultiIndex.from_product(
        [df["dataset"].unique(), df["split"].unique(), df["setup_id"].unique(), sorted(df["annotation_type"].unique())],
        names=["dataset", "split", "setup_id", "annotation_type"],
    )

    # Reindex to include all combinations with zeros
    results = (
        grouped.set_index(["dataset", "split", "setup_id", "annotation_type"]).reindex(idx, fill_value=0).reset_index()
    )

    return results


def compute_avg_ann_counts(ann_counts, example_index):
    logger.info("Computing average annotation counts")

    # Get example counts through groupby operation
    example_counts = (
        example_index.groupby(["dataset", "split", "setup_id"])
        .agg(example_count=("example_idx", "nunique"))
        .reset_index()
        .astype({"example_count": int})
    )

    # Merge counts with original dataframe
    ann_counts = ann_counts.merge(example_counts, on=["dataset", "split", "setup_id"], how="left")

    # Compute average counts vectorized
    ann_counts["avg_count"] = (ann_counts["ann_count"] / ann_counts["example_count"]).round(3)

    return ann_counts


def compute_prevalence(ann_counts, example_index):
    logger.info("Computing annotation prevalence")

    # Compute affected counts for all rows at once
    ann_counts["prevalence"] = ann_counts.apply(
        lambda row: (
            (
                (example_index["dataset"] == row["dataset"])
                & (example_index["split"] == row["split"])
                & (example_index["setup_id"] == row["setup_id"])
                & (example_index[f"cat_{row['annotation_type']}"] > 0)
            ).sum()
            / row["example_count"]
            if row["example_count"] > 0
            else 0
        ),
        axis=1,
    ).round(3)

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


def compute_extra_fields_stats(example_index):
    # compute aggregate statistics for flags, options and text_fields (aggregates of value counts for each label)
    extra_fields_stats = {}

    try:
        for field in ["flags", "options", "text_fields"]:
            # each of `example_index[field]` is a list of dicts
            # each dict contains `label` and `value` keys
            # we want to count the number of occurrences of each `value` for each unique `label`
            # and then assign the dictionary with these counts to extra_fields_stats[label]

            # find unique labels
            labels = set()
            for example in example_index[field]:
                for d in example:
                    labels.add(d["label"])

            # create a dictionary for each label
            for label in labels:
                extra_fields_stats[label] = defaultdict(int)

            # count the occurrences of each value for each label
            for example in example_index[field]:
                for d in example:
                    extra_fields_stats[d["label"]][d["value"]] += 1
    except Exception as e:
        logger.error(f"Error while computing extra fields statistics: {e}")
        traceback.print_exc()

    return extra_fields_stats


def compute_statistics(app, campaign):
    statistics = {}

    span_index = generate_span_index(app, campaign)
    example_index = generate_example_index(app, campaign)

    if not span_index.empty:
        span_index = preprocess_annotations(span_index, campaign)

        annotation_counts = compute_ann_counts(span_index)
        annotation_counts = compute_avg_ann_counts(annotation_counts, example_index)
        annotation_counts = compute_prevalence(annotation_counts, example_index)

        # replace NaNs with 0
        annotation_counts = annotation_counts.fillna(0.0)

        statistics["ann_counts"] = {
            "full": annotation_counts.to_dict(orient="records"),
            "span": aggregate_ann_counts(annotation_counts, "span"),
            "setup": aggregate_ann_counts(annotation_counts, "setup"),
            "dataset": aggregate_ann_counts(annotation_counts, "dataset"),
        }

    if not example_index.empty:
        extra_fields_stats = compute_extra_fields_stats(example_index)
        statistics["extra_fields"] = extra_fields_stats

    return statistics


def compute_span_counts(example_index, annotator_count, combinations, cat_columns):
    # create a list for span counts from each annotator (do this separately for each error category)
    dataset_level_counts = []
    example_level_counts = []

    annotator_group_ids = example_index.iloc[0].annotator_group_id

    for dataset, split, setup_id in combinations:
        example_index_subset = example_index[
            (example_index["dataset"] == dataset)
            & (example_index["split"] == split)
            & (example_index["setup_id"] == setup_id)
        ]

        error_counts = [{"cat_" + str(i): [] for i in range(len(cat_columns))} for _ in range(annotator_count)]

        for i, row in example_index_subset.iterrows():
            for a in range(annotator_count):
                for j, c in enumerate(cat_columns):
                    error_counts[a]["cat_" + str(j)].append(row[c][a])

                    example_level_counts.append(
                        {
                            "dataset": dataset,
                            "split": split,
                            "setup_id": setup_id,
                            "example_idx": row["example_idx"],
                            "annotator_group_id": annotator_group_ids[a],
                            "annotation_type": c.split("_")[1],
                            "count": row[c][a],
                        }
                    )

    example_level_counts = pd.DataFrame(example_level_counts)

    # average counts for each (dataset, split, setup_id)
    dataset_level_counts = (
        example_level_counts.groupby(["dataset", "split", "setup_id", "annotation_type", "annotator_group_id"])
        .agg({"count": "mean"})
        .reset_index()
    )

    return dataset_level_counts, example_level_counts


def prepare_example_index(app, combinations, selected_campaigns, campaigns):
    # gather a list of all examples with some annotations
    example_index = pd.DataFrame()

    for campaign_id in selected_campaigns:
        campaign = campaigns[campaign_id]

        ei = generate_example_index(app, campaign)
        example_index = pd.concat([example_index, ei], ignore_index=True)

    # a combination is a tuple (dataset, split, setup_id)
    # leave only examples in example_index that are in the combinations selected by the user
    example_index = example_index[
        example_index.apply(lambda x: (x["dataset"], x["split"], x["setup_id"]) in combinations, axis=1)
    ]

    # add a column "annotator_group_id" to example_index, concatenating the campaign_id with str(annotator_group)
    example_index["annotator_group_id"] = (
        example_index["campaign_id"] + "-anngroup-" + example_index["annotator_group"].astype(str)
    )

    # get the number of annotators we are considering
    annotator_group_ids = list(example_index["annotator_group_id"].unique())
    annotator_count = len(annotator_group_ids)

    # group examples by dataset, split, setup_id, example_idx
    # aggregate annotations, annotator_ids, and counts for each category into a list
    aggregations = {"annotations": list, "annotator_group_id": list}
    cat_columns = [x for x in example_index.columns if x.startswith("cat_")]

    for c in cat_columns:
        aggregations[c] = list

    example_index = (
        example_index.groupby(["dataset", "split", "setup_id", "example_idx"]).agg(aggregations).reset_index()
    )
    # remove all examples that do not have annotations from all annotators
    example_index = example_index[example_index["annotator_group_id"].apply(lambda x: len(x) == annotator_count)]

    return example_index, annotator_count, annotator_group_ids, cat_columns


def compute_gamma_spans(app, selected_campaigns, campaigns):
    span_index = []

    for campaign_id in selected_campaigns:
        df = generate_span_index(app, campaigns[campaign_id])

        df["annotation_end"] = df["annotation_start"] + df["annotation_text"].str.len()
        df["annotator_group_id"] = df["campaign_id"] + "-anngroup-" + df["annotator_group"].astype(str)
        span_index.append(df)

    span_index = pd.concat(span_index, ignore_index=True)

    span_index = span_index.drop(
        columns=[
            "annotation_span_categories",
            "annotator_id",
            "annotation_granularity",
            "annotation_overlap_allowed",
            "flags",
            "options",
            "text_fields",
            "jsonl_file",
            "annotation_text",
        ]
    )

    return span_index


def generate_iaa_files(app, selected_campaigns, combinations, campaigns, temp_dir):
    combinations = [(c["dataset"], c["split"], c["setup_id"]) for c in combinations]

    example_index, annotator_count, annotator_group_ids, cat_columns = prepare_example_index(
        app, combinations=combinations, selected_campaigns=selected_campaigns, campaigns=campaigns
    )

    dataset_level_counts, example_level_counts = compute_span_counts(
        example_index=example_index, annotator_count=annotator_count, combinations=combinations, cat_columns=cat_columns
    )

    gamma_spans = compute_gamma_spans(app, selected_campaigns, campaigns)

    results = {
        "dataset_level_counts": dataset_level_counts,
        "example_level_counts": example_level_counts,
        "gamma_spans": gamma_spans,
    }

    # Save each dataframe as CSV
    for name, df in results.items():
        csv_path = os.path.join(temp_dir, f"{name}.csv")

        # set precision of the `count` column to 3 decimal places
        if "count" in df.columns:
            df["count"] = df["count"].round(3)

        df.to_csv(csv_path, index=False)

    # Create ZIP file
    zip_path = os.path.join(temp_dir, "agreement_results.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for name in results.keys():
            csv_path = os.path.join(temp_dir, f"{name}.csv")
            zipf.write(csv_path, os.path.basename(csv_path))

    return zip_path
