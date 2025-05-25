#!/usr/bin/env python3

import logging
import os
import sys
import traceback
from collections import defaultdict

import pandas as pd

import factgenie.workflows as workflows

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger("factgenie")


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
        span_index["annotation_reason"] = span_index["annotations"].apply(lambda x: x.get("reason", ""))

        # Drop the original annotations column
        span_index = span_index.drop("annotations", axis=1)

        # Remove any annotations that have NaN start or type or empty text
        span_index = span_index.dropna(
            subset=["annotation_start", "annotation_type", "annotation_text", "annotation_reason"]
        )
        span_index = span_index[span_index["annotation_text"].apply(lambda x: len(x) > 0)]

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
        for field in ["flags", "options", "sliders", "text_fields"]:
            # each of `example_index[field]` is a list of dicts
            # each dict contains `label` and `value` keys
            # we want to count the number of occurrences of each `value` for each unique `label`
            # and then assign the dictionary with these counts to extra_fields_stats[label]

            if field not in example_index.columns:
                continue

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


def compute_span_index(app, selected_campaigns, campaigns, combinations=None):
    span_index = []

    # deduplicate
    selected_campaigns = list(set(selected_campaigns))

    for campaign_id in selected_campaigns:
        df = generate_span_index(app, campaigns[campaign_id])

        # filter out examples that are not in the selected combinations (dataset, split, setup_id)
        if combinations:
            df = df[df.apply(lambda x: (x["dataset"], x["split"], x["setup_id"]) in combinations, axis=1)]

        # Make sure that annotation_start is an integer
        df["annotation_start"] = df["annotation_start"].astype(int)
        df["annotation_end"] = df["annotation_start"] + df["annotation_text"].str.len()
        df["annotator_group_id"] = df.apply(
            lambda x: format_group_id(x["campaign_id"], str(x["annotator_group"])), axis=1
        )
        span_index.append(df)

    span_index = pd.concat(span_index, ignore_index=True)

    columns_to_drop = [
        "annotation_span_categories",
        # "annotator_id",
        "annotation_granularity",
        "annotation_overlap_allowed",
        "flags",
        "options",
        "sliders",
        "text_fields",
        "jsonl_file",
        # "annotation_text",
    ]

    # Only drop columns that exist in the DataFrame
    existing_columns = [col for col in columns_to_drop if col in span_index.columns]
    span_index = span_index.drop(columns=existing_columns)

    span_index["annotator_group"] = span_index["annotator_group"].astype(int)

    return span_index


def assert_common_categories(campaign_ids, campaign_index):
    """Verify that all campaigns share the same annotation categories."""
    common_category_names = None

    for campaign_id in campaign_ids:
        campaign = campaign_index[campaign_id]
        categories = campaign.metadata["config"]["annotation_span_categories"]
        category_names = [category["name"] for category in categories]

        if common_category_names is None:
            common_category_names = category_names
        elif common_category_names != category_names:
            logger.error(
                f"Annotation span categories do not match across campaigns: {common_category_names} vs {category_names}"
            )
            return False

    category_list = ", ".join([f"{i}: {category_name}" for i, category_name in enumerate(common_category_names)])
    logger.info(f"Categories: {category_list}")
    return True


def format_group_id(campaign_id, group):
    """Format annotator group ID."""
    if group is None:
        return f"{campaign_id}-anngroup-all"

    return f"{campaign_id}-anngroup-{group}"


def get_common_examples(first_campaign_data, second_campaign_data, first_group=None, second_group=None):
    """Find common examples between two annotator groups.

    If a group is None, all examples from that campaign are considered regardless of group.
    """
    # Filter examples for first group (or all if first_group is None)
    if first_group is None:
        first_examples = first_campaign_data[first_campaign_data["status"] == "finished"][
            ["dataset", "split", "setup_id"]
        ].drop_duplicates()
    else:
        first_examples = first_campaign_data[
            (first_campaign_data["annotator_group"] == first_group) & (first_campaign_data["status"] == "finished")
        ][["dataset", "split", "setup_id"]].drop_duplicates()

    # Filter examples for second group (or all if second_group is None)
    if second_group is None:
        second_examples = second_campaign_data[second_campaign_data["status"] == "finished"][
            ["dataset", "split", "setup_id"]
        ].drop_duplicates()
    else:
        second_examples = second_campaign_data[
            (second_campaign_data["annotator_group"] == second_group) & (second_campaign_data["status"] == "finished")
        ][["dataset", "split", "setup_id"]].drop_duplicates()

    # Find intersection using merge
    common = pd.merge(first_examples, second_examples, how="inner")

    # Convert to list of tuples
    return list(map(tuple, common.values))


def get_ref_hyp_spans(span_index, annotator_groups):
    """Get reference and hypothesis spans for a set of spans."""
    # Unpack annotator groups
    ref_camp_id, ref_group = annotator_groups[0]
    hyp_camp_id, hyp_group = annotator_groups[1]

    # Get reference and hypothesis spans for this example
    ref_spans = span_index[
        (span_index["campaign_id"] == ref_camp_id)
        & (span_index["annotator_group"] == ref_group if ref_group is not None else True)
    ]
    hyp_spans = span_index[
        (span_index["campaign_id"] == hyp_camp_id)
        & (span_index["annotator_group"] == hyp_group if hyp_group is not None else True)
    ]

    return ref_spans, hyp_spans


def get_example_list(
    campaign_index, annotator_groups, include_dataset=None, include_split=None, include_example_id=None
):
    """
    Get list of examples to consider for gamma score computation.

    Args:
        campaign_index: Dictionary mapping campaign_id to campaign object
        annotator_groups: List of tuples (campaign_id, ann_group)
        include_dataset: List of dataset IDs to include (None means include all)
        include_split: List of splits to include (None means include all)
        include_example_id: List of example IDs to include (None means include all)

    Returns:
        DataFrame with columns dataset, split, setup_id, example_idx
    """
    all_examples = []

    # Collect examples from each campaign and annotator group
    for campaign_id, ann_group in annotator_groups:
        campaign = campaign_index[campaign_id]
        campaign_examples = campaign.db.copy()

        # Filter by annotator group if specified
        if ann_group is not None:
            campaign_examples = campaign_examples[campaign_examples["annotator_group"] == ann_group]

        # Filter examples with finished status
        campaign_examples = campaign_examples[campaign_examples["status"] == "finished"]

        # Filter by datasets if specified
        if include_dataset:
            campaign_examples = campaign_examples[campaign_examples["dataset"].isin(include_dataset)]

        # Filter by splits if specified
        if include_split:
            campaign_examples = campaign_examples[campaign_examples["split"].isin(include_split)]

        # Filter by example_ids if specified
        if include_example_id:
            campaign_examples = campaign_examples[campaign_examples["example_idx"].isin(include_example_id)]

        # Check for multiple groups annotating the same examples
        if (
            ann_group is None
            and campaign_examples.duplicated(subset=["dataset", "split", "setup_id", "example_idx"], keep=False).any()
        ):
            logger.warning(
                f"Warning: Campaign {campaign_id} has multiple groups annotating the same example(s) and no annotator groups are specified. This may mix outputs from different annotators, potentially affecting metrics."
            )

        # Keep only relevant columns
        campaign_examples = campaign_examples[["annotator_group", "dataset", "split", "setup_id", "example_idx"]]
        campaign_examples["campaign_id"] = campaign_id

        # Add to the list of all examples
        all_examples.append(campaign_examples)

    # Combine all example sets
    if not all_examples:
        return pd.DataFrame(columns=["dataset", "split", "setup_id", "example_idx"])

    combined_examples = pd.concat(all_examples, ignore_index=True)

    # Filter examples: keep only those that are annotated by ALL given annotator groups.
    # An annotator group is defined as a (campaign_id, annotator_group) tuple.
    # If annotator_group is None, any example from that campaign is acceptable.
    def is_annotated_by_all_groups(group):
        for camp, req_group in annotator_groups:
            if req_group is None:
                if camp not in group["campaign_id"].values:
                    return False
            else:
                if not (((group["campaign_id"] == camp) & (group["annotator_group"] == req_group)).any()):
                    return False
        return True

    # Group by the identifying columns and filter accordingly
    groups = combined_examples.groupby(["dataset", "split", "setup_id", "example_idx"])
    valid_keys = [key for key, group in groups if is_annotated_by_all_groups(group)]

    # Create a DataFrame from the valid example keys
    filtered_examples = pd.DataFrame(valid_keys, columns=["dataset", "split", "setup_id", "example_idx"])

    # Report number of examples skipped
    total_unique = groups.ngroups
    skipped = total_unique - filtered_examples.shape[0]

    if skipped > 0:
        logger.info(f"Skipped {skipped} examples not annotated by all required groups.")

    return filtered_examples
