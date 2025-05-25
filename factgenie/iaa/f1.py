#!/usr/bin/env python3

import logging

import pandas as pd
from flask import current_app as app

from factgenie.analysis import get_ref_hyp_spans

logger = logging.getLogger(__name__)


def compute_span_counts(example_index, combinations):
    # create a list for span counts from each annotator (do this separately for each error category)
    dataset_level_counts = []
    example_level_counts = []

    for dataset, split, setup_id in combinations:
        example_index_subset = example_index[
            (example_index["dataset"] == dataset)
            & (example_index["split"] == split)
            & (example_index["setup_id"] == setup_id)
        ]

        cat_columns = [x for x in example_index_subset.columns if x.startswith("cat_")]

        for i, row in example_index_subset.iterrows():
            for cat in cat_columns:
                # if there are less than 2 annotators, skip this example and print warning
                if len(row["annotator_group_id"]) < 2:
                    logger.warning(
                        f"Skipping example {dataset}/{split}/{setup_id}/{row['example_idx']} as it has less than 2 annotators."
                    )
                    continue
                for cat_count, ann_group in zip(row[cat], row["annotator_group_id"]):
                    example_level_counts.append(
                        {
                            "dataset": dataset,
                            "split": split,
                            "setup_id": setup_id,
                            "example_idx": row["example_idx"],
                            "annotator_group_id": ann_group,
                            "annotation_type": cat.split("_")[1],
                            "count": cat_count,
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


def compute_f1_scores(span_index, annotator_groups, example_list, match_mode="hard", category_breakdown=False):
    """
    Compute precision, recall, and F1-score between reference and hypothesis spans.

    Args:
        span_index: DataFrame containing all spans from both annotator groups
        annotator_groups: List of tuples with reference and hypothesis annotator groups
        example_list: DataFrame with example IDs to compute metrics for
        match_mode: 'hard' requires exact category match, 'soft' allows any category
        category_breakdown: Return metrics broken down by annotation category

    Returns:
        Dictionary with overall precision, recall, F1 and optionally per-category metrics
    """
    if example_list.empty or span_index.empty:
        logger.warning("Input span_index to compute_f1_scores is empty.")
        metrics = _initialize_metrics(categories=None)

    # Initialize tracking metrics
    categories = sorted(span_index["annotation_type"].unique())
    metrics = _initialize_metrics(categories=categories)

    # Process each example
    for example_tuple in example_list.itertuples(index=False):
        metrics = _process_example(example_tuple, span_index, annotator_groups, metrics, match_mode)

    # Calculate final scores
    _calculate_results(metrics)

    # Add per-category metrics if requested
    if category_breakdown:
        _calculate_category_results(metrics, categories)

    return metrics


def _initialize_metrics(categories):
    """Initialize tracking metrics for overall and per-category calculations."""
    metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "ref_count": 0,
        "hyp_count": 0,
        "hyp_length": 0,
        "ref_length": 0,
        "overlap_length": 0,
    }

    if categories:
        metrics["categories"] = {cat: metrics.copy() for cat in categories}

    return metrics


def _process_example(example_tuple, span_index, annotator_groups, metrics, match_mode):
    """Process a single example and update metrics."""
    dataset, split, setup_id, example_idx = example_tuple

    # Filter spans for this example
    example_spans = span_index[
        (span_index["dataset"] == dataset)
        & (span_index["split"] == split)
        & (span_index["setup_id"] == setup_id)
        & (span_index["example_idx"] == example_idx)
    ]
    example_ref_spans, example_hyp_spans = get_ref_hyp_spans(example_spans, annotator_groups)

    metrics["ref_count"] += example_ref_spans.shape[0]
    metrics["hyp_count"] += example_hyp_spans.shape[0]

    if not example_ref_spans.empty and not example_hyp_spans.empty:
        metrics = _process_overlaps_for_example(example_hyp_spans, example_ref_spans, metrics, match_mode)

    return metrics


def _process_overlaps_for_example(example_hyp_spans, example_ref_spans, metrics, match_mode):
    """Find and process overlaps between hypothesis and reference spans."""
    # Track position pairs that have been matched to avoid double counting
    matched_pairs = set()  # Will store (hyp_pos, ref_id) pairs

    # First, calculate all reference span lengths for metrics
    for _, ref_row in example_ref_spans.iterrows():
        ref_start = ref_row["annotation_start"]
        ref_end = ref_row["annotation_end"]
        ref_type = ref_row["annotation_type"]
        ref_length = ref_end - ref_start

        metrics["ref_length"] += ref_length
        metrics["categories"][ref_type]["ref_length"] += ref_length

    # Process each hypothesis span
    for _, hyp_row in example_hyp_spans.iterrows():
        hyp_start = hyp_row["annotation_start"]
        hyp_end = hyp_row["annotation_end"]
        hyp_type = hyp_row["annotation_type"]
        hyp_length = hyp_end - hyp_start

        # Update hypothesis length metrics
        metrics["hyp_length"] += hyp_length
        metrics["categories"][hyp_type]["hyp_length"] += hyp_length

        # For each position in the hypothesis span, try to find a matching reference position
        for hyp_pos in range(hyp_start, hyp_end):
            # Check all reference spans for potential matches at this position
            for ref_id, ref_row in example_ref_spans.iterrows():
                ref_start = ref_row["annotation_start"]
                ref_end = ref_row["annotation_end"]
                ref_type = ref_row["annotation_type"]

                # Check if this reference span covers this position
                if ref_start <= hyp_pos < ref_end:
                    # For hard mode, categories must match
                    if match_mode == "hard" and hyp_type != ref_type:
                        continue

                    # If this reference position hasn't been matched yet, create a match
                    if (hyp_pos, ref_id) not in matched_pairs:
                        matched_pairs.add((hyp_pos, ref_id))

                        # Update overlap metrics
                        metrics["overlap_length"] += 1
                        metrics["categories"][hyp_type]["overlap_length"] += 1

                        # Once we've found a match for this hypothesis position, we can stop
                        break

    return metrics


def _calculate_results(metrics):
    """Calculate precision, recall, and F1 from accumulated metrics."""
    precision = metrics["overlap_length"] / metrics["hyp_length"] if metrics["hyp_length"] > 0 else 0
    recall = metrics["overlap_length"] / metrics["ref_length"] if metrics["ref_length"] > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results = {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }

    metrics.update(results)


def _calculate_category_results(metrics, categories):
    """Calculate per-category precision, recall, and F1."""
    for cat in categories:
        _calculate_results(metrics["categories"][cat])


def compute_f1(
    ref_camp_id,
    ref_group,
    hyp_camp_id,
    hyp_group,
    match_mode="hard",
    category_breakdown=False,
    include_dataset=None,
    include_split=None,
    include_example_id=None,
):
    """
    Compute precision, recall, F1 between reference and hypothesis annotator groups.
    Always aggregates all groups from each campaign.

    Args:
        ref_camp_id: Reference campaign ID
        ref_group: Reference annotator group (None means all groups)
        hyp_camp_id: Hypothesis campaign ID
        hyp_group: Hypothesis annotator group (None means all groups)
        match_mode: 'hard' requires exact category match, 'soft' allows any category
        category_breakdown: Whether to compute metrics per annotation category
        include_dataset: List of dataset IDs to include (None means include all)
        include_split: List of splits to include (None means include all)
        include_example_id: List of example IDs to include (None means include all)

    Returns:
        Dictionary with F1 metrics for aggregated reference-hypothesis pair
    """
    from factgenie.analysis import (
        assert_common_categories,
        compute_span_index,
        get_example_list,
    )
    from factgenie.workflows import generate_campaign_index

    # Generate campaign index using current_app
    campaign_index = generate_campaign_index(app, force_reload=True)
    campaign_ids = [ref_camp_id, hyp_camp_id]

    if not assert_common_categories(campaign_ids, campaign_index):
        return None

    annotator_groups = [
        (ref_camp_id, ref_group),
        (hyp_camp_id, hyp_group),
    ]

    span_index = compute_span_index(app, campaign_ids, campaign_index)
    example_list = get_example_list(
        campaign_index, annotator_groups, include_dataset, include_split, include_example_id
    )

    if example_list.empty:
        logger.error("No examples found that are annotated by both groups")
        return _initialize_metrics(categories=None)

    logger.info(f"Computing F1 on {len(example_list)} examples")

    f1_results = compute_f1_scores(
        span_index=span_index,
        annotator_groups=annotator_groups,
        example_list=example_list,
        match_mode=match_mode,
        category_breakdown=category_breakdown,
    )

    return f1_results
