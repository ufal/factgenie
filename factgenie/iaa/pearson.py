#!/usr/bin/env python3

"""Pearson correlation computation for factgenie annotation campaigns."""

import logging

import numpy as np
import pandas as pd
from flask import current_app as app
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


def _initialize_metrics(categories):
    """Initialize tracking metrics for overall and per-category calculations."""
    metrics = {
        "micro_pearson": 0.0,
        "micro_p_value": 0.0,
        "macro_pearson": 0.0,
        "macro_p_value": 0.0,
        "example_count": 0,
        "categories": {},
    }

    if categories:
        metrics["categories"] = {cat: metrics.copy() for cat in categories}

    return metrics


def compute_pearson_scores(span_index, example_list, annotator_groups):
    """
    Compute Pearson correlation between counts from two annotator groups.

    Args:
        span_index: DataFrame with individual annotations
        example_list: DataFrame with examples to consider for the computation
        annotator_groups: List of (campaign_id, group) tuples (must be exactly 2)

    Returns:
        Dictionary with correlation metrics
    """
    # Get unique categories from all annotations
    categories = sorted(span_index["annotation_type"].unique())
    metrics = _initialize_metrics(categories)

    # Process each example to get counts by group and category
    counts = []
    for dataset, split, setup_id, example_idx in example_list.itertuples(index=False):
        # Filter spans for this example
        example_spans = span_index[
            (span_index["dataset"] == dataset)
            & (span_index["split"] == split)
            & (span_index["setup_id"] == setup_id)
            & (span_index["example_idx"] == example_idx)
        ]

        # Count annotations for each group and category
        for group_idx, (campaign_id, group) in enumerate(annotator_groups):
            group_filter = example_spans["campaign_id"] == campaign_id
            if group is not None:
                group_filter &= example_spans["annotator_group"] == group

            group_spans = example_spans[group_filter]

            for category in categories:
                counts.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "setup_id": setup_id,
                        "example_idx": example_idx,
                        "group_id": group_idx,
                        "category": category,
                        "count": sum(group_spans["annotation_type"] == category),
                    }
                )

    # Convert to DataFrame and pivot for correlation calculation
    counts_df = pd.DataFrame(counts)

    # Calculate micro correlation (all categories together)
    pivot_micro = counts_df.pivot_table(
        index=["dataset", "split", "setup_id", "example_idx", "category"], columns="group_id", values="count"
    ).reset_index()

    metrics["micro_pearson"], metrics["micro_p_value"] = pearsonr(pivot_micro[0], pivot_micro[1])

    # Calculate per-category correlations
    category_correlations = []
    category_p_values = []

    for category in categories:
        cat_data = counts_df[counts_df["category"] == category]
        pivot_cat = cat_data.pivot_table(
            index=["dataset", "split", "setup_id", "example_idx"], columns="group_id", values="count"
        ).reset_index()

        corr, p_val = pearsonr(pivot_cat[0], pivot_cat[1])
        metrics["categories"][category]["micro_pearson"] = corr
        metrics["categories"][category]["micro_p_value"] = p_val

        category_correlations.append(corr)
        category_p_values.append(p_val)

    metrics["macro_pearson"] = float(np.mean(category_correlations))
    metrics["macro_p_value"] = float(np.mean(category_p_values))
    metrics["example_count"] = len(example_list)
    return metrics


def compute_pearson(
    campaign1,
    group1,
    campaign2,
    group2,
    include_dataset=None,
    include_split=None,
    include_example_id=None,
):
    """
    Compute Pearson correlation between counts from two annotator groups.

    Args:
        campaign1: First campaign ID
        group1: First annotator group (None means all groups)
        campaign2: Second campaign ID
        group2: Second annotator group (None means all groups)
        include_dataset: List of dataset IDs to include (None means include all)
        include_split: List of splits to include (None means include all)
        include_example_id: List of example IDs to include (None means include all)

    Returns:
        Dictionary with correlation metrics
    """
    from factgenie.analysis import (
        assert_common_categories,
        compute_span_index,
        get_example_list,
    )
    from factgenie.workflows import generate_campaign_index

    # Generate campaign index
    campaign_index = generate_campaign_index(app, force_reload=True)
    campaign_ids = [campaign1, campaign2]

    if not assert_common_categories(campaign_ids, campaign_index):
        return None

    annotator_groups = [
        (campaign1, group1),
        (campaign2, group2),
    ]

    span_index = compute_span_index(app, campaign_ids, campaign_index)
    example_list = get_example_list(
        campaign_index, annotator_groups, include_dataset, include_split, include_example_id
    )

    logger.info(f"Computing Pearson correlation on {len(example_list)} examples")

    pearson_results = compute_pearson_scores(
        span_index=span_index,
        example_list=example_list,
        annotator_groups=annotator_groups,
    )

    return pearson_results
