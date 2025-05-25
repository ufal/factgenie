#!/usr/bin/env python3

"""Gamma agreement computation for factgenie annotation campaigns."""

import logging
import os
import traceback

import numpy as np
import pandas as pd  # Added for type hinting and potential use
from flask import current_app as app
from tqdm import tqdm

from factgenie.analysis import format_group_id

logger = logging.getLogger(__name__)


def save_plot(alignment, plt, ntb, save_dir, img_name):
    """Save an alignment plot."""
    fig, ax = plt.subplots(figsize=(10, 2))
    ntb.plot_alignment(alignment, ax)
    plt.tight_layout()

    # Save the plot
    plt.savefig(
        os.path.join(save_dir, f"{img_name}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def compute_s_empty_score(example_spans):
    """Compute the s_empty score for a set of spans."""
    ann_count = example_spans.shape[0]
    s_empty_score = 1 / (1 + ann_count)

    return s_empty_score


def compute_gamma_score(
    example_spans,
    dissim,
    soft_gamma,
    pa,
    Segment,
    example_id,
    save_plots_config=None,
):
    continuum = pa.Continuum()
    for _, row in example_spans.iterrows():
        continuum.add(
            str(row["annotator_group_id"]),
            Segment(row["annotation_start"], row["annotation_end"]),
            str(row["annotation_type"]),
        )

    gamma_value = np.nan
    try:
        # Set random seed for reproducibility of pygamma's internal processes
        np.random.seed(42)
        gamma_results = continuum.compute_gamma(dissim, soft=soft_gamma)
        gamma_value = gamma_results.gamma

        if save_plots_config:
            plt, ntb, save_dir = save_plots_config
            save_plot(gamma_results.best_alignment, plt, ntb, save_dir, example_id)

    except Exception as e:
        logger.error(f"Error computing gamma for example {example_id}: {e}")
        logger.error(traceback.format_exc())
        gamma_value = 0.0  # Default to 0.0 on error, as in original code

    return gamma_value


def _initialize_gamma_metrics():
    """Initialize metrics for gamma calculation."""
    return {
        "gamma_scores": [],
        "s_empty_scores": [],
        "gamma_mean": 0.0,
        "s_empty_mean": 0.0,
    }


def _get_camp_group_spans(example_spans: pd.DataFrame, annotator_groups: list) -> pd.DataFrame:
    """
    Filter spans to include only those from specified campaign-annotator group pairs.
    """
    # Create a mask to filter spans
    mask = pd.Series(False, index=example_spans.index)

    # Add each annotator group's spans to the mask
    for camp_id, group in annotator_groups:
        group_mask = example_spans["campaign_id"] == camp_id

        # If group is specified (not None), add that condition
        if group is not None:
            group_mask &= example_spans["annotator_group"] == group

        mask |= group_mask

    # Apply the mask to get only spans from the specified groups
    filtered_spans = example_spans[mask].copy()

    # Add a unique identifier for each annotator group to use in gamma calculation
    filtered_spans["annotator_group_id"] = filtered_spans.apply(
        lambda row: format_group_id(row["campaign_id"], row["annotator_group"]), axis=1
    )

    return filtered_spans


def _process_example_gamma(
    example_tuple: tuple,
    span_index: pd.DataFrame,
    annotator_groups: list,
    dissim,  # pygamma_agreement.CombinedCategoricalDissimilarity
    soft_gamma: bool,
    pa,  # pygamma_agreement module
    Segment,  # pyannote.core.Segment class
    save_plots_config: tuple = None,
):
    """
    Process a single example to compute its gamma and s_empty score.
    Returns a tuple (gamma_value, s_empty_value).
    gamma_value can be float or np.nan. s_empty_value can be float or np.nan.
    """
    dataset, split, setup_id, example_idx = example_tuple

    example_spans = span_index[
        (span_index["dataset"] == dataset)
        & (span_index["split"] == split)
        & (span_index["setup_id"] == setup_id)
        & (span_index["example_idx"] == example_idx)
    ]
    example_spans = _get_camp_group_spans(example_spans, annotator_groups)

    # Remove empty segments (where start == end)
    example_spans = example_spans[example_spans["annotation_start"] != example_spans["annotation_end"]]

    # Check number of unique annotator groups for this example after filtering
    # Gamma requires at least 2 annotators. 'annotator_group_id' should uniquely identify an annotator.
    unique_annotators = example_spans["annotator_group_id"].unique()

    example_id = f"{dataset}_{split}_{setup_id}_{example_idx}"

    if len(unique_annotators) < 2:
        gamma_value = np.nan
        s_empty = compute_s_empty_score(example_spans)

    else:
        gamma_value = compute_gamma_score(
            example_spans=example_spans,
            dissim=dissim,
            soft_gamma=soft_gamma,
            pa=pa,
            Segment=Segment,
            example_id=example_id,
            save_plots_config=save_plots_config,
        )
        s_empty = np.nan

    return gamma_value, s_empty


def compute_gamma_scores(
    span_index, annotator_groups, example_list, alpha, beta, delta_empty, soft_gamma, save_plots_dir
):
    """Compute gamma agreement score for the given examples."""

    # Import heavy dependencies here to avoid loading them if not needed
    # or if function exits early.
    try:
        import pygamma_agreement as pa
        from pyannote.core import Segment
    except ImportError:
        logger.error("pygamma-agreement or pyannote.core is not installed. Cannot compute gamma scores.")
        # Return initialized metrics, but with a clear indication of failure if possible,
        # or raise the error. For now, returning empty/default metrics.
        return _initialize_gamma_metrics()

    if example_list.empty or span_index.empty:
        logger.warning("Input example_list or span_index to compute_gamma_scores is empty.")
        return _initialize_gamma_metrics()

    dissim = pa.CombinedCategoricalDissimilarity(alpha=alpha, beta=beta, delta_empty=delta_empty)

    gamma_scores = []
    s_empty_scores = []

    save_plots_config = None
    if save_plots_dir:
        try:
            import matplotlib.pyplot as plt

            ntb = pa.notebook.Notebook()
            os.makedirs(save_plots_dir, exist_ok=True)
            save_plots_config = (plt, ntb, save_plots_dir)
        except ImportError:
            logger.warning("Matplotlib not installed. Cannot save plots.")
            save_plots_dir = None  # Disable plotting

    logger.info(f"Computing gamma scores over {len(example_list)} examples.")

    pbar = tqdm(total=len(example_list), desc="Computing gamma score")

    # Store original logging level to restore it later
    original_logging_level = logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel(logging.WARNING)

    for example_tuple in example_list.itertuples(index=False):
        gamma_val, s_empty_val = _process_example_gamma(
            example_tuple,
            span_index,
            annotator_groups,
            dissim,
            soft_gamma,
            pa,  # pygamma_agreement module
            Segment,  # Segment class
            save_plots_config,
        )

        if not np.isnan(gamma_val):
            gamma_scores.append(gamma_val)
        if not np.isnan(s_empty_val):
            s_empty_scores.append(s_empty_val)

        current_avg_gamma = np.nanmean(gamma_scores) if gamma_scores else 0.0
        pbar.set_postfix({"avg_gamma": f"{current_avg_gamma:.3f}"})
        pbar.update(1)

    logging.getLogger().setLevel(original_logging_level)  # Restore logging level
    pbar.close()

    final_gamma_mean = float(np.nanmean(gamma_scores)) if gamma_scores else 0.0
    final_s_empty_mean = float(np.nanmean(s_empty_scores)) if s_empty_scores else 0.0

    return {
        "gamma_scores": gamma_scores,  # List of actual gamma values computed
        "s_empty_scores": s_empty_scores,  # List of actual s_empty values computed
        "gamma_mean": round(final_gamma_mean, 3),
        "s_empty_mean": round(final_s_empty_mean, 3),
        "example_count": len(example_list),
    }


def compute_gamma(
    campaign_ids,
    groups,
    alpha=1.0,
    beta=1.0,
    delta_empty=1.0,
    soft_gamma=True,
    include_dataset=None,
    include_split=None,
    include_example_id=None,
    save_plots=None,  # Renamed from save_plots_dir for clarity if it's a path
):
    """
    Compute gamma agreement score between multiple annotator groups.

    Args:
        campaign_ids: List of campaign IDs
        groups: List of annotator groups (None for all)
        alpha: Coefficient weighting the positional dissimilarity
        beta: Coefficient weighting the categorical dissimilarity
        delta_empty: Empty dissimilarity value
        soft_gamma: Use soft version of gamma score (counting splitted segments as a single segment)
        include_dataset: List of dataset IDs to include (None means include all)
        include_split: List of splits to include (None means include all)
        include_example_id: List of example IDs to include (None means include all)
        save_plots: Directory to save alignment plots (None for no plots)

    Returns:
        Dictionary with gamma scores
    """
    from factgenie.analysis import (
        assert_common_categories,
        compute_span_index,
        get_example_list,
    )
    from factgenie.workflows import generate_campaign_index

    # Generate campaign index
    campaign_index = generate_campaign_index(app, force_reload=True)

    if not assert_common_categories(campaign_ids, campaign_index):
        return None

    annotator_groups = [(campaign_id, group) for campaign_id, group in zip(campaign_ids, groups)]

    span_index = compute_span_index(app, campaign_ids, campaign_index)
    example_list = get_example_list(
        campaign_index, annotator_groups, include_dataset, include_split, include_example_id
    )

    # Compute gamma scores
    gamma_results = compute_gamma_scores(  # Call the refactored function
        span_index=span_index,
        annotator_groups=annotator_groups,
        example_list=example_list,
        alpha=alpha,
        beta=beta,
        delta_empty=delta_empty,
        soft_gamma=soft_gamma,
        save_plots_dir=save_plots,  # Pass the directory path
    )

    return gamma_results
