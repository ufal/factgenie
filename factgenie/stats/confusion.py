"""Functions for computing confusion matrices between annotation groups."""

import logging

import numpy as np
import pandas as pd
from flask import current_app as app

from factgenie.analysis import get_ref_hyp_spans

logger = logging.getLogger(__name__)


def plot_confusion_matrix(conf_mat_np, labels, normalize, output_file, ref_group_id, hyp_group_id):
    """
    Plot the confusion matrix using seaborn heatmap.

    Args:
        confusion_matrix_data: Dictionary with confusion matrix and related data
        output_file: Path to save the image
        normalize: Whether to normalize the matrix by row
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError("Please install matplotlib and seaborn for plotting confusion matrices.")

    # Normalize by row (reference) if requested
    if normalize and np.sum(conf_mat_np) > 0:
        row_sums = conf_mat_np.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        conf_mat_np = conf_mat_np / row_sums

    plt.figure(figsize=(5, 5))

    # Create the heatmap
    if normalize:
        fmt = ".2f"
        vmax = 1.0
    else:
        fmt = "d"
        vmax = None

    cmap = sns.cubehelix_palette(as_cmap=True)

    ax = sns.heatmap(
        conf_mat_np,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=vmax,
    )

    # Set up the axes
    plt.ylabel("Reference (True)")
    plt.xlabel("Hypothesis (Predicted)")

    # Add title
    plt.title(f"Confusion Matrix: {hyp_group_id} → {ref_group_id}")

    plt.tight_layout()

    # Save the figure if an output path is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Confusion matrix plot saved to {output_file}")

    return plt


def _fill_confusion_matrix(example_hyp_spans, example_ref_spans, confusion_matrix):
    # For tracking matches to avoid double counting
    matched_ref_indices = set()

    # For each hypothesis span, find best matching reference span
    for hyp_idx, hyp_row in example_hyp_spans.iterrows():
        hyp_start = hyp_row["annotation_start"]
        hyp_end = hyp_row["annotation_end"]
        hyp_type = hyp_row["annotation_type"]

        best_match = None
        best_overlap = 0
        best_ref_idx = None

        # Check against all reference spans
        for ref_idx, ref_row in example_ref_spans.iterrows():
            # Skip if this reference span has already been matched
            if ref_idx in matched_ref_indices:
                continue

            ref_start = ref_row["annotation_start"]
            ref_end = ref_row["annotation_end"]
            ref_type = ref_row["annotation_type"]

            # Calculate overlap
            overlap_start = max(hyp_start, ref_start)
            overlap_end = min(hyp_end, ref_end)
            overlap_length = max(0, overlap_end - overlap_start)

            # Only consider as match if there's any overlap at all
            if overlap_length > 0 and overlap_length > best_overlap:
                best_overlap = overlap_length
                best_match = ref_type
                best_ref_idx = ref_idx

        # Update confusion matrix if we found a match
        if best_match is not None:
            confusion_matrix[best_match][hyp_type] += 1
            matched_ref_indices.add(best_ref_idx)

    return confusion_matrix


def _process_example(example_tuple, span_index, annotator_groups, confusion_matrix):

    dataset, split, setup_id, example_idx = example_tuple

    # Filter spans for this example
    example_spans = span_index[
        (span_index["dataset"] == dataset)
        & (span_index["split"] == split)
        & (span_index["setup_id"] == setup_id)
        & (span_index["example_idx"] == example_idx)
    ]
    example_ref_spans, example_hyp_spans = get_ref_hyp_spans(example_spans, annotator_groups)

    if not example_ref_spans.empty and not example_hyp_spans.empty:
        confusion_matrix = _fill_confusion_matrix(
            example_hyp_spans=example_hyp_spans, example_ref_spans=example_ref_spans, confusion_matrix=confusion_matrix
        )

    return confusion_matrix


def compute_confusion_matrix_internal(span_index, annotator_groups, example_list):
    if example_list.empty or span_index.empty:
        logger.warning("Span index is empty.")
        return None

    # Initialize tracking metrics
    category_indices = sorted(span_index["annotation_type"].unique())

    # Initialize confusion matrix
    confusion_matrix = np.zeros((len(category_indices), len(category_indices)), dtype=int)

    # Process each example
    for example_tuple in example_list.itertuples(index=False):
        confusion_matrix = _process_example(example_tuple, span_index, annotator_groups, confusion_matrix)

    return {
        "confusion_matrix": confusion_matrix,
        "category_indices": category_indices,
    }


def compute_confusion_matrix(
    ref_camp_id,
    ref_group,
    hyp_camp_id,
    hyp_group,
    include_dataset=None,
    include_split=None,
    include_example_id=None,
):
    """
    Generate confusion matrix between reference and hypothesis annotator groups.

    Args:
        ref_camp_id: Reference campaign ID
        ref_group: Reference annotator group (None means all groups)
        hyp_camp_id: Hypothesis campaign ID
        hyp_group: Hypothesis annotator group (None means all groups)
        include_dataset: List of dataset IDs to include (None means include all)
        include_split: List of splits to include (None means include all)
        include_example_ids: List of example IDs to include (None means include all)

    Returns:
        Dictionary with confusion matrix and related data
    """
    from factgenie.analysis import (
        assert_common_categories,
        compute_span_index,
        get_example_list,
    )
    from factgenie.workflows import generate_campaign_index

    # Generate campaign index
    campaign_index = generate_campaign_index(app, force_reload=True)
    campaign_ids = [ref_camp_id, hyp_camp_id]

    if not assert_common_categories(campaign_ids, campaign_index):
        return None

    annotator_groups = [
        (ref_camp_id, ref_group),
        (hyp_camp_id, hyp_group),
    ]

    span_index = compute_span_index(app, campaign_ids, campaign_index)

    # Remove all annotations that have type -1
    span_index = span_index[span_index["annotation_type"] != -1]

    example_list = get_example_list(
        campaign_index, annotator_groups, include_dataset, include_split, include_example_id
    )

    if example_list.empty:
        logger.error("No examples found that are annotated by both groups")
        return None

    logger.info(f"Computing confusion matrix on {len(example_list)} examples")

    # Compute confusion matrix
    res = compute_confusion_matrix_internal(
        span_index=span_index,
        annotator_groups=annotator_groups,
        example_list=example_list,
    )

    if res is None:
        logger.error("Failed to compute confusion matrix due to empty span index or example list.")
        return None

    category_map = {}
    category_indices = res["category_indices"]

    try:
        # Attempt to get category names from campaign configuration
        annotation_span_categories = campaign_index[ref_camp_id].metadata["config"]["annotation_span_categories"]
        for i, cat in enumerate(annotation_span_categories):
            category_map[i] = cat["name"]
    except Exception as e:
        logger.warning("Could not retrieve category names from campaign configuration. Using numeric indices instead.")
        for i, name in enumerate(category_indices):
            category_map[name] = f"Category {name}"

    # Map numeric category IDs to their labels
    labeled_categories = [category_map.get(cat, f"Category {cat}") for cat in category_indices]

    # Create a DataFrame for better display
    res["confusion_matrix"] = pd.DataFrame(
        res["confusion_matrix"], index=labeled_categories, columns=labeled_categories
    )
    res["labeled_categories"] = labeled_categories

    return res
