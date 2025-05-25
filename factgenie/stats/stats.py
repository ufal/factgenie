#!/usr/bin/env python3

"""Compute campaign statistics."""

import logging

from flask import current_app as app

logger = logging.getLogger(__name__)


def _initialize_stats(campaign, annotator_group=None):
    return {
        "campaign_id": campaign.campaign_id,
        "annotator_group": annotator_group,
        "total_annotations": 0,
        "total_examples": 0,
        "annotations_per_example": 0.0,
        "empty_examples_percentage": 100.0,
        "avg_annotation_length": 0.0,
        "examples_with_annotations": 0,
        "examples_without_annotations": 0,
    }


def compute_campaign_stats_internal(
    campaign, example_index, annotator_group=None, filter_datasets=None, filter_splits=None
):
    """
    Internal function to compute statistics for a specific campaign based on a pre-filtered example_index.

    Args:
        campaign: Campaign object
        example_index: DataFrame with example-level information
        annotator_group: Optional annotator group to filter by
        filter_datasets: List of datasets to include (default: None = all)
        filter_splits: List of splits to include (default: None = all)

    Returns:
        Dictionary with statistics
    """
    # Filter by annotator group if specified
    if annotator_group is not None:
        example_index = example_index[example_index["annotator_group"] == annotator_group]

    # Filter by datasets if specified
    if filter_datasets:
        example_index = example_index[example_index["dataset"].isin(filter_datasets)]

    # Filter by splits if specified
    if filter_splits:
        example_index = example_index[example_index["split"].isin(filter_splits)]

    if example_index.empty:
        logger.warning(f"No examples found for campaign {campaign.campaign_id} after applying filters.")
        return _initialize_stats(
            campaign,
            annotator_group=annotator_group,
        )

    total_annotations = example_index["annotations"].apply(lambda x: len(x) if isinstance(x, list) else 0).sum()

    # Calculate average annotation length
    all_annotation_texts = []
    for annotations_list in example_index["annotations"]:
        if isinstance(annotations_list, list):
            for annotation in annotations_list:
                if isinstance(annotation, dict) and "text" in annotation and isinstance(annotation["text"], str):
                    all_annotation_texts.append(annotation["text"])

    avg_annotation_length = 0.0
    if all_annotation_texts:
        avg_annotation_length = round(sum(len(text) for text in all_annotation_texts) / len(all_annotation_texts), 2)

    # Count examples with no annotations
    empty_examples_count = (
        example_index["annotations"].apply(lambda anns: not (isinstance(anns, list) and len(anns) > 0)).sum()
    )

    total_examples = len(example_index)
    annotations_per_example = round(total_annotations / total_examples, 2) if total_examples > 0 else 0.0
    empty_examples_percentage = round(100 * empty_examples_count / total_examples, 2) if total_examples > 0 else 100.0

    return {
        "campaign_id": campaign.campaign_id,
        "annotator_group": annotator_group,
        "filter_datasets": filter_datasets,
        "filter_splits": filter_splits,
        "total_annotations": int(total_annotations),
        "total_examples": int(total_examples),
        "annotations_per_example": annotations_per_example,
        "empty_examples_percentage": empty_examples_percentage,
        "avg_annotation_length": avg_annotation_length,
        "examples_with_annotations": int(total_examples - empty_examples_count),
        "examples_without_annotations": int(empty_examples_count),
    }


def compute_stats(
    campaign_id: str, annotator_group: int = None, include_dataset: list = None, include_split: list = None
):
    """
    Compute and retrieve statistics for a campaign.

    Args:
        campaign_id: The ID of the campaign.
        annotator_group: Optional annotator group to filter by.
        include_dataset: List of dataset IDs to include.
        include_split: List of splits to include.

    Returns:
        Dictionary with statistics or None if campaign not found.
    """
    from factgenie.analysis import generate_example_index
    from factgenie.workflows import generate_campaign_index, load_campaign

    campaign_index = generate_campaign_index(app, force_reload=True)

    if campaign_id not in campaign_index:
        logger.error(f"Campaign {campaign_id} not found.")
        return None

    campaign = load_campaign(app, campaign_id)
    if not campaign:
        logger.error(f"Could not load campaign {campaign_id}.")
        return None

    # Generate example index for the campaign
    example_index_df = generate_example_index(app, campaign)

    if example_index_df.empty:
        logger.warning(f"No examples found in the initial index for campaign {campaign.campaign_id}")
        return _initialize_stats(campaign, annotator_group, include_dataset, include_split)

    stats = compute_campaign_stats_internal(
        campaign=campaign,
        example_index=example_index_df,
        annotator_group=annotator_group,
        filter_datasets=include_dataset,
        filter_splits=include_split,
    )

    return stats
