#!/usr/bin/env python3
from pathlib import Path
import argparse
import logging
from tabulate import tabulate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGNS_PATH = PROJECT_ROOT / "factgenie" / "campaigns"

from factgenie.workflows import load_campaign, generate_campaign_index
from factgenie.analysis import (
    compute_span_index,
    format_group_id,
    get_common_examples,
)
from factgenie.bin.run import create_app

# Initialize the Flask app
app = create_app()


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_f1_scores(span_index, ref_group_ids, hyp_group_ids, match_mode="hard", category_breakdown=False):
    """
    Compute precision, recall, and F1-score between reference and hypothesis spans.

    Args:
        span_index: DataFrame containing all spans from both annotator groups
        ref_group_id: Group ID of the reference annotator
        hyp_group_id: Group ID of the hypothesis annotator
        match_mode: 'hard' requires exact category match, 'soft' allows any category
        category_breakdown: Return metrics broken down by annotation category

    Returns:
        Dictionary with overall precision, recall, F1 and optionally per-category metrics
    """

    ref_spans = span_index[span_index["annotator_group_id"].isin(ref_group_ids)]
    hyp_spans = span_index[span_index["annotator_group_id"].isin(hyp_group_ids)]

    # If either set is empty, return zeros
    if ref_spans.empty or hyp_spans.empty:
        logger.warning(f"No spans found for one or both annotator groups")
        result = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "ref_count": len(ref_spans),
            "hyp_count": len(hyp_spans),
        }
        if category_breakdown:
            result["categories"] = {}
        return result

    # Track metrics per annotation category
    categories = sorted(span_index["annotation_type"].unique())

    per_category_metrics = {cat: {"hyp_length": 0, "ref_length": 0, "overlap_length": 0} for cat in categories}

    # Group by example to process one example at a time
    metrics = {"hyp_length": 0, "ref_length": 0, "overlap_length": 0}

    # Process each example independently
    for (dataset, split, setup_id, example_idx), example_group in span_index.groupby(
        ["dataset", "split", "setup_id", "example_idx"]
    ):
        # Get reference and hypothesis spans for this example
        example_ref_spans = example_group[example_group["annotator_group_id"].isin(ref_group_ids)]
        example_hyp_spans = example_group[example_group["annotator_group_id"].isin(hyp_group_ids)]

        if example_ref_spans.empty and example_hyp_spans.empty:
            continue

        # Track which parts of reference spans have been matched
        # For each character position, store which reference span has claimed it
        matched_positions = {}

        # For each hypothesis span, compute overlap with all reference spans
        for _, hyp_row in example_hyp_spans.iterrows():
            hyp_start = hyp_row["annotation_start"]
            hyp_end = hyp_row["annotation_end"]
            hyp_type = hyp_row["annotation_type"]
            hyp_length = hyp_end - hyp_start

            metrics["hyp_length"] += hyp_length
            per_category_metrics[hyp_type]["hyp_length"] += hyp_length

            # Find best overlapping reference spans
            best_overlaps = []

            # Check against all reference spans
            for _, ref_row in example_ref_spans.iterrows():
                ref_start = ref_row["annotation_start"]
                ref_end = ref_row["annotation_end"]
                ref_type = ref_row["annotation_type"]
                ref_id = id(ref_row)  # Unique identifier for this reference span

                # Check category match if in hard mode
                if match_mode == "hard" and hyp_type != ref_type:
                    continue

                # Calculate overlap
                overlap_start = int(max(hyp_start, ref_start))
                overlap_end = int(min(hyp_end, ref_end))
                overlap_length = int(max(0, overlap_end - overlap_start))

                if overlap_length > 0:
                    best_overlaps.append((overlap_start, overlap_end, overlap_length, ref_id))

            # Process overlaps
            if best_overlaps:
                # Sort by overlap length (descending)
                best_overlaps.sort(key=lambda x: x[2], reverse=True)

                # Track which positions have been claimed
                for overlap_start, overlap_end, overlap_length, ref_id in best_overlaps:
                    valid_overlap = 0

                    # Check each position in the overlap
                    for pos in range(overlap_start, overlap_end):
                        # If position not claimed or claimed by this reference, count it
                        if pos not in matched_positions or matched_positions[pos] == ref_id:
                            matched_positions[pos] = ref_id
                            valid_overlap += 1

                    if valid_overlap > 0:
                        metrics["overlap_length"] += valid_overlap
                        per_category_metrics[hyp_type]["overlap_length"] += valid_overlap

        # Process reference spans for recall computation
        for _, ref_row in example_ref_spans.iterrows():
            ref_start = ref_row["annotation_start"]
            ref_end = ref_row["annotation_end"]
            ref_type = ref_row["annotation_type"]
            ref_length = ref_end - ref_start

            metrics["ref_length"] += ref_length
            per_category_metrics[ref_type]["ref_length"] += ref_length

    # Calculate overall metrics
    precision = metrics["overlap_length"] / metrics["hyp_length"] if metrics["hyp_length"] > 0 else 0
    recall = metrics["overlap_length"] / metrics["ref_length"] if metrics["ref_length"] > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    result = {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "ref_count": len(ref_spans),
        "hyp_count": len(hyp_spans),
    }

    # Calculate per-category metrics if requested
    if category_breakdown:
        result["categories"] = {}
        for cat in categories:
            cat_metrics = per_category_metrics[cat]
            cat_precision = (
                cat_metrics["overlap_length"] / cat_metrics["hyp_length"] if cat_metrics["hyp_length"] > 0 else 0
            )
            cat_recall = (
                cat_metrics["overlap_length"] / cat_metrics["ref_length"] if cat_metrics["ref_length"] > 0 else 0
            )
            cat_f1 = (
                2 * cat_precision * cat_recall / (cat_precision + cat_recall) if (cat_precision + cat_recall) > 0 else 0
            )
            result["categories"][cat] = {
                "precision": round(cat_precision, 3),
                "recall": round(cat_recall, 3),
                "f1": round(cat_f1, 3),
            }

    return result


def compute_f1(
    ref_camp_id,
    ref_groups,
    hyp_camp_id,
    hyp_groups,
    match_mode="hard",
    category_breakdown=False,
    filter_datasets=None,
    filter_splits=None,
    filter_example_ids=None,
):
    """
    Compute precision, recall, F1 between reference and hypothesis annotator groups.
    Always aggregates all groups from each campaign.

    Args:
        reference: Tuple of (campaign_id, ann_groups) where ann_groups can be a list of group IDs or None
        hypothesis: Tuple of (campaign_id, ann_groups) where ann_groups can be a list of group IDs or None
        match_mode: 'hard' requires exact category match, 'soft' allows any category
        category_breakdown: Whether to compute metrics per annotation category
        filter_datasets: List of dataset IDs to include (None means include all)
        filter_splits: List of splits to include (None means include all)
        filter_example_ids: List of example IDs to include (None means include all)

    Returns:
        Dictionary with F1 metrics for aggregated reference-hypothesis pair
    """
    # Generate campaign index
    campaigns = generate_campaign_index(app, force_reload=True)

    # Load both campaigns
    loaded_campaigns = {}
    for camp_id in [ref_camp_id, hyp_camp_id]:
        if camp_id in campaigns:
            loaded_campaigns[camp_id] = load_campaign(app, camp_id)
        else:
            logger.error(f"Campaign {camp_id} not found")
            return {}

    # Skip if any campaign is missing
    if ref_camp_id not in loaded_campaigns or hyp_camp_id not in loaded_campaigns:
        logger.warning(f"Campaign {ref_camp_id} or {hyp_camp_id} not found, skipping.")
        return {}

    ref_camp = loaded_campaigns[ref_camp_id]
    hyp_camp = loaded_campaigns[hyp_camp_id]

    # If no groups specified, use all available groups
    if ref_groups is None:
        ref_groups = sorted(ref_camp.db["annotator_group"].unique())
        logger.info(f"No reference groups specified, using all {len(ref_groups)} available groups from {ref_camp_id}")

    if hyp_groups is None:
        hyp_groups = sorted(hyp_camp.db["annotator_group"].unique())
        logger.info(f"No hypothesis groups specified, using all {len(hyp_groups)} available groups from {hyp_camp_id}")

    # Find common examples between all reference and all hypothesis groups
    combinations_list = get_common_examples(ref_camp.db, hyp_camp.db, None, None)

    if not combinations_list:
        logger.warning(f"No common examples between campaigns {ref_camp_id} and {hyp_camp_id}")
        return {}

    # Filter combinations by dataset, split
    filtered_combinations = []
    for dataset, split, setup_id in combinations_list:
        if filter_datasets and dataset not in filter_datasets:
            continue
        if filter_splits and split not in filter_splits:
            continue
        filtered_combinations.append((dataset, split, setup_id))

    if not filtered_combinations:
        logger.warning(f"No examples left after filtering")
        return {}

    # Get span index for both campaigns
    selected_campaigns = set([ref_camp_id, hyp_camp_id])
    span_index = compute_span_index(app, selected_campaigns, campaigns)

    logger.info(f"Common examples found between the campaign:")
    for dataset, split, setup_id in filtered_combinations:
        logger.info(f"  {dataset} - {split} - {setup_id}")

    # Filter span index to only include these combinations
    filtered_span_index = span_index[
        span_index.apply(
            lambda x: (x["dataset"], x["split"], x["setup_id"]) in filtered_combinations,
            axis=1,
        )
    ]

    # Further filter by example IDs if specified
    if filter_example_ids:
        filtered_span_index = filtered_span_index[filtered_span_index["example_idx"].isin(filter_example_ids)]

    ref_group_ids = [format_group_id(ref_camp_id, group_id) for group_id in ref_groups]
    hyp_group_ids = [format_group_id(hyp_camp_id, group_id) for group_id in hyp_groups]

    # Filter span index to only include reference and hypothesis groups
    filtered_span_index = filtered_span_index[
        filtered_span_index["annotator_group_id"].isin(ref_group_ids + hyp_group_ids)
    ]
    filtered_span_index = filtered_span_index.reset_index(drop=True)

    # Count unique examples for logging
    unique_examples = filtered_span_index[["dataset", "split", "setup_id", "example_idx"]].drop_duplicates()
    logger.info(f"Computing F1 scores between campaigns on {len(unique_examples)} examples")

    # Compute F1 scores
    results = compute_f1_scores(
        filtered_span_index,
        ref_group_ids,
        hyp_group_ids,
        match_mode,
        category_breakdown,
    )

    # Store results with appropriate key
    comparison_key = f"{ref_camp_id} :: {hyp_camp_id}"
    all_results = {comparison_key: results}

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Compute precision, recall, and F1-score between reference and hypothesis annotator groups"
    )
    parser.add_argument(
        "--ref-campaign",
        type=str,
        required=True,
        help="Reference campaign ID",
    )
    parser.add_argument(
        "--ref-groups",
        type=int,
        nargs="*",
        help="Reference annotator groups (if not specified, all groups are used)",
    )
    parser.add_argument(
        "--hyp-campaign",
        type=str,
        required=True,
        help="Hypothesis campaign ID",
    )
    parser.add_argument(
        "--hyp-groups",
        type=int,
        nargs="*",
        help="Hypothesis annotator groups (if not specified, all groups are used)",
    )
    parser.add_argument(
        "--match-mode",
        type=str,
        choices=["hard", "soft"],
        default="hard",
        help="Match mode: 'hard' requires same category, 'soft' allows any category",
    )
    parser.add_argument(
        "--category-breakdown",
        action="store_true",
        help="Calculate metrics per annotation category",
    )
    parser.add_argument(
        "--filter-datasets",
        type=str,
        nargs="+",
        help="Only include these datasets",
    )
    parser.add_argument(
        "--filter-splits",
        type=str,
        nargs="+",
        help="Only include these splits",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for the F1 results (JSON format)",
    )
    args = parser.parse_args()

    # Define filters for specific datasets, splits, and example IDs
    filter_datasets = args.filter_datasets  # None or list of datasets
    filter_splits = args.filter_splits  # None or list of splits
    filter_example_ids = None  # Include all example IDs

    # Compute F1 scores
    f1_results = compute_f1(
        ref_camp_id=args.ref_campaign,
        ref_groups=args.ref_groups,
        hyp_camp_id=args.hyp_campaign,
        hyp_groups=args.hyp_groups,
        match_mode=args.match_mode,
        category_breakdown=args.category_breakdown,
        filter_datasets=filter_datasets,
        filter_splits=filter_splits,
        filter_example_ids=filter_example_ids,
    )

    if not f1_results:
        print(f"\nNo results found. Check that the campaigns and groups exist and share examples.")
        return

    # Print filtering information
    print("\nFiltering options applied:")
    print(f"  Match mode: {args.match_mode}")
    print(f"  Datasets: {filter_datasets if filter_datasets else 'All'}")
    print(f"  Splits: {filter_splits if filter_splits else 'All'}")
    print(f"  Example IDs: {filter_example_ids if filter_example_ids else 'All'}")
    print(f"  Reference groups: {args.ref_groups if args.ref_groups else 'All'}")
    print(f"  Hypothesis groups: {args.hyp_groups if args.hyp_groups else 'All'}")

    # Print results in a table
    print("\nPrecision, Recall and F1 scores:")
    headers = [
        "Reference :: Hypothesis",
        "Precision",
        "Recall",
        "F1",
        "Ref Count",
        "Hyp Count",
    ]

    table_data = []

    for comparison_key, metrics in f1_results.items():
        table_data.append(
            [
                comparison_key,
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
                metrics["ref_count"],
                metrics["hyp_count"],
            ]
        )

    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Print category breakdown if requested
    if args.category_breakdown:
        for comparison_key, metrics in f1_results.items():
            if "categories" in metrics:
                print(f"\nMetrics breakdown by category for {comparison_key}:")
                cat_headers = [
                    "Category",
                    "Precision",
                    "Recall",
                    "F1",
                ]
                cat_data = []

                for cat, cat_metrics in metrics["categories"].items():
                    cat_data.append(
                        [
                            cat,
                            cat_metrics["precision"],
                            cat_metrics["recall"],
                            cat_metrics["f1"],
                        ]
                    )

                print(tabulate(cat_data, headers=cat_headers, tablefmt="grid"))

    # Save to JSON if output file specified
    if args.output:
        import json

        with open(args.output, "w") as f:
            json.dump(f1_results, f, indent=2)
        logger.info(f"F1 results saved to {args.output}")


if __name__ == "__main__":
    main()
