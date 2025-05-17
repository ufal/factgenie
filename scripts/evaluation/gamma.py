#!/usr/bin/env python3

import argparse
import logging
import os
import pandas as pd
import pygamma_agreement as pa
import json
import traceback
import numpy as np

from pathlib import Path
from tabulate import tabulate
from tqdm import tqdm
from pyannote.core import Segment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGNS_PATH = PROJECT_ROOT / "factgenie" / "campaigns"

from factgenie.workflows import load_campaign, generate_campaign_index, get_annotation_index
from factgenie.analysis import format_group_id, compute_iaa_dfs, compute_span_counts, compute_span_index
from factgenie.bin.run import create_app

# Initialize the Flask app
app = create_app()


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def save_plot(alignment, plt, ntb, save_dir, img_name):
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


def assert_common_categories(campaign_ids, campaign_index):
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
    return True


def compute_s_empty_score(example_spans):
    ann_count = example_spans.shape[0]
    s_empty_score = 1 / (1 + ann_count)

    return s_empty_score


def get_example_list(
    campaign_index, annotator_groups, filter_datasets=None, filter_splits=None, filter_example_ids=None
):
    """
    Get list of examples to consider for gamma score computation.

    Args:
        campaign_index: Dictionary mapping campaign_id to campaign object
        annotator_groups: List of tuples (campaign_id, ann_group)
        filter_datasets: List of dataset IDs to include (None means include all)
        filter_splits: List of splits to include (None means include all)
        filter_example_ids: List of example IDs to include (None means include all)

    Returns:
        DataFrame with columns dataset, split, setup_id, example_idx
    """
    all_examples = []

    # Collect examples from each campaign and annotator group
    for campaign_id, ann_group in annotator_groups:
        campaign = campaign_index[campaign_id]
        campaign_examples = campaign.db.copy()

        # Filter by annotator group if specified (not 'all')
        if ann_group != "all":
            campaign_examples = campaign_examples[campaign_examples["annotator_group"] == ann_group]

        # Filter examples with finished status
        campaign_examples = campaign_examples[campaign_examples["status"] == "finished"]

        # Filter by datasets if specified
        if filter_datasets:
            campaign_examples = campaign_examples[campaign_examples["dataset"].isin(filter_datasets)]

        # Filter by splits if specified
        if filter_splits:
            campaign_examples = campaign_examples[campaign_examples["split"].isin(filter_splits)]

        # Filter by example_ids if specified
        if filter_example_ids:
            campaign_examples = campaign_examples[campaign_examples["example_idx"].isin(filter_example_ids)]

        # Keep only relevant columns
        campaign_examples = campaign_examples[["dataset", "split", "setup_id", "example_idx"]]

        # Add to the list of all examples
        all_examples.append(campaign_examples)

    # Combine all example sets
    if not all_examples:
        return pd.DataFrame(columns=["dataset", "split", "setup_id", "example_idx"])

    combined_examples = pd.concat(all_examples, ignore_index=True)

    # Count occurrences of each example
    example_counts = (
        combined_examples.groupby(["dataset", "split", "setup_id", "example_idx"]).size().reset_index(name="count")
    )

    # Keep only examples that appear in all campaign/group combinations
    common_examples = example_counts[example_counts["count"] == len(annotator_groups)]

    # Get the final list of examples
    example_list = common_examples[["dataset", "split", "setup_id", "example_idx"]]

    return example_list


# `alpha`: coefficient weighting the *positional* dissimilarity value, defaults to 1
# `beta`: coefficient weighting the *categorical* dissimilarity value, defaults to 1
# `delta_empty`: empty dissimilarity value, defaults to 1
def compute_gamma_score(span_index, example_list, alpha, beta, delta_empty, soft, save_plots):
    dissim = pa.CombinedCategoricalDissimilarity(alpha=alpha, beta=beta, delta_empty=delta_empty)

    gamma_scores = []
    s_empty_scores = []
    running_avg = 0

    if save_plots:
        import matplotlib.pyplot as plt

        ntb = pa.notebook.Notebook()
        os.makedirs(save_plots, exist_ok=True)

    logger.info(f"Computing gamma scores over {len(example_list)} examples.")

    # Preview first 10 examples in the list
    logger.info(f"Examples for which the gamma score will be computed:")
    print(example_list)

    # Create progress bar
    pbar = tqdm(total=len(example_list), desc="Computing gamma score")

    for dataset, split, setup_id, example_idx in example_list.itertuples(index=False):
        # Get corresponding spans for this example
        example_spans = span_index[
            (span_index["dataset"] == dataset)
            & (span_index["split"] == split)
            & (span_index["setup_id"] == setup_id)
            & (span_index["example_idx"] == example_idx)
        ]

        # Remove empty segments
        example_spans = example_spans[example_spans["annotation_start"] != example_spans["annotation_end"]]

        # Check number of unique annotator groups
        unique_annotators = example_spans["annotator_group_id"].unique()

        if len(unique_annotators) < 2:
            # One or both annotators did not add any annotation to this example
            s_empty = compute_s_empty_score(example_spans)
            s_empty_scores.append(s_empty)
            pbar.update(1)
            continue

        if len(unique_annotators) > 2:
            breakpoint()

        # Add each annotation to continuum
        continuum = pa.Continuum()

        for _, row in example_spans.iterrows():
            continuum.add(
                str(row["annotator_group_id"]),
                Segment(row["annotation_start"], row["annotation_end"]),
                str(row["annotation_type"]),
            )

        # Compute gamma score
        logging.getLogger().setLevel(logging.WARNING)
        try:
            np.random.seed(42)

            gamma_results = continuum.compute_gamma(dissim, soft=soft)
            gamma_scores.append(gamma_results.gamma)

            if save_plots:
                img_name = f"{dataset}_{split}_{setup_id}_{example_idx}"
                save_plot(gamma_results.best_alignment, plt, ntb, save_dir=save_plots, img_name=img_name)

        except Exception as e:
            traceback.print_exc()
            print(f"Error computing gamma for example {dataset}/{split}/{setup_id}/{example_idx}: {e}")
            gamma_scores.append(0.0)

        logging.getLogger().setLevel(logging.INFO)

        running_avg = np.mean(gamma_scores)
        pbar.set_postfix({"avg_gamma": f"{running_avg:.3f}"})
        pbar.update(1)

    pbar.close()

    gamma_score = float(np.mean(gamma_scores)) if gamma_scores else 0.0
    s_empty_score = float(np.mean(s_empty_scores)) if s_empty_scores else 0.0

    out = {
        "gamma_scores": gamma_scores,
        "s_empty_scores": s_empty_scores,
        "gamma": gamma_score,
        "s_empty": s_empty_score,
    }
    return out


def compute_gamma(
    annotator_groups,
    alpha=1.0,
    beta=1.0,
    delta_empty=1.0,
    soft=True,
    filter_datasets=None,
    filter_splits=None,
    filter_example_ids=None,
    save_plots=None,
):
    """
    Compute gamma agreement score between multiple annotator groups.

    Args:
        annotators: List of tuples (campaign_id, ann_group)
        alpha: Coefficient weighting the positional dissimilarity
        beta: Coefficient weighting the categorical dissimilarity
        delta_empty: Empty dissimilarity value
        soft: If True, use "soft" alignment allowing for partial matches
        filter_datasets: List of dataset IDs to include (None means include all)
        filter_splits: List of splits to include (None means include all)
        filter_example_ids: List of example IDs to include (None means include all)
        save_plots: Directory to save alignment plots (None for no plots)

    Returns:
        Dictionary with gamma scores
    """
    # Generate campaign index
    campaign_index = generate_campaign_index(app, force_reload=True)
    campaign_ids = [campaign_id for campaign_id, _ in annotator_groups]

    if not assert_common_categories(campaign_ids, campaign_index):
        return None

    span_index = compute_span_index(app, campaign_ids, campaign_index)
    example_list = get_example_list(
        campaign_index, annotator_groups, filter_datasets, filter_splits, filter_example_ids
    )

    # Compute gamma scores
    gamma_results = compute_gamma_score(
        span_index=span_index,
        example_list=example_list,
        alpha=alpha,
        beta=beta,
        delta_empty=delta_empty,
        soft=soft,
        save_plots=save_plots,
    )

    # Add metadata
    gamma_results["annotator_groups"] = annotator_groups
    gamma_results["example_count"] = len(example_list)
    gamma_results["alpha"] = alpha
    gamma_results["beta"] = beta
    gamma_results["delta_empty"] = delta_empty
    gamma_results["soft"] = soft

    return gamma_results


def main():
    parser = argparse.ArgumentParser(description="Compute gamma agreement scores between annotator groups")
    parser.add_argument(
        "--campaign",
        type=str,
        action="append",
        required=True,
        help="Campaign ID (can specify multiple times)",
    )
    parser.add_argument(
        "--group",
        type=str,
        action="append",
        help="Annotator group for corresponding campaign (can specify multiple times, 'all' for all groups)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Coefficient weighting the positional dissimilarity",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Coefficient weighting the categorical dissimilarity",
    )
    parser.add_argument("--delta-empty", type=float, default=1.0, help="Empty dissimilarity value")
    parser.add_argument(
        "--soft",
        action="store_true",
        help="Use soft version of gamma score (counting splitted segments as a single segment: see https://pygamma-agreement.readthedocs.io/en/latest/soft-gamma.html)",
    )
    parser.add_argument("--filter-datasets", type=str, nargs="+", help="Only include these datasets")
    parser.add_argument("--filter-splits", type=str, nargs="+", help="Only include these splits")
    parser.add_argument("--filter-example-ids", type=int, nargs="+", help="Only include these example IDs")
    parser.add_argument("--save-plots", type=str, help="Directory to save alignment plots")
    parser.add_argument("--output", type=str, help="Output file for the gamma results (JSON format)")
    args = parser.parse_args()

    # Generate campaign index
    campaigns = generate_campaign_index(app, force_reload=True)

    # Process campaign and group arguments
    annotator_groups = []

    if len(args.group) < len(args.campaign):
        logger.error("Number of groups must match number of campaigns")
        return

    # Create annotators list
    for i, camp_id in enumerate(args.campaign):
        if camp_id not in campaigns:
            logger.error(f"Campaign {camp_id} not found")
            return
        group = args.group[i]

        if group != "all":
            group = int(group)

        annotator_groups.append((camp_id, group))

    # Create save_plots directory if needed
    if args.save_plots:
        os.makedirs(args.save_plots, exist_ok=True)

    # Compute gamma scores
    gamma_results = compute_gamma(
        annotator_groups=annotator_groups,
        alpha=args.alpha,
        beta=args.beta,
        delta_empty=args.delta_empty,
        soft=args.soft,
        filter_datasets=args.filter_datasets,
        filter_splits=args.filter_splits,
        filter_example_ids=args.filter_example_ids,
        save_plots=args.save_plots,
    )

    if not gamma_results:
        print("No gamma results. Check that the campaigns and groups exist and share examples.")
        return

    # Print filtering information
    print("\nConfiguration:")
    print(f"  Alpha: {args.alpha}")
    print(f"  Beta: {args.beta}")
    print(f"  Delta empty: {args.delta_empty}")
    print(f"  Soft alignment: {args.soft}")
    print(f"  Datasets: {args.filter_datasets if args.filter_datasets else 'All'}")
    print(f"  Splits: {args.filter_splits if args.filter_splits else 'All'}")
    print(f"  Example IDs: {args.filter_example_ids if args.filter_example_ids else 'All'}")

    print(f"\nGroups:")
    for campaign_id, group in annotator_groups:
        if group == "all":
            print(f"  {campaign_id}: all groups")
        else:
            print(f"  {campaign_id}: {group}")

    print("\nResults:")
    print(
        f"  Gamma score: {round(gamma_results['gamma'], 3)} (computed on {len(gamma_results['gamma_scores'])}/{gamma_results['example_count']} examples)"
    )
    print(
        f"  S_empty score: {round(gamma_results['s_empty'], 3)} (computed on {len(gamma_results['s_empty_scores'])}/{gamma_results['example_count']} examples)"
    )

    # Save to JSON if output file specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(gamma_results, f, indent=2)
        logger.info(f"Gamma results saved to {args.output}")


if __name__ == "__main__":
    main()
