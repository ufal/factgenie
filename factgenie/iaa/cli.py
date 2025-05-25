"""Command line interface for inter-annotator agreement tools."""

import json
import logging

import click

from factgenie.iaa.f1 import compute_f1
from factgenie.iaa.gamma import compute_gamma
from factgenie.iaa.pearson import compute_pearson

logger = logging.getLogger(__name__)


@click.group("iaa")
def iaa_cli():
    """Tools for computing inter-annotator agreement."""
    pass


@iaa_cli.command("gamma")
@click.option(
    "--campaign",
    type=str,
    multiple=True,
    required=True,
    help="Campaign ID (can be specified multiple times)",
)
@click.option(
    "--group",
    type=int,
    multiple=True,
    help="Annotator group (either specified as many times as campaigns or omitted to use all groups)",
)
@click.option(
    "--alpha",
    type=float,
    default=1.0,
    help="Coefficient weighting the positional dissimilarity",
)
@click.option(
    "--beta",
    type=float,
    default=1.0,
    help="Coefficient weighting the categorical dissimilarity",
)
@click.option("--delta-empty", type=float, default=1.0, help="Empty dissimilarity value")
@click.option(
    "--soft_gamma",
    is_flag=True,
    help="Use soft version of gamma score (counting splitted segments as a single segment)",
)
@click.option(
    "--include-dataset",
    type=str,
    multiple=True,
    help="Only include the specified dataset (can be specified multiple times)",
)
@click.option(
    "--include-split",
    type=str,
    multiple=True,
    help="Only include the specified split (can be specified multiple times)",
)
@click.option(
    "--include-example-id",
    type=int,
    multiple=True,
    help="Only include the specified example ID (can be specified multiple times)",
)
@click.option("--save-plots", type=str, help="Directory to save alignment plots")
@click.option("--output", type=str, help="Output file for the gamma results (JSON format)")
def gamma_command(
    campaign,
    group,
    alpha,
    beta,
    delta_empty,
    soft_gamma,
    include_dataset,
    include_split,
    include_example_id,
    save_plots,
    output,
):
    """Compute gamma agreement scores between annotator groups."""
    if group and len(group) != len(campaign):
        logger.error("Number of groups must match number of campaigns")
        return

    if not group:
        group = [None] * len(campaign)

    # Compute gamma scores
    gamma_results = compute_gamma(
        campaign_ids=campaign,
        groups=group,
        alpha=alpha,
        beta=beta,
        delta_empty=delta_empty,
        soft_gamma=soft_gamma,
        include_dataset=include_dataset,
        include_split=include_split,
        include_example_id=include_example_id,
        save_plots=save_plots,
    )

    if not gamma_results:
        print("No gamma results. Check that the campaigns and groups exist and share examples.")
        return

    # Print filtering information
    print("\nConfiguration:")
    print(f"  Alpha: {alpha}")
    print(f"  Beta: {beta}")
    print(f"  Delta empty: {delta_empty}")
    print(f"  Soft alignment: {soft_gamma}")
    print(f"  Datasets: {include_dataset if include_dataset else 'All'}")
    print(f"  Splits: {include_split if include_split else 'All'}")
    print(f"  Example IDs: {include_example_id if include_example_id else 'All'}")

    print(f"\nGroups:")
    for campaign_id, group_val in zip(campaign, group):
        if group_val is None:
            print(f"  {campaign_id}: all groups")
        else:
            print(f"  {campaign_id}: {group_val}")

    print("\nResults:")
    print(
        f"  Gamma score: {round(gamma_results['gamma_mean'], 3)} (computed on {len(gamma_results['gamma_scores'])}/{gamma_results['example_count']} examples)"
    )
    print(
        f"  S_empty score: {round(gamma_results['s_empty_mean'], 3)} (computed on {len(gamma_results['s_empty_scores'])}/{gamma_results['example_count']} examples)"
    )

    # Save to JSON if output file specified
    if output:
        with open(output, "w") as f:
            json.dump(gamma_results, f, indent=2)
        logger.info(f"Gamma results saved to {output}")


@iaa_cli.command("f1")
@click.option(
    "--ref-campaign",
    type=str,
    required=True,
    help="Reference campaign ID",
)
@click.option(
    "--ref-group",
    type=int,
    help="Reference annotator group (if not specified, all groups are used)",
)
@click.option(
    "--hyp-campaign",
    type=str,
    required=True,
    help="Hypothesis campaign ID",
)
@click.option(
    "--hyp-group",
    type=int,
    help="Hypothesis annotator group (if not specified, all groups are used)",
)
@click.option(
    "--match-mode",
    type=str,
    default="hard",
    help="Match mode: 'hard' requires same category, 'soft' allows any category",
)
@click.option(
    "--category-breakdown",
    is_flag=True,
    help="Calculate metrics per annotation category",
)
@click.option(
    "--include-dataset",
    type=str,
    multiple=True,
    help="Only include the specified dataset (can be specified multiple times)",
)
@click.option(
    "--include-split",
    type=str,
    multiple=True,
    help="Only include the specified split (can be specified multiple times)",
)
@click.option(
    "--include-example-id",
    type=int,
    multiple=True,
    help="Only include the specified example ID (can be specified multiple times)",
)
@click.option(
    "--output",
    type=str,
    help="Output file for the F1 results (JSON format)",
)
def f1_command(
    ref_campaign,
    ref_group,
    hyp_campaign,
    hyp_group,
    match_mode,
    category_breakdown,
    include_dataset,
    include_split,
    include_example_id,
    output,
):
    """Compute precision, recall, and F1-score between reference and hypothesis annotator groups."""

    # Compute F1 scores
    f1_results = compute_f1(
        ref_camp_id=ref_campaign,
        ref_group=ref_group,
        hyp_camp_id=hyp_campaign,
        hyp_group=hyp_group,
        match_mode=match_mode,
        category_breakdown=category_breakdown,
        include_dataset=include_dataset,
        include_split=include_split,
        include_example_id=include_example_id,
    )

    if not f1_results:
        print(f"\nNo results found. Check that the campaigns and groups exist and share examples.")
        return

    # Print filtering information
    print("\nConfiguration:")
    print(f"  Match mode: {match_mode}")
    print(f"  Datasets: {include_dataset if include_dataset else 'All'}")
    print(f"  Splits: {include_split if include_split else 'All'}")
    print(f"  Example IDs: {include_example_id if include_example_id else 'All'}")

    print(f"\nGroups:")
    print(f"  Reference: {ref_campaign}: {ref_group if ref_group is not None else 'all groups'}")
    print(f"  Hypothesis: {hyp_campaign}: {hyp_group if hyp_group is not None else 'all groups'}")

    # Print results directly instead of using tabulate
    print("\nResults:")
    print(f"  Precision: {round(f1_results['precision'], 3)}")
    print(f"  Recall: {round(f1_results['recall'], 3)}")
    print(f"  F1: {round(f1_results['f1'], 3)}")
    print(f"  Ref Spans: {f1_results['ref_count']}")
    print(f"  Ref Len (Chars): {f1_results['ref_length']}")
    print(f"  Hyp Spans: {f1_results['hyp_count']}")
    print(f"  Hyp Len (Chars): {f1_results['hyp_length']}")
    print(f"  Overlap Len (Chars): {f1_results['overlap_length']}")

    # Print category breakdown if requested
    if category_breakdown:
        print(f"\nCategory breakdown:")
        for cat, cat_metrics in f1_results["categories"].items():
            print(f"  {cat}:")
            print(f"    Precision: {round(cat_metrics['precision'], 3)}")
            print(f"    Recall: {round(cat_metrics['recall'], 3)}")
            print(f"    F1: {round(cat_metrics['f1'], 3)}")

    # Save to JSON if output file specified
    if output:
        with open(output, "w") as f:
            json.dump(f1_results, f, indent=2)
        logger.info(f"F1 results saved to {output}")


@iaa_cli.command("pearson")
@click.option(
    "--campaign1",
    type=str,
    required=True,
    help="First campaign ID",
)
@click.option(
    "--group1",
    type=int,
    help="First annotator group (if not specified, all groups are used)",
)
@click.option(
    "--campaign2",
    type=str,
    required=True,
    help="Second campaign ID",
)
@click.option(
    "--group2",
    type=int,
    help="Second annotator group (if not specified, all groups are used)",
)
@click.option(
    "--include-dataset",
    type=str,
    multiple=True,
    help="Only include the specified dataset (can be specified multiple times)",
)
@click.option(
    "--include-split",
    type=str,
    multiple=True,
    help="Only include the specified split (can be specified multiple times)",
)
@click.option(
    "--include-example-id",
    type=int,
    multiple=True,
    help="Only include the specified example ID (can be specified multiple times)",
)
@click.option(
    "--output",
    type=str,
    help="Output file for the Pearson correlation results (JSON format)",
)
def pearson_command(
    campaign1,
    group1,
    campaign2,
    group2,
    include_dataset,
    include_split,
    include_example_id,
    output,
):
    """Compute Pearson correlation between error counts of two annotator groups."""
    # Compute Pearson correlation
    pearson_results = compute_pearson(
        campaign1=campaign1,
        group1=group1,
        campaign2=campaign2,
        group2=group2,
        include_dataset=include_dataset,
        include_split=include_split,
        include_example_id=include_example_id,
    )

    if not pearson_results:
        print("\nNo results found. Check that the campaigns and groups exist and share examples.")
        return

    # Print filtering information
    print("\nConfiguration:")
    print(f"  Datasets: {include_dataset if include_dataset else 'All'}")
    print(f"  Splits: {include_split if include_split else 'All'}")
    print(f"  Example IDs: {include_example_id if include_example_id else 'All'}")

    print(f"\nGroups:")
    print(f"  Reference: {campaign1}: {group1 if group1 is not None else 'all groups'}")
    print(f"  Hypothesis: {campaign2}: {group2 if group2 is not None else 'all groups'}")

    # Print results
    print("\nResults:")
    print(
        f"  Micro correlation: {round(pearson_results['micro_pearson'], 3) if pearson_results['micro_pearson'] is not None else 'N/A'}"
    )
    print(
        f"  Micro p-value: {round(pearson_results['micro_p_value'], 3) if pearson_results['micro_p_value'] is not None else 'N/A'}"
    )
    print(
        f"  Macro correlation: {round(pearson_results['macro_pearson'], 3) if pearson_results['macro_pearson'] is not None else 'N/A'}"
    )
    print(f"  Total examples: {pearson_results['example_count']}")

    print("\nCategory breakdown:")
    for cat, cat_metrics in pearson_results["categories"].items():
        print(f"  {cat}:")
        print(
            f"    Correlation: {round(cat_metrics['micro_pearson'], 3) if cat_metrics['micro_pearson'] is not None else 'N/A'}"
        )
        print(
            f"    P-value: {round(cat_metrics['micro_p_value'], 3) if cat_metrics['micro_p_value'] is not None else 'N/A'}"
        )

    # Save to JSON if output file specified
    if output:
        with open(output, "w") as f:
            json.dump(pearson_results, f, indent=2)
        logger.info(f"Pearson correlation results saved to {output}")
