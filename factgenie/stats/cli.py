"""Command line interface for campaign statistics tools."""

import json
import logging

import click
import numpy as np
import pandas as pd
from tabulate import tabulate

from factgenie.analysis import format_group_id
from factgenie.stats.confusion import compute_confusion_matrix, plot_confusion_matrix
from factgenie.stats.stats import compute_stats

logger = logging.getLogger(__name__)


@click.group("stats")
def stats_cli():
    """Tools for computing campaign statistics."""
    pass


@stats_cli.command("counts")
@click.option(
    "--campaign",
    type=str,
    required=True,
    help="Campaign ID",
)
@click.option(
    "--annotator-group",
    type=int,
    help="Annotator group ID (optional, filters stats for this group)",
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
    "--output",
    type=click.Path(dir_okay=False, writable=True),
    help="Output file for the statistics (JSON format)",
)
def counts_command(
    campaign,
    annotator_group,
    include_dataset,
    include_split,
    output,
):
    """Compute annotation counts and other basic statistics for a campaign."""

    # Convert empty tuples from click multiple=True to None for cleaner API
    datasets_filter = list(include_dataset) if include_dataset else None
    splits_filter = list(include_split) if include_split else None

    stats = compute_stats(
        campaign_id=campaign,
        annotator_group=annotator_group,
        include_dataset=datasets_filter,
        include_split=splits_filter,
    )

    if not stats:
        print(f"Could not compute statistics for campaign {campaign}. Check logs for details.")
        return

    print(f"\n--- Statistics for Campaign: {stats.get('campaign_id', campaign)} ---")
    if stats.get("annotator_group") is not None:
        print(f"  Annotator Group: {stats['annotator_group']}")
    if stats.get("include_dataset"):
        print(f"  Included Datasets: {', '.join(stats['include_dataset'])}")
    if stats.get("include_split"):
        print(f"  Included Splits: {', '.join(stats['include_split'])}")

    print("\nOverall Counts:")
    print(f"  Total examples considered: {stats['total_examples']}")
    print(f"  Examples with annotations: {stats['examples_with_annotations']}")
    print(f"  Examples without annotations: {stats['examples_without_annotations']}")
    print(f"  Percentage of empty examples: {stats['empty_examples_percentage']}%")
    print(f"  Total annotations: {stats['total_annotations']}")
    print(f"  Annotations per example (avg): {stats['annotations_per_example']}")
    print(f"  Average annotation length (chars): {stats['avg_annotation_length']}")

    if output:
        try:
            with open(output, "w") as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Statistics saved to {output}")
            print(f"\nStatistics saved to {output}")
        except Exception as e:
            logger.error(f"Failed to save statistics to {output}: {e}")
            print(f"\nFailed to save statistics to {output}: {e}")


@stats_cli.command("confusion")
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
@click.option("--normalize", is_flag=True, help="Normalize confusion matrix by row (reference).")
@click.option(
    "--output", type=click.Path(dir_okay=False, writable=True), help="Output CSV file for the confusion matrix."
)
@click.option(
    "--output-plot",
    type=click.Path(dir_okay=False, writable=True),
    help="Output file for confusion matrix plot (optional).",
)
def confusion_command(
    ref_campaign,
    ref_group,
    hyp_campaign,
    hyp_group,
    include_dataset,
    include_split,
    include_example_id,
    normalize,
    output,
    output_plot,
):
    """Compute a confusion matrix between annotations from two groups."""
    datasets_filter = list(include_dataset) if include_dataset else None
    splits_filter = list(include_split) if include_split else None
    example_ids_filter = list(include_example_id) if include_example_id else None

    result = compute_confusion_matrix(
        ref_camp_id=ref_campaign,
        ref_group=ref_group,
        hyp_camp_id=hyp_campaign,
        hyp_group=hyp_group,
        include_dataset=datasets_filter,
        include_split=splits_filter,
        include_example_id=example_ids_filter,
    )

    if not result:
        print(f"Could not compute confusion matrix for the specified groups. Check logs for details.")
        return

    conf_mat_np = np.array(result["confusion_matrix"])

    labels = result["labeled_categories"]

    if conf_mat_np.size == 0:
        print("No data to compute confusion matrix (e.g., no overlapping annotations or categories).")
        return

    df = pd.DataFrame(conf_mat_np, index=labels, columns=labels)

    print(f"\n--- Confusion Matrix ---")
    display_df = df
    if normalize:
        row_sums = df.sum(axis=1)
        # Avoid division by zero for display
        safe_row_sums = row_sums.replace(0, 1)
        display_df = df.div(safe_row_sums, axis=0).round(3)
        print("\nNormalized Confusion Matrix (by row):")
    else:
        print("\nConfusion Matrix (Counts):")

    if not display_df.empty:
        print(tabulate(display_df, headers="keys", tablefmt="grid"))
        print("\nReference counts (sum over rows):")
        print(df.sum(axis=1))
        print("\nHypothesis counts (sum over columns):")
        print(df.sum(axis=0))
    else:
        print("Matrix is empty.")

    if output:
        df.to_csv(output)
        logger.info(f"Confusion matrix (counts) saved to {output}")

    if output_plot:
        try:
            plot_confusion_matrix(
                conf_mat_np,
                labels=labels,
                normalize=normalize,
                output_file=output_plot,
                ref_group_id=format_group_id(ref_campaign, ref_group),
                hyp_group_id=format_group_id(hyp_campaign, hyp_group),
            )
        except Exception as e:
            logger.error(f"Failed to save confusion matrix plot: {e}")
            print(f"Failed to save confusion matrix plot: {e}")
