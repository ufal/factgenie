#!/usr/bin/env python3

import os
import pandas as pd
import sys
import logging
import traceback
import zipfile
import factgenie.workflows as workflows
import pygamma_agreement as pa
import numpy as np
from collections import defaultdict
from scipy.stats import pearsonr
from tqdm import tqdm
from pyannote.core import Segment

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

        # Drop the original annotations column
        span_index = span_index.drop("annotations", axis=1)

        # Remove any annotations that have NaN start or type or empty text
        span_index = span_index.dropna(subset=["annotation_start", "annotation_type", "annotation_text"])
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


def prepare_example_index(app, combinations, selected_campaigns, campaigns):
    # gather a list of all examples with some annotations
    example_index = pd.DataFrame()

    for campaign_id in selected_campaigns:
        campaign = campaigns[campaign_id]

        ei = generate_example_index(app, campaign)
        example_index = pd.concat([example_index, ei], ignore_index=True)

    # a combination is a tuple (dataset, split, setup_id)
    # leave only examples in example_index that are in the combinations selected by the user
    example_index = example_index[
        example_index.apply(lambda x: (x["dataset"], x["split"], x["setup_id"]) in combinations, axis=1)
    ]

    # add a column "annotator_group_id" to example_index, concatenating the campaign_id with str(annotator_group)
    # apply it separately for each row
    example_index["annotator_group_id"] = example_index.apply(
        lambda x: format_group_id(x["campaign_id"], str(x["annotator_group"])), axis=1
    )

    # group examples by dataset, split, setup_id, example_idx
    # aggregate annotations, annotator_ids, and counts for each category into a list
    aggregations = {"annotations": list, "annotator_group_id": list}
    cat_columns = [x for x in example_index.columns if x.startswith("cat_")]

    for c in cat_columns:
        aggregations[c] = list

    example_index = (
        example_index.groupby(["dataset", "split", "setup_id", "example_idx"]).agg(aggregations).reset_index()
    )
    return example_index


def compute_span_index(app, selected_campaigns, campaigns):
    span_index = []

    for campaign_id in selected_campaigns:
        df = generate_span_index(app, campaigns[campaign_id])

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
        "annotation_text",
    ]

    # Only drop columns that exist in the DataFrame
    existing_columns = [col for col in columns_to_drop if col in span_index.columns]
    span_index = span_index.drop(columns=existing_columns)

    return span_index


def compute_iaa_dfs(app, selected_campaigns, combinations, campaigns):
    example_index = prepare_example_index(
        app, combinations=combinations, selected_campaigns=selected_campaigns, campaigns=campaigns
    )
    dataset_level_counts, example_level_counts = compute_span_counts(
        example_index=example_index, combinations=combinations
    )

    span_index = compute_span_index(app, selected_campaigns, campaigns)

    results = {
        "dataset_level_counts": dataset_level_counts,
        "example_level_counts": example_level_counts,
        "span_index": span_index,
    }
    return results


def generate_iaa_files(app, selected_campaigns, combinations, campaigns, temp_dir):
    combinations = [(c["dataset"], c["split"], c["setup_id"]) for c in combinations]

    results = compute_iaa_dfs(app, selected_campaigns, combinations, campaigns)

    # Save each dataframe as CSV
    for name, df in results.items():
        csv_path = os.path.join(temp_dir, f"{name}.csv")

        # set precision of the `count` column to 3 decimal places
        if "count" in df.columns:
            df["count"] = df["count"].round(3)

        df.to_csv(csv_path, index=False)

    # Create ZIP file
    zip_path = os.path.join(temp_dir, "agreement_results.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for name in results.keys():
            csv_path = os.path.join(temp_dir, f"{name}.csv")
            zipf.write(csv_path, os.path.basename(csv_path))

    return zip_path


def format_group_id(campaign_id, group):
    """Format annotator group ID."""
    return f"{campaign_id}-anngroup-{group}"


# --------------------------------------------------------------
# the following methods are used only in CLI for now


def get_common_examples(first_campaign_data, second_campaign_data, first_group, second_group):
    """Find common examples between two annotator groups."""
    # Filter finished examples for first group
    first_examples = first_campaign_data[
        (first_campaign_data["annotator_group"] == first_group) & (first_campaign_data["status"] == "finished")
    ][["dataset", "split", "setup_id"]].drop_duplicates()

    # Filter finished examples for second group
    second_examples = second_campaign_data[
        (second_campaign_data["annotator_group"] == second_group) & (second_campaign_data["status"] == "finished")
    ][["dataset", "split", "setup_id"]].drop_duplicates()

    # Find intersection using merge
    common = pd.merge(first_examples, second_examples, how="inner")

    # Convert to list of tuples
    return list(map(tuple, common.values))


def compute_pearson_r(df, group1, group2):
    """Compute Pearson correlation between two annotator groups."""
    group1_data = df[df["annotator_group_id"] == group1]
    group2_data = df[df["annotator_group_id"] == group2]

    group1_counts = list(group1_data["count"])
    group2_counts = list(group2_data["count"])

    # Micro correlation - correlation of counts
    micro_corr = pearsonr(group1_counts, group2_counts)[0]

    # Macro correlation - average of per-type correlations
    type_corrs = []
    for ann_type in df["annotation_type"].unique():
        g1_type = list(group1_data[group1_data["annotation_type"] == ann_type]["count"])
        g2_type = list(group2_data[group2_data["annotation_type"] == ann_type]["count"])

        type_corrs.append(pearsonr(g1_type, g2_type)[0])

    return {"micro": micro_corr, "macro": np.mean(type_corrs), "category_correlations": type_corrs}


# `alpha`: coefficient weighting the *positional* dissimilarity value, defaults to 1
# `beta`: coefficient weighting the *categorical* dissimilarity value, defaults to 1
# `delta_empty`: empty dissimilarity value, defaults to 1
def compute_gamma_score(
    span_index, example_level_counts, alpha, beta, delta_empty, soft, save_plots, handle_empty_annotations
):
    dissim = pa.CombinedCategoricalDissimilarity(alpha=alpha, beta=beta, delta_empty=delta_empty)

    gamma_scores = []
    running_avg = 0

    # Group by same fields as in example_level_counts
    examples = example_level_counts.groupby(["dataset", "split", "setup_id", "example_idx"])

    # Create progress bar
    pbar = tqdm(total=len(examples), desc="Computing gamma score")

    if save_plots:
        import matplotlib.pyplot as plt

        ntb = pa.notebook.Notebook()
        os.makedirs(save_plots, exist_ok=True)

    for (dataset, split, setup_id, example_idx), example_group in examples:
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
            if handle_empty_annotations:
                # One or both annotators did not add any annotation
                # Compute score as 1 / (1 + annotation_cnt) to promote better matching in annotation count
                ann_count = example_spans.shape[0]
                aux_gamma_score = 1 / (1 + ann_count)
                gamma_scores.append(aux_gamma_score)
            else:
                # Skip this example
                logger.warning(
                    f"Skipping example {dataset}/{split}/{setup_id}/{example_idx} as it has less than 2 annotators. Consider using --gamma_handle_empty_annotations."
                )
                pbar.update(1)
                continue
        else:
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
                gamma_results = continuum.compute_gamma(dissim, soft=soft)
                gamma_scores.append(gamma_results.gamma)

                if save_plots:
                    fig, ax = plt.subplots(figsize=(10, 2))
                    ntb.plot_alignment(gamma_results.best_alignment, ax)
                    plt.tight_layout()

                    # Save the plot
                    plt.savefig(
                        os.path.join(save_plots, f"{dataset}_{split}_{setup_id}_{example_idx}.png"),
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close(fig)

            except Exception as e:
                traceback.print_exc()
                print(f"Error computing gamma for example {dataset}/{split}/{setup_id}/{example_idx}: {e}")
                gamma_scores.append(0.0)

            logging.getLogger().setLevel(logging.INFO)

        running_avg = np.mean(gamma_scores)
        pbar.set_postfix({"avg_gamma": f"{running_avg:.3f}"})
        pbar.update(1)

    pbar.close()
    return float(np.mean(gamma_scores)) if gamma_scores else 0.0


# --------------------------------------------------------------
