from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

import factgenie.iaa.pearson as pearson_module
from factgenie.iaa.pearson import (
    _initialize_metrics,
    compute_pearson,
    compute_pearson_scores,
)


# Helper function to create span DataFrames for Pearson tests
def create_pearson_span_df(spans_data_list_of_dicts):
    """
    Creates a DataFrame from a list of span dictionaries for Pearson testing.
    Each dictionary in the list can represent one or more identical spans
    if 'num_spans' key is provided.
    Essential columns: dataset, split, setup_id, example_idx, campaign_id,
                       annotator_group, annotation_type.
    """
    if not spans_data_list_of_dicts:
        return pd.DataFrame(
            columns=[
                "dataset",
                "split",
                "setup_id",
                "example_idx",
                "campaign_id",
                "annotator_group",
                "annotation_type",
                "annotation_start",
                "annotation_end",  # Dummy columns
            ]
        )

    processed_spans = []
    for D_item in spans_data_list_of_dicts:
        # Make a copy to avoid modifying the original dict from the test
        D = D_item.copy()
        num_spans = D.pop("num_spans", 1)
        base_span = {
            "annotation_start": D.get("annotation_start", 0),  # Dummy value
            "annotation_end": D.get("annotation_end", 1),  # Dummy value
            **D,
        }
        for _ in range(num_spans):
            processed_spans.append(base_span.copy())

    df = pd.DataFrame(processed_spans)
    return df


def create_example_list_from_df(df_with_coords):
    """Creates an example list DataFrame from a DataFrame containing example coordinates."""
    if df_with_coords.empty or not all(
        col in df_with_coords.columns for col in ["dataset", "split", "setup_id", "example_idx"]
    ):
        return pd.DataFrame(columns=["dataset", "split", "setup_id", "example_idx"])
    return df_with_coords[["dataset", "split", "setup_id", "example_idx"]].drop_duplicates().reset_index(drop=True)


@pytest.fixture
def default_example_coords():
    """Provides a single default set of example coordinates."""
    return {"dataset": "ds1", "split": "train", "setup_id": "s1", "example_idx": 0}


@pytest.fixture
def multi_example_coords():
    """Provides coordinates for three distinct examples."""
    return [
        {"dataset": "ds1", "split": "train", "setup_id": "s1", "example_idx": 0},
        {"dataset": "ds1", "split": "train", "setup_id": "s1", "example_idx": 1},
        {"dataset": "ds1", "split": "train", "setup_id": "s1", "example_idx": 2},
    ]


@pytest.fixture
def annotator_groups_fixture():
    """Provides a default pair of annotator groups."""
    return [("c1", "g0"), ("c2", "g1")]


class TestInitializeMetrics:
    def test_initialize_no_categories(self):
        metrics = _initialize_metrics(categories=None)
        assert metrics["micro_pearson"] == 0.0
        assert "categories" in metrics
        assert not metrics["categories"]

    def test_initialize_with_categories(self):
        cats = ["A", "B"]
        metrics = _initialize_metrics(categories=cats)
        assert "A" in metrics["categories"]
        assert "B" in metrics["categories"]
        assert metrics["categories"]["A"]["micro_pearson"] == 0.0
        assert "categories" in metrics["categories"]["A"]
        assert not metrics["categories"]["A"]["categories"]


class TestComputePearsonScores:

    def test_perfect_positive_correlation_single_category(self, multi_example_coords, annotator_groups_fixture):
        cat = "A"
        spans_data = [
            {
                **multi_example_coords[0],
                "campaign_id": "c1",
                "annotator_group": "g0",
                "annotation_type": cat,
                "num_spans": 1,
            },
            {
                **multi_example_coords[0],
                "campaign_id": "c2",
                "annotator_group": "g1",
                "annotation_type": cat,
                "num_spans": 1,
            },
            {
                **multi_example_coords[1],
                "campaign_id": "c1",
                "annotator_group": "g0",
                "annotation_type": cat,
                "num_spans": 2,
            },
            {
                **multi_example_coords[1],
                "campaign_id": "c2",
                "annotator_group": "g1",
                "annotation_type": cat,
                "num_spans": 2,
            },
            {
                **multi_example_coords[2],
                "campaign_id": "c1",
                "annotator_group": "g0",
                "annotation_type": cat,
                "num_spans": 3,
            },
            {
                **multi_example_coords[2],
                "campaign_id": "c2",
                "annotator_group": "g1",
                "annotation_type": cat,
                "num_spans": 3,
            },
        ]
        span_index = create_pearson_span_df(spans_data)
        example_list = create_example_list_from_df(span_index)
        metrics = compute_pearson_scores(span_index, example_list, annotator_groups_fixture)

        assert metrics["micro_pearson"] == pytest.approx(1.0)
        assert metrics["macro_pearson"] == pytest.approx(1.0)
        assert metrics["categories"][cat]["micro_pearson"] == pytest.approx(1.0)
        assert metrics["example_count"] == 3

    def test_perfect_negative_correlation_single_category(self, multi_example_coords, annotator_groups_fixture):
        cat = "A"
        spans_data = [
            {
                **multi_example_coords[0],
                "campaign_id": "c1",
                "annotator_group": "g0",
                "annotation_type": cat,
                "num_spans": 1,
            },
            {
                **multi_example_coords[0],
                "campaign_id": "c2",
                "annotator_group": "g1",
                "annotation_type": cat,
                "num_spans": 3,
            },
            {
                **multi_example_coords[1],
                "campaign_id": "c1",
                "annotator_group": "g0",
                "annotation_type": cat,
                "num_spans": 2,
            },
            {
                **multi_example_coords[1],
                "campaign_id": "c2",
                "annotator_group": "g1",
                "annotation_type": cat,
                "num_spans": 2,
            },
            {
                **multi_example_coords[2],
                "campaign_id": "c1",
                "annotator_group": "g0",
                "annotation_type": cat,
                "num_spans": 3,
            },
            {
                **multi_example_coords[2],
                "campaign_id": "c2",
                "annotator_group": "g1",
                "annotation_type": cat,
                "num_spans": 1,
            },
        ]
        span_index = create_pearson_span_df(spans_data)
        example_list = create_example_list_from_df(span_index)
        metrics = compute_pearson_scores(span_index, example_list, annotator_groups_fixture)

        assert metrics["micro_pearson"] == pytest.approx(-1.0)
        assert metrics["macro_pearson"] == pytest.approx(-1.0)
        assert metrics["categories"][cat]["micro_pearson"] == pytest.approx(-1.0)

    def test_zero_variance_one_group_nan_correlation(self, multi_example_coords, annotator_groups_fixture):
        cat = "A"
        spans_data = [
            {
                **multi_example_coords[0],
                "campaign_id": "c1",
                "annotator_group": "g0",
                "annotation_type": cat,
                "num_spans": 2,
            },  # Group 0 is constant
            {
                **multi_example_coords[0],
                "campaign_id": "c2",
                "annotator_group": "g1",
                "annotation_type": cat,
                "num_spans": 1,
            },
            {
                **multi_example_coords[1],
                "campaign_id": "c1",
                "annotator_group": "g0",
                "annotation_type": cat,
                "num_spans": 2,
            },
            {
                **multi_example_coords[1],
                "campaign_id": "c2",
                "annotator_group": "g1",
                "annotation_type": cat,
                "num_spans": 2,
            },
            {
                **multi_example_coords[2],
                "campaign_id": "c1",
                "annotator_group": "g0",
                "annotation_type": cat,
                "num_spans": 2,
            },
            {
                **multi_example_coords[2],
                "campaign_id": "c2",
                "annotator_group": "g1",
                "annotation_type": cat,
                "num_spans": 3,
            },
        ]
        span_index = create_pearson_span_df(spans_data)
        example_list = create_example_list_from_df(span_index)
        metrics = compute_pearson_scores(span_index, example_list, annotator_groups_fixture)

        assert np.isnan(metrics["micro_pearson"])
        assert np.isnan(metrics["macro_pearson"])
        assert np.isnan(metrics["categories"][cat]["micro_pearson"])

    def test_multiple_categories_mixed_correlation(self, multi_example_coords, annotator_groups_fixture):
        # Cat A: perfect positive. Cat B: perfect negative.
        spans_data = []
        ex = multi_example_coords
        # Ex0: A(1,1), B(3,1)
        spans_data.extend(
            [
                {**ex[0], "campaign_id": "c1", "annotator_group": "g0", "annotation_type": "A", "num_spans": 1},
                {**ex[0], "campaign_id": "c2", "annotator_group": "g1", "annotation_type": "A", "num_spans": 1},
                {**ex[0], "campaign_id": "c1", "annotator_group": "g0", "annotation_type": "B", "num_spans": 3},
                {**ex[0], "campaign_id": "c2", "annotator_group": "g1", "annotation_type": "B", "num_spans": 1},
            ]
        )
        # Ex1: A(2,2), B(2,2)
        spans_data.extend(
            [
                {**ex[1], "campaign_id": "c1", "annotator_group": "g0", "annotation_type": "A", "num_spans": 2},
                {**ex[1], "campaign_id": "c2", "annotator_group": "g1", "annotation_type": "A", "num_spans": 2},
                {**ex[1], "campaign_id": "c1", "annotator_group": "g0", "annotation_type": "B", "num_spans": 2},
                {**ex[1], "campaign_id": "c2", "annotator_group": "g1", "annotation_type": "B", "num_spans": 2},
            ]
        )
        # Ex2: A(3,3), B(1,3)
        spans_data.extend(
            [
                {**ex[2], "campaign_id": "c1", "annotator_group": "g0", "annotation_type": "A", "num_spans": 3},
                {**ex[2], "campaign_id": "c2", "annotator_group": "g1", "annotation_type": "A", "num_spans": 3},
                {**ex[2], "campaign_id": "c1", "annotator_group": "g0", "annotation_type": "B", "num_spans": 1},
                {**ex[2], "campaign_id": "c2", "annotator_group": "g1", "annotation_type": "B", "num_spans": 3},
            ]
        )

        span_index = create_pearson_span_df(spans_data)
        example_list = create_example_list_from_df(span_index)
        metrics = compute_pearson_scores(span_index, example_list, annotator_groups_fixture)

        # Micro: G0_counts (A0,B0,A1,B1,A2,B2) = [1,3,2,2,3,1], G1_counts = [1,1,2,2,3,3]
        g0_micro = [1, 3, 2, 2, 3, 1]
        g1_micro = [1, 1, 2, 2, 3, 3]
        expected_micro_r, _ = pearsonr(g0_micro, g1_micro)
        assert metrics["micro_pearson"] == pytest.approx(expected_micro_r)

        # Cat A: G0=[1,2,3], G1=[1,2,3] -> r=1.0
        assert metrics["categories"]["A"]["micro_pearson"] == pytest.approx(1.0)
        # Cat B: G0=[3,2,1], G1=[1,2,3] -> r=-1.0
        assert metrics["categories"]["B"]["micro_pearson"] == pytest.approx(-1.0)
        # Macro: mean(1.0, -1.0) = 0.0
        assert metrics["macro_pearson"] == pytest.approx(0.0)
        assert metrics["example_count"] == 3


@patch("factgenie.iaa.pearson.app", MagicMock())
@patch("factgenie.workflows.generate_campaign_index")
@patch("factgenie.analysis.assert_common_categories")
@patch("factgenie.analysis.compute_span_index")
@patch("factgenie.analysis.get_example_list")
@patch("factgenie.iaa.pearson.compute_pearson_scores")
class TestComputePearson:

    def _setup_common_mocks(
        self,
        mock_cps,
        mock_gel,
        mock_csi,
        mock_acc,
        mock_gci,
        campaign_index_val=None,
        assert_common_val=True,
        span_index_val=None,
        example_list_val=None,
        cps_return_val=None,
    ):
        mock_gci.return_value = campaign_index_val if campaign_index_val is not None else {}
        mock_acc.return_value = assert_common_val
        mock_csi.return_value = span_index_val if span_index_val is not None else create_pearson_span_df([])
        mock_gel.return_value = (
            example_list_val if example_list_val is not None else create_example_list_from_df(pd.DataFrame())
        )
        # Ensure cps_return_val is a complete dict as expected from _initialize_metrics
        if cps_return_val is None:
            cps_return_val = _initialize_metrics(categories=None)  # Default empty metrics
        mock_cps.return_value = cps_return_val

    def test_successful_flow(self, mock_cps, mock_gel, mock_csi, mock_acc, mock_gci, default_example_coords):
        c1, g1 = "campA", "groupX"
        c2, g2 = "campB", "groupY"

        mock_campaign_index = {"campA": MagicMock(), "campB": MagicMock()}
        # Dummy non-empty DataFrames for span_index and example_list
        mock_span_index = create_pearson_span_df([default_example_coords])
        mock_example_list = create_example_list_from_df(pd.DataFrame([default_example_coords]))
        # Construct a full expected result from compute_pearson_scores
        expected_cps_result = _initialize_metrics(categories=None)
        expected_cps_result.update({"micro_pearson": 0.5, "example_count": 1})

        self._setup_common_mocks(
            mock_cps,
            mock_gel,
            mock_csi,
            mock_acc,
            mock_gci,
            campaign_index_val=mock_campaign_index,
            span_index_val=mock_span_index,
            example_list_val=mock_example_list,
            cps_return_val=expected_cps_result,
        )

        result = compute_pearson(c1, g1, c2, g2, ["ds1"], ["train"], [0])

        assert result == expected_cps_result
        mock_gci.assert_called_once_with(pearson_module.app, force_reload=True)
        mock_acc.assert_called_once_with([c1, c2], mock_campaign_index)

        expected_annotator_groups_arg = [(c1, g1), (c2, g2)]
        mock_csi.assert_called_once_with(pearson_module.app, [c1, c2], mock_campaign_index)
        mock_gel.assert_called_once_with(mock_campaign_index, expected_annotator_groups_arg, ["ds1"], ["train"], [0])

        mock_cps.assert_called_once_with(
            span_index=mock_span_index, example_list=mock_example_list, annotator_groups=expected_annotator_groups_arg
        )

    def test_assert_common_categories_false_returns_none(self, mock_cps, mock_gel, mock_csi, mock_acc, mock_gci):
        self._setup_common_mocks(mock_cps, mock_gel, mock_csi, mock_acc, mock_gci, assert_common_val=False)

        result = compute_pearson("c1", "g1", "c2", "g2")
        assert result is None
        mock_csi.assert_not_called()
        mock_gel.assert_not_called()
        mock_cps.assert_not_called()

    def test_group_none_propagated_correctly(self, mock_cps, mock_gel, mock_csi, mock_acc, mock_gci):
        self._setup_common_mocks(mock_cps, mock_gel, mock_csi, mock_acc, mock_gci)
        compute_pearson("c1", None, "c2", None)

        expected_annotator_groups = [("c1", None), ("c2", None)]

        # Check get_example_list call
        mock_gel.assert_called_once()
        assert (
            mock_gel.call_args[0][1] == expected_annotator_groups
        )  # annotator_groups is the 2nd arg to get_example_list

        # Check compute_pearson_scores call
        mock_cps.assert_called_once()
        assert mock_cps.call_args[1]["annotator_groups"] == expected_annotator_groups  # Passed as kwarg
