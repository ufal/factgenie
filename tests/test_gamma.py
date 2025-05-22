import sys
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

import factgenie.iaa.gamma as gamma_module  # For mocking elements within the gamma module
from factgenie.analysis import format_group_id
from factgenie.iaa.gamma import (
    compute_gamma_score,  # The inner function that calls pygamma
)
from factgenie.iaa.gamma import (
    _initialize_gamma_metrics,
    _process_example_gamma,
    compute_gamma,
    compute_gamma_scores,
    compute_s_empty_score,
    save_plot,
)


# Helper function to create span DataFrames
def create_gamma_span_df(spans_data):
    """
    Creates a DataFrame from a list of span dictionaries for gamma testing.
    Essential columns: dataset, split, setup_id, example_idx, campaign_id,
                       annotator_group, annotation_start, annotation_end, annotation_type.
    """
    if not spans_data:
        return pd.DataFrame(
            columns=[
                "dataset",
                "split",
                "setup_id",
                "example_idx",
                "campaign_id",
                "annotator_group",
                "annotation_start",
                "annotation_end",
                "annotation_type",
                "annotation_text",
                "annotator_group_id",  # Include for direct use in some tests
            ]
        )
    df = pd.DataFrame(spans_data)
    if "annotation_text" not in df.columns and "annotation_start" in df.columns and "annotation_end" in df.columns:
        df["annotation_text"] = df.apply(lambda r: "x" * (r["annotation_end"] - r["annotation_start"]), axis=1)

    # Ensure annotator_group_id is present if campaign_id and annotator_group are,
    # as this is what compute_gamma_score expects in its input `example_spans`.
    # _get_camp_group_spans normally creates this.
    if "campaign_id" in df.columns and "annotator_group" in df.columns and "annotator_group_id" not in df.columns:
        df["annotator_group_id"] = df.apply(
            lambda row: format_group_id(row["campaign_id"], row["annotator_group"]), axis=1
        )
    return df


def create_example_list_from_spans(span_df):
    if span_df.empty:
        return pd.DataFrame(columns=["dataset", "split", "setup_id", "example_idx"])
    return span_df[["dataset", "split", "setup_id", "example_idx"]].drop_duplicates().reset_index(drop=True)


@pytest.fixture
def default_example_coords():
    return {"dataset": "ds1", "split": "train", "setup_id": "s1", "example_idx": 0}


@pytest.fixture
def mock_pygamma_core_deps():
    mock_pa = MagicMock()
    mock_Segment = MagicMock()

    # Set up the Continuum and its gamma computation
    mock_continuum_instance = MagicMock()
    mock_pa.Continuum.return_value = mock_continuum_instance

    mock_gamma_results = MagicMock()
    mock_gamma_results.gamma = 0.8  # Default successful gamma
    mock_gamma_results.best_alignment = MagicMock(name="BestAlignmentObj")
    mock_continuum_instance.compute_gamma.return_value = mock_gamma_results

    # Set up other required attributes on pa
    mock_dissim_instance = MagicMock(name="DissimObj")
    mock_pa.CombinedCategoricalDissimilarity.return_value = mock_dissim_instance

    mock_ntb_instance = MagicMock(name="NotebookObj")
    mock_pa.notebook = MagicMock()
    mock_pa.notebook.Notebook.return_value = mock_ntb_instance

    yield mock_pa, mock_Segment, mock_continuum_instance, mock_dissim_instance, mock_ntb_instance


class TestComputeSEmptyScore:
    def test_one_annotation(self, default_example_coords):
        spans = create_gamma_span_df([{**default_example_coords, "annotation_start": 0, "annotation_end": 1}])
        assert compute_s_empty_score(spans) == 0.5

    def test_no_annotations(self):
        spans = create_gamma_span_df([])
        assert compute_s_empty_score(spans) == 1.0

    def test_three_annotations(self, default_example_coords):
        spans = create_gamma_span_df(
            [
                {**default_example_coords, "annotation_start": 0, "annotation_end": 1},
                {**default_example_coords, "annotation_start": 1, "annotation_end": 2},
                {**default_example_coords, "annotation_start": 2, "annotation_end": 3},
            ]
        )
        assert compute_s_empty_score(spans) == 0.25


class TestUnitComputeGammaScore:  # Tests the inner compute_gamma_score

    def test_successful_gamma_computation(self, mock_pygamma_core_deps, default_example_coords):
        mock_pa, mock_Segment, mock_continuum, _, _ = mock_pygamma_core_deps
        mock_gamma_results = MagicMock()
        mock_gamma_results.gamma = 0.85
        mock_continuum.compute_gamma.return_value = mock_gamma_results

        example_spans_data = [
            {
                **default_example_coords,
                "campaign_id": "c1",
                "annotator_group": "g1",
                "annotation_start": 0,
                "annotation_end": 5,
                "annotation_type": "A",
            },
            {
                **default_example_coords,
                "campaign_id": "c1",
                "annotator_group": "g2",
                "annotation_start": 0,
                "annotation_end": 5,
                "annotation_type": "A",
            },
        ]
        # This df needs annotator_group_id as it's what compute_gamma_score receives
        example_spans_df = create_gamma_span_df(example_spans_data)

        dissim_mock = MagicMock()
        with patch("factgenie.iaa.gamma.np.random.seed") as mock_seed:
            gamma_val = compute_gamma_score(
                example_spans_df, dissim_mock, soft_gamma=True, pa=mock_pa, Segment=mock_Segment, example_id="ex1"
            )
            mock_seed.assert_called_once_with(42)

        assert gamma_val == 0.85
        mock_pa.Continuum.assert_called_once()
        expected_add_calls = [
            call(format_group_id("c1", "g1"), mock_Segment(0, 5), "A"),
            call(format_group_id("c1", "g2"), mock_Segment(0, 5), "A"),
        ]
        mock_continuum.add.assert_has_calls(expected_add_calls, any_order=True)
        mock_continuum.compute_gamma.assert_called_once_with(dissim_mock, soft=True)

    def test_gamma_computation_exception(self, mock_pygamma_core_deps, default_example_coords):
        mock_pa, mock_Segment, mock_continuum, _, _ = mock_pygamma_core_deps
        mock_continuum.compute_gamma.side_effect = Exception("pygamma internal error")

        example_spans_df = create_gamma_span_df(
            [
                {
                    **default_example_coords,
                    "campaign_id": "c1",
                    "annotator_group": "g1",
                    "annotation_start": 0,
                    "annotation_end": 5,
                    "annotation_type": "A",
                }
            ]
        )
        dissim_mock = MagicMock()

        with patch.object(gamma_module.logger, "error") as mock_log_error:
            gamma_val = compute_gamma_score(
                example_spans_df, dissim_mock, soft_gamma=False, pa=mock_pa, Segment=mock_Segment, example_id="ex_error"
            )
        assert gamma_val == 0.0
        assert mock_log_error.call_count >= 1  # Error and traceback
        assert "pygamma internal error" in mock_log_error.call_args_list[0][0][0]
        assert "ex_error" in mock_log_error.call_args_list[0][0][0]

    def test_perfect_match_gamma_is_one(self, mock_pygamma_core_deps, default_example_coords):
        mock_pa, mock_Segment, mock_continuum, _, _ = mock_pygamma_core_deps
        # Configure the mock continuum to return gamma = 1.0 for the perfect match
        mock_gamma_results = MagicMock()
        mock_gamma_results.gamma = 1.0  # Perfect match
        mock_continuum.compute_gamma.return_value = mock_gamma_results

        # Define perfectly matching annotations from two annotators
        example_spans_data = [
            {
                **default_example_coords,
                "campaign_id": "c1",
                "annotator_group": "g1",
                "annotation_start": 10,
                "annotation_end": 20,
                "annotation_type": "MATCH_TYPE",
            },
            {
                **default_example_coords,
                "campaign_id": "c1",
                "annotator_group": "g2",
                "annotation_start": 10,  # Identical start
                "annotation_end": 20,  # Identical end
                "annotation_type": "MATCH_TYPE",  # Identical type
            },
        ]
        # This df needs annotator_group_id as it's what compute_gamma_score receives
        example_spans_df = create_gamma_span_df(example_spans_data)

        dissim_mock = MagicMock()
        example_id_str = "ex_perfect_match"

        with patch("factgenie.iaa.gamma.np.random.seed") as mock_seed:
            gamma_val = compute_gamma_score(
                example_spans_df,
                dissim_mock,
                soft_gamma=True,
                pa=mock_pa,
                Segment=mock_Segment,
                example_id=example_id_str,
            )
            mock_seed.assert_called_once_with(42)

        assert gamma_val == 1.0
        mock_pa.Continuum.assert_called_once()  # Continuum is created once

        # Check that continuum.add was called correctly for each annotation
        expected_add_calls = [
            call(format_group_id("c1", "g1"), mock_Segment(10, 20), "MATCH_TYPE"),
            call(format_group_id("c1", "g2"), mock_Segment(10, 20), "MATCH_TYPE"),
        ]
        mock_continuum.add.assert_has_calls(expected_add_calls, any_order=True)
        assert mock_continuum.add.call_count == 2

        # Check that compute_gamma was called with the correct dissimilarity object and soft_gamma setting
        mock_continuum.compute_gamma.assert_called_once_with(dissim_mock, soft=True)

    @patch("factgenie.iaa.gamma.compute_gamma_score", side_effect=compute_gamma_score)  # Use the real function
    def test_real_perfect_match_gamma(self, mock_compute_gamma, default_example_coords):
        """Test that perfectly matching annotations between two annotators result in gamma = 1.0"""
        # Define perfectly matching annotations from two annotators
        example_spans_data = [
            {
                **default_example_coords,
                "campaign_id": "c1",
                "annotator_group": "g1",
                "annotation_start": 10,
                "annotation_end": 20,
                "annotation_type": "MATCH_TYPE",
            },
            {
                **default_example_coords,
                "campaign_id": "c1",
                "annotator_group": "g2",
                "annotation_start": 10,  # Identical start
                "annotation_end": 20,  # Identical end
                "annotation_type": "MATCH_TYPE",  # Identical type
            },
        ]
        example_spans_df = create_gamma_span_df(example_spans_data)

        # Create actual example tuple for _process_example_gamma
        example_tuple = (
            default_example_coords["dataset"],
            default_example_coords["split"],
            default_example_coords["setup_id"],
            default_example_coords["example_idx"],
        )

        # Test with real pygamma-agreement
        try:
            import pygamma_agreement as pa
            from pyannote.core import Segment

            # Create a real dissimilarity object
            dissim = pa.CombinedCategoricalDissimilarity(alpha=1.0, beta=1.0, delta_empty=1.0)

            # Process the example with real implementation
            gamma_val, _ = _process_example_gamma(
                example_tuple,
                example_spans_df,
                [("c1", "g1"), ("c1", "g2")],
                dissim,
                soft_gamma=True,
                pa=pa,
                Segment=Segment,
            )

            # Check that gamma is exactly 1.0 for perfect match
            assert gamma_val == 1.0, f"Expected gamma=1.0 for perfect match, got {gamma_val}"

        except ImportError:
            pytest.skip("pygamma-agreement or pyannote.core not installed")


@patch("factgenie.iaa.gamma.compute_s_empty_score", wraps=compute_s_empty_score)  # Wrap to use real one but still track
@patch("factgenie.iaa.gamma.compute_gamma_score")  # Mock the inner one
@patch("factgenie.iaa.gamma._get_camp_group_spans")
class TestProcessExampleGamma:

    def test_two_plus_annotators_calls_compute_gamma_score(
        self,
        mock_get_camp_spans,
        mock_unit_compute_gamma,
        mock_unit_compute_s_empty,
        default_example_coords,
        mock_pygamma_core_deps,
    ):
        mock_pa_mod, mock_segment_cls, _, _, _ = mock_pygamma_core_deps
        example_tuple = (
            default_example_coords["dataset"],
            default_example_coords["split"],
            default_example_coords["setup_id"],
            default_example_coords["example_idx"],
        )

        # _get_camp_group_spans returns df with 'annotator_group_id' and other needed cols
        processed_spans_data = [
            {"annotator_group_id": "c1_g1", "annotation_start": 1, "annotation_end": 2, "annotation_type": "T1"},
            {"annotator_group_id": "c1_g2", "annotation_start": 3, "annotation_end": 4, "annotation_type": "T1"},
        ]
        mock_processed_spans_df = pd.DataFrame(processed_spans_data)
        mock_get_camp_spans.return_value = mock_processed_spans_df
        mock_unit_compute_gamma.return_value = 0.75

        span_index_df = create_gamma_span_df([])  # Content doesn't matter due to mock_get_camp_spans
        annotator_groups = [("c1", "g1"), ("c1", "g2")]
        dissim_mock = MagicMock()

        gamma_val, s_empty_val = _process_example_gamma(
            example_tuple,
            span_index_df,
            annotator_groups,
            dissim_mock,
            soft_gamma=True,
            pa=mock_pa_mod,
            Segment=mock_segment_cls,
            save_plots_config=None,
        )

        assert mock_get_camp_spans.call_count == 1
        args_gcs, _ = mock_get_camp_spans.call_args
        pd.testing.assert_frame_equal(args_gcs[0], span_index_df)
        assert args_gcs[1] == annotator_groups

        assert mock_unit_compute_gamma.call_count == 1  # Check call count directly

        args_cg, kwargs_cg = mock_unit_compute_gamma.call_args

        # Check positional arguments
        # All arguments to compute_gamma_score are passed as keyword arguments
        # in _process_example_gamma, so args_cg should be empty.
        assert not args_cg  # No positional arguments expected

        # Check keyword arguments
        pd.testing.assert_frame_equal(kwargs_cg["example_spans"], mock_processed_spans_df)
        assert kwargs_cg["dissim"] is dissim_mock
        assert kwargs_cg["soft_gamma"] is True
        assert kwargs_cg["example_id"] == f"ds1_train_s1_{0}"
        assert kwargs_cg["pa"] is mock_pa_mod
        assert kwargs_cg["Segment"] is mock_segment_cls
        assert kwargs_cg["save_plots_config"] is None

        assert gamma_val == 0.75
        assert np.isnan(s_empty_val)
        mock_unit_compute_s_empty.assert_not_called()

    def test_less_than_two_annotators_calls_compute_s_empty_score(
        self,
        mock_get_camp_spans,
        mock_unit_compute_gamma,
        mock_unit_compute_s_empty,
        default_example_coords,
        mock_pygamma_core_deps,
    ):
        mock_pa_mod, mock_segment_cls, _, _, _ = mock_pygamma_core_deps
        example_tuple = (
            default_example_coords["dataset"],
            default_example_coords["split"],
            default_example_coords["setup_id"],
            default_example_coords["example_idx"],
        )

        processed_spans_data = [
            {"annotator_group_id": "c1_g1", "annotation_start": 1, "annotation_end": 2, "annotation_type": "T1"}
        ]
        mock_processed_spans_df = pd.DataFrame(processed_spans_data)
        mock_get_camp_spans.return_value = mock_processed_spans_df
        # mock_unit_compute_s_empty is wrapped, so it will calculate 1/(1+1) = 0.5

        span_index_df = create_gamma_span_df([])
        annotator_groups = [("c1", "g1")]
        dissim_mock = MagicMock()

        gamma_val, s_empty_val = _process_example_gamma(
            example_tuple,
            span_index_df,
            annotator_groups,
            dissim_mock,
            soft_gamma=True,
            pa=mock_pa_mod,
            Segment=mock_segment_cls,
        )

        mock_unit_compute_gamma.assert_not_called()

        # Check call to compute_s_empty_score manually
        assert mock_unit_compute_s_empty.call_count == 1
        # The first argument to compute_s_empty_score is example_spans (a DataFrame)
        called_args, _ = mock_unit_compute_s_empty.call_args
        pd.testing.assert_frame_equal(called_args[0], mock_processed_spans_df)

        assert np.isnan(gamma_val)
        assert s_empty_val == 0.5  # 1 / (1 + 1 annotation)

    def test_filters_empty_segments(
        self,
        mock_get_camp_spans,
        mock_unit_compute_gamma,
        mock_unit_compute_s_empty,
        default_example_coords,
        mock_pygamma_core_deps,
    ):
        mock_pa_mod, mock_segment_cls, _, _, _ = mock_pygamma_core_deps
        example_tuple = (
            default_example_coords["dataset"],
            default_example_coords["split"],
            default_example_coords["setup_id"],
            default_example_coords["example_idx"],
        )

        spans_from_get = [
            {
                "annotator_group_id": "c1_g1",
                "annotation_start": 1,
                "annotation_end": 1,
                "annotation_type": "T1",
            },  # Empty
            {"annotator_group_id": "c1_g2", "annotation_start": 3, "annotation_end": 4, "annotation_type": "T1"},
        ]
        mock_get_camp_spans.return_value = pd.DataFrame(spans_from_get)

        # After filtering empty, only c1_g2 remains (1 annotator) -> s_empty_score
        # Expected input to s_empty_score:
        # The row comes from index 1 of spans_from_get
        expected_spans_for_s_empty = pd.DataFrame([spans_from_get[1]], index=[1])

        span_index_df = create_gamma_span_df([])
        annotator_groups = [("c1", "g1"), ("c1", "g2")]
        dissim_mock = MagicMock()

        gamma_val, s_empty_val = _process_example_gamma(
            example_tuple,
            span_index_df,
            annotator_groups,
            dissim_mock,
            soft_gamma=True,
            pa=mock_pa_mod,
            Segment=mock_segment_cls,
        )

        mock_unit_compute_gamma.assert_not_called()
        mock_unit_compute_s_empty.assert_called_once()

        pd.testing.assert_frame_equal(mock_unit_compute_s_empty.call_args[0][0], expected_spans_for_s_empty)
        assert np.isnan(gamma_val)
        assert s_empty_val == 0.5  # 1 unique annotator with 1 span


@patch("factgenie.iaa.gamma._process_example_gamma")
class TestComputeGammaScores:  # Tests the main compute_gamma_scores orchestrator

    def test_empty_inputs_return_initial_metrics(self, mock_process_example):
        initial_metrics = _initialize_gamma_metrics()
        # The following patches caused an AttributeError because 'pa' and 'Segment'
        # are imported locally within compute_gamma_scores, not module-level attributes.
        # with patch("factgenie.iaa.gamma.pa", MagicMock()), patch("factgenie.iaa.gamma.Segment", MagicMock()):

        # Empty example_list
        res_empty_ex = compute_gamma_scores(create_gamma_span_df([{"a": 1}]), [], pd.DataFrame(), 1, 1, 1, True, None)
        assert res_empty_ex == initial_metrics

        # Empty span_index
        ex_list = pd.DataFrame([{"dataset": "d", "split": "s", "setup_id": "s", "example_idx": 0}])
        res_empty_span = compute_gamma_scores(create_gamma_span_df([]), [], ex_list, 1, 1, 1, True, None)
        assert res_empty_span == initial_metrics

        mock_process_example.assert_not_called()

    def test_aggregation_of_gamma_and_s_empty_scores(self, mock_process_example, default_example_coords):
        ex1 = default_example_coords
        ex2 = {**default_example_coords, "example_idx": 1}
        ex3 = {**default_example_coords, "example_idx": 2}
        example_list = pd.DataFrame([ex1, ex2, ex3])

        mock_process_example.side_effect = [
            (0.8, np.nan),
            (np.nan, 0.5),
            (0.6, np.nan),
        ]
        span_idx = create_gamma_span_df([{"a": 1}])  # Dummy

        mock_tqdm_instance = MagicMock()
        # Make the mock_tqdm_instance itself iterable if the code were to do `for _ in pbar:`
        # However, the code iterates `example_list.iterrows()` and calls `pbar.update()`
        # So, the key is that `tqdm()` returns something that has an `update` method.

        with (
            patch("pygamma_agreement.continuum", MagicMock()),
            patch("pyannote.core.Segment", MagicMock()),
            patch("pygamma_agreement.notebook.Notebook", MagicMock()),
            patch("factgenie.iaa.gamma.tqdm", MagicMock(return_value=mock_tqdm_instance)),
        ):
            results = compute_gamma_scores(span_idx, [], example_list, 1, 1, 1, True, None)

        assert mock_process_example.call_count == 3
        # Results are appended only if not np.nan
        assert results["gamma_scores"] == [0.8, 0.6]
        assert results["s_empty_scores"] == [0.5]
        assert results["gamma_mean"] == pytest.approx(0.7)
        assert results["s_empty_mean"] == pytest.approx(0.5)
        assert results["example_count"] == 3


@patch("factgenie.iaa.gamma.app", MagicMock())
@patch("factgenie.workflows.generate_campaign_index")
@patch("factgenie.analysis.assert_common_categories")
@patch("factgenie.analysis.compute_span_index")
@patch("factgenie.analysis.get_example_list")
@patch("factgenie.iaa.gamma.compute_gamma_scores")  # The one being wrapped
class TestComputeGamma:  # Tests the main compute_gamma wrapper

    def _setup_mocks(
        self,
        mock_cgs,
        mock_gel,
        mock_csi,
        mock_acc,
        mock_gci,
        campaign_index={},
        assert_common=True,
        span_index_df=None,
        example_list_df=None,
        cgs_return_val=None,
    ):
        mock_gci.return_value = campaign_index
        mock_acc.return_value = assert_common
        mock_csi.return_value = span_index_df if span_index_df is not None else create_gamma_span_df([])
        mock_gel.return_value = example_list_df if example_list_df is not None else pd.DataFrame()
        mock_cgs.return_value = cgs_return_val if cgs_return_val is not None else _initialize_gamma_metrics()

    def test_successful_flow_and_params_propagation(
        self, mock_cgs, mock_gel, mock_csi, mock_acc, mock_gci, default_example_coords
    ):
        c_ids, grps = ["c1", "c2"], ["gA", None]
        a, b, de, sg = 0.5, 0.6, 0.7, False
        plots_dir = "output_plots/"

        ex_list = pd.DataFrame([default_example_coords])
        sp_idx = create_gamma_span_df([default_example_coords])  # Dummy
        expected_cgs_res = {"gamma_mean": 0.9, "example_count": 1}

        self._setup_mocks(
            mock_cgs,
            mock_gel,
            mock_csi,
            mock_acc,
            mock_gci,
            campaign_index={"c1": MagicMock(), "c2": MagicMock()},
            span_index_df=sp_idx,
            example_list_df=ex_list,
            cgs_return_val=expected_cgs_res,
        )

        res = compute_gamma(c_ids, grps, a, b, de, sg, ["ds1"], ["train"], [0], plots_dir)

        assert res == expected_cgs_res
        mock_gci.assert_called_once_with(gamma_module.app, force_reload=True)
        mock_acc.assert_called_once_with(c_ids, mock_gci.return_value)

        expected_annotator_groups = [("c1", "gA"), ("c2", None)]
        mock_csi.assert_called_once_with(gamma_module.app, c_ids, mock_gci.return_value)
        mock_gel.assert_called_once_with(mock_gci.return_value, expected_annotator_groups, ["ds1"], ["train"], [0])
        mock_cgs.assert_called_once_with(
            span_index=sp_idx,
            annotator_groups=expected_annotator_groups,
            example_list=ex_list,
            alpha=a,
            beta=b,
            delta_empty=de,
            soft_gamma=sg,
            save_plots_dir=plots_dir,
        )

    def test_assert_common_categories_false_returns_none(self, mock_cgs, mock_gel, mock_csi, mock_acc, mock_gci):
        self._setup_mocks(mock_cgs, mock_gel, mock_csi, mock_acc, mock_gci, assert_common=False)
        assert compute_gamma(["c1"], [None]) is None
        mock_csi.assert_not_called()
        mock_cgs.assert_not_called()
