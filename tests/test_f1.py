from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

import factgenie.iaa.f1 as f1
from factgenie.iaa.f1 import compute_f1, compute_f1_scores


# Helper function to create span DataFrames
def create_span_df(spans_data):
    """
    Creates a DataFrame from a list of span dictionaries.
    Each dict should have: annotator_group, annotation_start,0,
                           annotation_type, dataset, split, setup_id, example_idx.
    """
    if not spans_data:
        return pd.DataFrame(
            columns=[
                "dataset",
                "split",
                "setup_id",
                "example_idx",
                "campaign_id",  # Add campaign_id column
                "annotation_start",
                "annotation_end",
                "annotator_group",
                "annotation_type",
                "annotation_text",
            ]
        )
    df = pd.DataFrame(spans_data)
    # Ensure 'annotation_text' exists, e.g. by deriving from start/end
    # Also ensure essential columns for example_list extraction are present
    df["annotation_text"] = df.apply(lambda r: "x" * (r["annotation_end"] - r["annotation_start"]), axis=1)
    return df


def create_example_list_from_spans(span_df):
    if span_df.empty:
        return pd.DataFrame(columns=["dataset", "split", "setup_id", "example_idx"])
    return span_df[["dataset", "split", "setup_id", "example_idx"]].drop_duplicates().reset_index(drop=True)


@pytest.fixture
def default_example_coords():
    return {"dataset": "ds1", "split": "train", "setup_id": "s1", "example_idx": 0}


class TestComputeF1Scores:
    def test_perfect_overlap_hard_match(self, default_example_coords):
        ref_spans = create_span_df(
            [
                {
                    **default_example_coords,
                    "campaign_id": "ref_camp_id",
                    "annotator_group": 0,
                    "annotation_start": 0,
                    "annotation_end": 5,
                    "annotation_type": "A",
                }
            ]
        )
        hyp_spans = create_span_df(
            [
                {
                    **default_example_coords,
                    "campaign_id": "hyp_camp_id",
                    "annotator_group": 0,
                    "annotation_start": 0,
                    "annotation_end": 5,
                    "annotation_type": "A",
                }
            ]
        )
        span_index = pd.concat([ref_spans, hyp_spans])
        example_list = create_example_list_from_spans(span_index)

        result = compute_f1_scores(
            span_index=span_index,
            annotator_groups=[("ref_camp_id", 0), ("hyp_camp_id", 0)],
            example_list=example_list,
            match_mode="hard",
        )
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
        assert result["ref_count"] == 1
        assert result["hyp_count"] == 1

    def test_perfect_overlap_soft_match_diff_type(self, default_example_coords):
        ref_spans = create_span_df(
            [
                {
                    **default_example_coords,
                    "campaign_id": "ref_camp_id",
                    "annotator_group": 0,
                    "annotation_start": 0,
                    "annotation_end": 5,
                    "annotation_type": "A",
                }
            ]
        )
        hyp_spans = create_span_df(
            [
                {
                    **default_example_coords,
                    "campaign_id": "hyp_camp_id",
                    "annotator_group": 0,
                    "annotation_start": 0,
                    "annotation_end": 5,
                    "annotation_type": "B",
                }
            ]
        )
        span_index = pd.concat([ref_spans, hyp_spans])
        example_list = create_example_list_from_spans(span_index)

        result = compute_f1_scores(
            span_index=span_index,
            annotator_groups=[("ref_camp_id", 0), ("hyp_camp_id", 0)],
            example_list=example_list,
            match_mode="soft",
        )
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_perfect_overlap_hard_match_diff_type(self, default_example_coords):
        ref_spans = create_span_df(
            [
                {
                    **default_example_coords,
                    "campaign_id": "ref_camp_id",
                    "annotator_group": 0,
                    "annotation_start": 0,
                    "annotation_end": 5,
                    "annotation_type": "A",
                }
            ]
        )
        hyp_spans = create_span_df(
            [
                {
                    **default_example_coords,
                    "campaign_id": "hyp_camp_id",
                    "annotator_group": 0,
                    "annotation_start": 0,
                    "annotation_end": 5,
                    "annotation_type": "B",
                }
            ]
        )
        span_index = pd.concat([ref_spans, hyp_spans])
        example_list = create_example_list_from_spans(span_index)

        result = compute_f1_scores(
            span_index=span_index,
            annotator_groups=[("ref_camp_id", 0), ("hyp_camp_id", 0)],
            example_list=example_list,
            match_mode="hard",
        )
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_partial_overlap(self, default_example_coords):
        ref_spans = create_span_df(
            [
                {
                    **default_example_coords,
                    "campaign_id": "ref_camp_id",
                    "annotator_group": 0,
                    "annotation_start": 0,
                    "annotation_end": 5,
                    "annotation_type": "A",
                }
            ]
        )  # len 5
        hyp_spans = create_span_df(
            [
                {
                    **default_example_coords,
                    "campaign_id": "hyp_camp_id",
                    "annotator_group": 0,
                    "annotation_start": 3,
                    "annotation_end": 8,
                    "annotation_type": "A",
                }
            ]
        )  # len 5, overlap 2 (3,4)
        span_index = pd.concat([ref_spans, hyp_spans])
        example_list = create_example_list_from_spans(span_index)

        # TP = 2 (chars 3, 4)
        # FP = 3 (chars 5, 6, 7 of hyp)
        # FN = 3 (chars 0, 1, 2 of ref)
        # P = TP / (TP+FP) = 2 / (2+3) = 2/5 = 0.4
        # R = TP / (TP+FN) = 2 / (2+3) = 2/5 = 0.4
        # F1 = 2 * 0.4 * 0.4 / (0.4+0.4) = 0.32 / 0.8 = 0.4
        result = compute_f1_scores(
            span_index=span_index,
            annotator_groups=[("ref_camp_id", 0), ("hyp_camp_id", 0)],
            example_list=example_list,
            match_mode="hard",
        )
        assert result["precision"] == 0.4
        assert result["recall"] == 0.4
        assert result["f1"] == 0.4

    def test_no_overlap(self, default_example_coords):
        ref_spans = create_span_df(
            [
                {
                    **default_example_coords,
                    "campaign_id": "ref_camp_id",
                    "annotator_group": 0,
                    "annotation_start": 0,
                    "annotation_end": 5,
                    "annotation_type": "A",
                }
            ]
        )
        hyp_spans = create_span_df(
            [
                {
                    **default_example_coords,
                    "campaign_id": "hyp_camp_id",
                    "annotator_group": 0,
                    "annotation_start": 6,
                    "annotation_end": 10,
                    "annotation_type": "A",
                }
            ]
        )
        span_index = pd.concat([ref_spans, hyp_spans])
        example_list = create_example_list_from_spans(span_index)

        result = compute_f1_scores(
            span_index=span_index,
            annotator_groups=[("ref_camp_id", 0), ("hyp_camp_id", 0)],
            example_list=example_list,
            match_mode="hard",
        )
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_hyp_contained_in_ref(self, default_example_coords):
        ref_spans = create_span_df(
            [
                {
                    **default_example_coords,
                    "campaign_id": "ref_camp_id",
                    "annotator_group": 0,
                    "annotation_start": 0,
                    "annotation_end": 10,
                    "annotation_type": "A",
                }
            ]
        )  # len 10
        hyp_spans = create_span_df(
            [
                {
                    **default_example_coords,
                    "campaign_id": "hyp_camp_id",
                    "annotator_group": 0,
                    "annotation_start": 2,
                    "annotation_end": 5,
                    "annotation_type": "A",
                }
            ]
        )  # len 3
        span_index = pd.concat([ref_spans, hyp_spans])
        example_list = create_example_list_from_spans(span_index)

        # TP = 3. Hyp len = 3. Ref len = 10.
        # P = 3/3 = 1.0
        # R = 3/10 = 0.3
        # F1 = 2 * 1 * 0.3 / (1+0.3) = 0.6 / 1.3 = 0.4615...
        result = compute_f1_scores(
            span_index=span_index,
            annotator_groups=[("ref_camp_id", 0), ("hyp_camp_id", 0)],
            example_list=example_list,
            match_mode="hard",
        )
        assert result["precision"] == 1.0
        assert result["recall"] == 0.3
        assert pytest.approx(result["f1"]) == 0.462  # round(0.46153846153, 3)

    def test_ref_contained_in_hyp(self, default_example_coords):
        ref_spans = create_span_df(
            [
                {
                    **default_example_coords,
                    "campaign_id": "ref_camp_id",
                    "annotator_group": 0,
                    "annotation_start": 2,
                    "annotation_end": 5,
                    "annotation_type": "A",
                }
            ]
        )  # len 3
        hyp_spans = create_span_df(
            [
                {
                    **default_example_coords,
                    "campaign_id": "hyp_camp_id",
                    "annotator_group": 0,
                    "annotation_start": 0,
                    "annotation_end": 10,
                    "annotation_type": "A",
                }
            ]
        )  # len 10
        span_index = pd.concat([ref_spans, hyp_spans])
        example_list = create_example_list_from_spans(span_index)

        # TP = 3. Hyp len = 10. Ref len = 3.
        # P = 3/10 = 0.3
        # R = 3/3 = 1.0
        # F1 = 0.4615...
        result = compute_f1_scores(
            span_index=span_index,
            annotator_groups=[("ref_camp_id", 0), ("hyp_camp_id", 0)],
            example_list=example_list,
            match_mode="hard",
        )
        assert result["precision"] == 0.3
        assert result["recall"] == 1.0
        assert pytest.approx(result["f1"]) == 0.462

    def test_multiple_hyp_overlap_one_ref(self, default_example_coords):
        ref_spans = create_span_df(
            [
                {
                    **default_example_coords,
                    "campaign_id": "ref_camp_id",
                    "annotator_group": 0,
                    "annotation_start": 0,
                    "annotation_end": 10,
                    "annotation_type": "A",
                }
            ]
        )  # len 10
        hyp_spans_data = [
            {
                **default_example_coords,
                "campaign_id": "hyp_camp_id",
                "annotator_group": 0,
                "annotation_start": 0,
                "annotation_end": 3,
                "annotation_type": "A",
            },  # len 3, TP 3
            {
                **default_example_coords,
                "campaign_id": "hyp_camp_id",
                "annotator_group": 0,
                "annotation_start": 7,
                "annotation_end": 10,
                "annotation_type": "A",
            },  # len 3, TP 3
        ]
        hyp_spans = create_span_df(hyp_spans_data)
        span_index = pd.concat([ref_spans, hyp_spans])
        example_list = create_example_list_from_spans(span_index)

        # Total TP = 3+3 = 6. Total Hyp len = 3+3 = 6. Total Ref len = 10.
        # P = 6/6 = 1.0
        # R = 6/10 = 0.6
        # F1 = 2 * 1 * 0.6 / (1+0.6) = 1.2 / 1.6 = 0.75
        result = compute_f1_scores(
            span_index=span_index,
            annotator_groups=[("ref_camp_id", 0), ("hyp_camp_id", 0)],
            example_list=example_list,
            match_mode="hard",
        )
        assert result["precision"] == 1.0
        assert result["recall"] == 0.6
        assert result["f1"] == 0.750
        assert result["hyp_count"] == 2

    def test_one_hyp_overlap_multiple_ref(self, default_example_coords):
        ref_spans_data = [
            {
                **default_example_coords,
                "campaign_id": "ref_camp_id",
                "annotator_group": 0,
                "annotation_start": 0,
                "annotation_end": 3,
                "annotation_type": "A",
            },  # len 3
            {
                **default_example_coords,
                "campaign_id": "ref_camp_id",
                "annotator_group": 0,
                "annotation_start": 7,
                "annotation_end": 10,
                "annotation_type": "A",
            },  # len 3
        ]
        ref_spans = create_span_df(ref_spans_data)
        hyp_spans = create_span_df(
            [
                {
                    **default_example_coords,
                    "campaign_id": "hyp_camp_id",
                    "annotator_group": 0,
                    "annotation_start": 0,
                    "annotation_end": 10,
                    "annotation_type": "A",
                }
            ]
        )  # len 10
        span_index = pd.concat([ref_spans, hyp_spans])
        example_list = create_example_list_from_spans(span_index)

        # Hyp (0,10) overlaps Ref1 (0,3) -> TP part 3
        # Hyp (0,10) overlaps Ref2 (7,10) -> TP part 3
        # Total TP = 3+3 = 6. Total Hyp len = 10. Total Ref len = 3+3 = 6.
        # P = 6/10 = 0.6
        # R = 6/6 = 1.0
        # F1 = 0.75
        result = compute_f1_scores(
            span_index=span_index,
            annotator_groups=[("ref_camp_id", 0), ("hyp_camp_id", 0)],
            example_list=example_list,
            match_mode="hard",
        )
        assert result["precision"] == 0.6
        assert result["recall"] == 1.0
        assert result["f1"] == 0.750
        assert result["ref_count"] == 2

    def test_empty_ref_spans(self, default_example_coords):
        ref_spans = create_span_df([])
        hyp_spans = create_span_df(
            [
                {
                    **default_example_coords,
                    "campaign_id": "hyp_camp_id",
                    "annotator_group": 0,
                    "annotation_start": 0,
                    "annotation_end": 5,
                    "annotation_type": "A",
                }
            ]
        )
        span_index = pd.concat([ref_spans, hyp_spans])  # span_index will only contain hyp_spans
        example_list = create_example_list_from_spans(hyp_spans)  # example_list based on available hyp spans

        result = compute_f1_scores(
            span_index=span_index, annotator_groups=[("ref_camp_id", 0), ("hyp_camp_id", 0)], example_list=example_list
        )
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0
        assert result["ref_count"] == 0
        assert result["hyp_count"] == 1

    def test_empty_hyp_spans(self, default_example_coords):
        ref_spans = create_span_df(
            [
                {
                    **default_example_coords,
                    "campaign_id": "ref_camp_id",
                    "annotator_group": 0,
                    "annotation_start": 0,
                    "annotation_end": 5,
                    "annotation_type": "A",
                }
            ]
        )
        hyp_spans = create_span_df([])
        span_index = pd.concat([ref_spans, hyp_spans])  # span_index will only contain ref_spans
        example_list = create_example_list_from_spans(ref_spans)  # example_list based on available ref spans

        result = compute_f1_scores(
            span_index=span_index, annotator_groups=[("ref_camp_id", 0), ("hyp_camp_id", 0)], example_list=example_list
        )
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0
        assert result["ref_count"] == 1
        assert result["hyp_count"] == 0

    def test_both_empty_spans(self):
        ref_spans = create_span_df([])
        hyp_spans = create_span_df([])
        span_index = pd.concat([ref_spans, hyp_spans])
        example_list = create_example_list_from_spans(span_index)  # empty DataFrame

        result = compute_f1_scores(
            span_index=span_index, annotator_groups=[("ref_camp_id", 0), ("hyp_camp_id", 0)], example_list=example_list
        )
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0
        assert result["ref_count"] == 0
        assert result["hyp_count"] == 0

    def test_category_breakdown(self, default_example_coords):
        spans_data = [
            {
                **default_example_coords,
                "campaign_id": "ref_camp_id",
                "annotator_group": 0,
                "annotation_start": 0,
                "annotation_end": 5,
                "annotation_type": "A",
            },  # Ref A, len 5
            {
                **default_example_coords,
                "campaign_id": "ref_camp_id",
                "annotator_group": 0,
                "annotation_start": 10,
                "annotation_end": 15,
                "annotation_type": "B",
            },  # Ref B, len 5
            {
                **default_example_coords,
                "campaign_id": "hyp_camp_id",
                "annotator_group": 0,
                "annotation_start": 0,
                "annotation_end": 5,
                "annotation_type": "A",
            },  # Hyp A, len 5, TP 5
            {
                **default_example_coords,
                "campaign_id": "hyp_camp_id",
                "annotator_group": 0,
                "annotation_start": 10,
                "annotation_end": 12,
                "annotation_type": "B",
            },  # Hyp B, len 2, TP 2
        ]
        span_index = create_span_df(spans_data)
        example_list = create_example_list_from_spans(span_index)

        result = compute_f1_scores(
            span_index=span_index,
            annotator_groups=[("ref_camp_id", 0), ("hyp_camp_id", 0)],
            example_list=example_list,
            category_breakdown=True,
        )

        # Overall: TP = 5+2=7. HypLen = 5+2=7. RefLen = 5+5=10
        # P = 7/7 = 1.0. R = 7/10 = 0.7. F1 = 2*1*0.7 / 1.7 = 1.4/1.7 = 0.8235...
        assert result["precision"] == 1.0
        assert result["recall"] == 0.7
        assert pytest.approx(result["f1"]) == 0.824

        assert "categories" in result

        cat_a_metrics = result["categories"]["A"]
        # Cat A: TP=5, HypLen=5, RefLen=5. P=1, R=1, F1=1
        assert cat_a_metrics["precision"] == 1.0
        assert cat_a_metrics["recall"] == 1.0
        assert cat_a_metrics["f1"] == 1.0

        cat_b_metrics = result["categories"]["B"]
        # Cat B: TP=2, HypLen=2, RefLen=5. P=1, R=0.4, F1=0.571...
        assert cat_b_metrics["precision"] == 1.0
        assert cat_b_metrics["recall"] == 0.4
        assert pytest.approx(cat_b_metrics["f1"]) == 0.571

    def test_multiple_examples_isolated_metrics(self):
        spans_data = [
            # Example 0: Perfect match
            {
                "dataset": "ds1",
                "split": "train",
                "setup_id": "s1",
                "example_idx": 0,
                "campaign_id": "ref_camp_id",
                "annotator_group": 0,
                "annotation_start": 0,
                "annotation_end": 5,
                "annotation_type": "A",
            },
            {
                "dataset": "ds1",
                "split": "train",
                "setup_id": "s1",
                "example_idx": 0,
                "campaign_id": "hyp_camp_id",
                "annotator_group": 0,
                "annotation_start": 0,
                "annotation_end": 5,
                "annotation_type": "A",
            },
            # Example 1: No overlap
            {
                "dataset": "ds1",
                "split": "train",
                "setup_id": "s1",
                "example_idx": 1,
                "campaign_id": "ref_camp_id",
                "annotator_group": 0,
                "annotation_start": 0,
                "annotation_end": 5,
                "annotation_type": "A",
            },
            {
                "dataset": "ds1",
                "split": "train",
                "setup_id": "s1",
                "example_idx": 1,
                "campaign_id": "hyp_camp_id",
                "annotator_group": 0,
                "annotation_start": 10,
                "annotation_end": 15,
                "annotation_type": "A",
            },
        ]
        span_index = create_span_df(spans_data)
        example_list = create_example_list_from_spans(span_index)

        # Ex0: TP=5, HypLen=5, RefLen=5
        # Ex1: TP=0, HypLen=5, RefLen=5
        # Total: TP=5. TotalHypLen=10. TotalRefLen=10.
        # P = 5/10 = 0.5. R = 5/10 = 0.5. F1 = 0.5.
        result = compute_f1_scores(
            span_index=span_index, annotator_groups=[("ref_camp_id", 0), ("hyp_camp_id", 0)], example_list=example_list
        )
        assert result["precision"] == 0.5
        assert result["recall"] == 0.5
        assert result["f1"] == 0.5


@patch("factgenie.iaa.f1.app", MagicMock())  # Mock flask current_app
@patch("factgenie.workflows.generate_campaign_index")
@patch("factgenie.analysis.assert_common_categories")  # Added for more control
@patch("factgenie.analysis.compute_span_index")
@patch("factgenie.analysis.get_example_list")
class TestComputeF1:

    def _setup_mocks(
        self,
        mock_get_example_list,
        mock_compute_span,
        mock_assert_common_cat,  # Added
        mock_gen_camp_idx,
        ref_camp_id="ref_c",
        hyp_camp_id="hyp_c",
        example_list_df=None,
        span_index_data=None,
        ref_groups_in_db=None,
        hyp_groups_in_db=None,
        common_categories_result=True,  # Added
    ):

        mock_campaigns_dict = {}
        # Setup ref campaign
        mock_ref_camp = MagicMock()
        mock_ref_camp.campaign_id = ref_camp_id
        mock_ref_camp.db = pd.DataFrame({"annotator_group": ref_groups_in_db if ref_groups_in_db else [0, 1]})
        mock_ref_camp.metadata = {"config": {"annotation_span_categories": [{"name": "A"}, {"name": "B"}]}}
        mock_campaigns_dict[ref_camp_id] = mock_ref_camp

        # Setup hyp campaign
        mock_hyp_camp = MagicMock()
        mock_hyp_camp.campaign_id = hyp_camp_id
        mock_hyp_camp.db = pd.DataFrame({"annotator_group": hyp_groups_in_db if hyp_groups_in_db else [0, 1]})
        mock_hyp_camp.metadata = {"config": {"annotation_span_categories": [{"name": "A"}, {"name": "B"}]}}
        mock_campaigns_dict[hyp_camp_id] = mock_hyp_camp

        mock_gen_camp_idx.return_value = mock_campaigns_dict
        mock_assert_common_cat.return_value = common_categories_result

        mock_get_example_list.return_value = (
            example_list_df
            if example_list_df is not None
            else pd.DataFrame(columns=["dataset", "split", "setup_id", "example_idx"])
        )

        mock_compute_span.return_value = span_index_data if span_index_data is not None else create_span_df([])

    def test_basic_f1_computation_perfect_match(
        self,
        mock_get_example_list,
        mock_compute_span,
        mock_assert_common_cat,
        mock_gen_camp_idx,
        default_example_coords,
    ):
        span_data = [
            {
                **default_example_coords,
                "campaign_id": "ref_c",
                "annotator_group": 0,
                "annotation_start": 0,
                "annotation_end": 5,
                "annotation_type": "A",
            },
            {
                **default_example_coords,
                "campaign_id": "hyp_c",
                "annotator_group": 0,
                "annotation_start": 0,
                "annotation_end": 5,
                "annotation_type": "A",
            },
        ]
        example_df = pd.DataFrame([default_example_coords])
        self._setup_mocks(
            mock_get_example_list,
            mock_compute_span,
            mock_assert_common_cat,
            mock_gen_camp_idx,
            example_list_df=example_df,
            span_index_data=create_span_df(span_data),
            ref_groups_in_db=[0],
            hyp_groups_in_db=[0],
        )

        results = compute_f1(ref_camp_id="ref_c", ref_group=0, hyp_camp_id="hyp_c", hyp_group=0)

        assert results is not None
        assert results["precision"] == 1.0
        assert results["recall"] == 1.0
        assert results["f1"] == 1.0

    def test_no_common_examples(
        self, mock_get_example_list, mock_compute_span, mock_assert_common_cat, mock_gen_camp_idx
    ):
        self._setup_mocks(
            mock_get_example_list,
            mock_compute_span,
            mock_assert_common_cat,
            mock_gen_camp_idx,
            example_list_df=pd.DataFrame(columns=["dataset", "split", "setup_id", "example_idx"]),  # Empty example list
        )

        results = compute_f1(ref_camp_id="ref_c", ref_group=None, hyp_camp_id="hyp_c", hyp_group=None)

        assert results is not None
        assert results["precision"] == 0.0
        assert results["recall"] == 0.0
        assert results["f1"] == 0.0
        assert results["ref_count"] == 0
        assert results["hyp_count"] == 0
        # mock_compute_span should still be called, but compute_f1_scores will receive an empty example_list
        mock_compute_span.assert_called_once()

    def test_campaign_not_found_or_categories_mismatch(
        self, mock_get_example_list, mock_compute_span, mock_assert_common_cat, mock_gen_camp_idx
    ):
        # Simulate assert_common_categories returning False
        self._setup_mocks(
            mock_get_example_list,
            mock_compute_span,
            mock_assert_common_cat,
            mock_gen_camp_idx,
            common_categories_result=False,  # This will make compute_f1 return None
        )
        # mock_gen_camp_idx can return a dict that would cause assert_common_categories to fail,
        # or we directly mock assert_common_categories to return False.
        # For instance, if one campaign is missing, assert_common_categories (unmocked) would error.
        # If categories differ, it returns False.
        # Here, we directly make it return False.

        results = compute_f1(ref_camp_id="ref_c", ref_group=None, hyp_camp_id="hyp_c_diff_cat", hyp_group=None)
        assert results is None
        mock_get_example_list.assert_not_called()  # Should exit before example list generation
        mock_compute_span.assert_not_called()  # Should exit before span index computation

    def test_filtering_dataset(
        self,
        mock_get_example_list,
        mock_compute_span,
        mock_assert_common_cat,
        mock_gen_camp_idx,
        default_example_coords,
    ):
        coords_ds1 = default_example_coords
        # example_list_df will be the result of get_example_list, so it should be already filtered
        # The mock for get_example_list will be called with include_dataset=["ds1"]
        example_df_filtered_ds1 = pd.DataFrame([coords_ds1])

        # span_index_data contains all spans, filtering happens based on example_list inside compute_f1_scores
        coords_ds2 = {**default_example_coords, "dataset": "ds2"}
        span_data_all = [
            {
                **coords_ds1,
                "campaign_id": "ref_c",
                "annotator_group": 0,
                "annotation_start": 0,
                "annotation_end": 5,
                "annotation_type": "A",
            },
            {
                **coords_ds1,
                "campaign_id": "hyp_c",
                "annotator_group": 0,
                "annotation_start": 0,
                "annotation_end": 5,
                "annotation_type": "A",
            },
            {  # This span from ds2 should be ignored by compute_f1_scores due to example_list
                **coords_ds2,
                "campaign_id": "ref_c",
                "annotator_group": 0,
                "annotation_start": 0,
                "annotation_end": 5,
                "annotation_type": "A",
            },
        ]

        self._setup_mocks(
            mock_get_example_list,
            mock_compute_span,
            mock_assert_common_cat,
            mock_gen_camp_idx,
            example_list_df=example_df_filtered_ds1,  # Mock get_example_list to return already filtered list
            span_index_data=create_span_df(span_data_all),
            ref_groups_in_db=[0],
            hyp_groups_in_db=[0],
        )

        results = compute_f1(
            ref_camp_id="ref_c", ref_group=0, hyp_camp_id="hyp_c", hyp_group=0, include_dataset=["ds1"]
        )

        mock_get_example_list.assert_called_once()
        call_args = mock_get_example_list.call_args

        assert call_args[0][2] == ["ds1"]

        assert results is not None
        assert results["f1"] == 1.0

    def test_filtering_split_and_example_id(
        self, mock_get_example_list, mock_compute_span, mock_assert_common_cat, mock_gen_camp_idx
    ):
        ex0_train_s1 = {"dataset": "ds1", "split": "train", "setup_id": "s1", "example_idx": 0}
        # Mock get_example_list to return only this specific example
        example_df_filtered = pd.DataFrame([ex0_train_s1])

        ex1_train_s1 = {"dataset": "ds1", "split": "train", "setup_id": "s1", "example_idx": 1}
        ex0_test_s1 = {"dataset": "ds1", "split": "test", "setup_id": "s1", "example_idx": 0}
        span_data_all = [
            {
                **ex0_train_s1,
                "campaign_id": "ref_c",
                "annotator_group": 0,
                "annotation_start": 0,
                "annotation_end": 5,
                "annotation_type": "A",
            },
            {
                **ex0_train_s1,
                "campaign_id": "hyp_c",
                "annotator_group": 0,
                "annotation_start": 0,
                "annotation_end": 5,
                "annotation_type": "A",
            },
            {  # This span should be ignored
                **ex1_train_s1,
                "campaign_id": "ref_c",
                "annotator_group": 0,
                "annotation_start": 0,
                "annotation_end": 5,
                "annotation_type": "A",
            },
            {  # This span should be ignored
                **ex0_test_s1,
                "campaign_id": "hyp_c",
                "annotator_group": 0,
                "annotation_start": 0,
                "annotation_end": 5,
                "annotation_type": "A",
            },
        ]
        self._setup_mocks(
            mock_get_example_list,
            mock_compute_span,
            mock_assert_common_cat,
            mock_gen_camp_idx,
            example_list_df=example_df_filtered,  # Mock returns the already filtered list
            span_index_data=create_span_df(span_data_all),
            ref_groups_in_db=[0],
            hyp_groups_in_db=[0],
        )

        results = compute_f1(
            ref_camp_id="ref_c",
            ref_group=0,
            hyp_camp_id="hyp_c",
            hyp_group=0,
            include_split=["train"],
            include_example_id=[0],
        )

        mock_get_example_list.assert_called_once()
        call_args = mock_get_example_list.call_args

        assert call_args[0][3] == ["train"]
        assert call_args[0][4] == [0]

        assert results is not None
        assert results["f1"] == 1.0

    def test_all_groups_used_if_none_specified(
        self,
        mock_get_example_list,
        mock_compute_span,
        mock_assert_common_cat,
        mock_gen_camp_idx,
        default_example_coords,
    ):
        span_data = [
            {
                **default_example_coords,
                "campaign_id": "ref_c",
                "annotator_group": 0,
                "annotation_start": 0,
                "annotation_end": 2,
                "annotation_type": "A",
            },
            {
                **default_example_coords,
                "campaign_id": "ref_c",
                "annotator_group": 1,
                "annotation_start": 3,
                "annotation_end": 5,
                "annotation_type": "A",
            },
            {
                **default_example_coords,
                "campaign_id": "hyp_c",
                "annotator_group": 1,
                "annotation_start": 0,
                "annotation_end": 2,
                "annotation_type": "A",
            },
            {
                **default_example_coords,
                "campaign_id": "hyp_c",
                "annotator_group": 0,
                "annotation_start": 3,
                "annotation_end": 5,
                "annotation_type": "A",
            },
        ]
        example_df = pd.DataFrame([default_example_coords])
        self._setup_mocks(
            mock_get_example_list,
            mock_compute_span,
            mock_assert_common_cat,
            mock_gen_camp_idx,
            example_list_df=example_df,
            span_index_data=create_span_df(span_data),
            ref_groups_in_db=[0, 1],  # Mock DBs have these groups
            hyp_groups_in_db=[0, 1],
        )

        results = compute_f1(ref_camp_id="ref_c", ref_group=None, hyp_camp_id="hyp_c", hyp_group=None)

        assert results is not None
        assert results["f1"] == 1.0
        assert results["precision"] == 1.0
        assert results["recall"] == 1.0

        # Check that get_example_list was called with 'all' for groups if ref_groups/hyp_groups were None.
        mock_get_example_list.assert_called_once()
        annotator_groups_arg = mock_get_example_list.call_args[0][1]  # campaign_index, annotator_groups
        assert annotator_groups_arg == [("ref_c", None), ("hyp_c", None)]

    def test_spans_deduplicated(
        self,
        mock_get_example_list,
        mock_compute_span,
        mock_assert_common_cat,
        mock_gen_camp_idx,
        default_example_coords,
    ):
        span_data = [
            {
                **default_example_coords,
                "campaign_id": "ref_c",
                "annotator_group": 0,
                "annotation_start": 0,
                "annotation_end": 5,
                "annotation_type": "A",
            },
            {
                **default_example_coords,
                "campaign_id": "ref_c",
                "annotator_group": 0,
                "annotation_start": 0,
                "annotation_end": 5,
                "annotation_type": "A",
            },
            {
                **default_example_coords,
                "campaign_id": "hyp_c",
                "annotator_group": 0,
                "annotation_start": 0,
                "annotation_end": 5,
                "annotation_type": "A",
            },
            {
                **default_example_coords,
                "campaign_id": "hyp_c",
                "annotator_group": 0,
                "annotation_start": 0,
                "annotation_end": 5,
                "annotation_type": "A",
            },
            {
                **default_example_coords,
                "campaign_id": "hyp_c",
                "annotator_group": 0,
                "annotation_start": 0,
                "annotation_end": 5,
                "annotation_type": "A",
            },
        ]
        example_df = pd.DataFrame([default_example_coords])
        self._setup_mocks(
            mock_get_example_list,
            mock_compute_span,
            mock_assert_common_cat,
            mock_gen_camp_idx,
            example_list_df=example_df,
            span_index_data=create_span_df(span_data),
            ref_groups_in_db=[0, 1],  # Mock DBs have these groups
            hyp_groups_in_db=[0, 1],
        )

        results = compute_f1(ref_camp_id="ref_c", ref_group=None, hyp_camp_id="hyp_c", hyp_group=None)

        assert results is not None

        assert results["f1"] == 0.8
        assert results["precision"] == 0.667
        assert results["recall"] == 1.0

        # Check that two of the ref-hyp spans were counted against each other and the last one reduced precision.
        mock_get_example_list.assert_called_once()
        annotator_groups_arg = mock_get_example_list.call_args[0][1]  # campaign_index, annotator_groups
        assert annotator_groups_arg == [("ref_c", None), ("hyp_c", None)]

    @patch("factgenie.iaa.f1.compute_f1_scores")
    def test_match_mode_and_category_breakdown_propagated(
        self,
        mock_f1_scores_func,
        mock_get_example_list,
        mock_compute_span,
        mock_assert_common_cat,
        mock_gen_camp_idx,
        default_example_coords,
    ):
        example_df = pd.DataFrame([default_example_coords])
        self._setup_mocks(
            mock_get_example_list,
            mock_compute_span,
            mock_assert_common_cat,
            mock_gen_camp_idx,
            example_list_df=example_df,  # This will be returned by mock_get_example_list
            span_index_data=create_span_df([]),
            ref_groups_in_db=[0],
            hyp_groups_in_db=[0],
        )

        mock_f1_scores_func.return_value = {"precision": 0.1, "recall": 0.2, "f1": 0.15, "ref_count": 1, "hyp_count": 1}

        compute_f1(
            ref_camp_id="ref_c",
            ref_group=0,
            hyp_camp_id="hyp_c",
            hyp_group=0,
            match_mode="soft",
            category_breakdown=True,
        )

        mock_f1_scores_func.assert_called_once()
        args, kwargs = mock_f1_scores_func.call_args
        assert kwargs.get("match_mode") == "soft"
        assert kwargs.get("category_breakdown") is True
        pd.testing.assert_frame_equal(kwargs.get("example_list"), example_df)

    def test_no_spans_in_span_index(  # Renamed for clarity from "no_spans_after_filtering"
        self,
        mock_get_example_list,
        mock_compute_span,
        mock_assert_common_cat,
        mock_gen_camp_idx,
        default_example_coords,
    ):
        # compute_span_index returns empty DataFrame
        example_df = pd.DataFrame([default_example_coords])
        self._setup_mocks(
            mock_get_example_list,
            mock_compute_span,
            mock_assert_common_cat,
            mock_gen_camp_idx,
            example_list_df=example_df,  # get_example_list returns non-empty
            span_index_data=create_span_df([]),  # compute_span_index returns empty
            ref_groups_in_db=[0],
            hyp_groups_in_db=[0],
        )

        results = compute_f1(ref_camp_id="ref_c", ref_group=0, hyp_camp_id="hyp_c", hyp_group=0)

        # compute_f1_scores will be called with an empty span_index
        # and should return 0s.
        assert results is not None
        assert results["precision"] == 0.0
        assert results["recall"] == 0.0
        assert results["f1"] == 0.0
        assert results["ref_count"] == 0
        assert results["hyp_count"] == 0


def main():
    """
    Main function to run the tests in this file when the script is executed directly.
    """
    import sys

    import pytest

    # Run only the tests in this file
    sys.exit(pytest.main(["-xvs", __file__]))


if __name__ == "__main__":
    main()
