import unittest
import pandas as pd
from unittest.mock import patch, MagicMock

import sys

sys.path.append("scripts/evaluation")  # Adjust this if f1.py is in a different directory
from f1 import compute_f1, compute_f1_scores
from factgenie.analysis import format_group_id


class TestF1Computation(unittest.TestCase):

    def create_mock_campaign_data(self, campaign_id, annotator_groups_data):
        """
        Creates mock data for a campaign.
        annotator_groups_data is a dict where keys are group_ids and
        values are lists of annotation dicts.
        Each annotation dict: {"dataset": "d1", "split": "s1", "setup_id": "setup1",
                               "example_idx": 0, "annotator_group": group_id,
                               "annotation_type": "typeA", "annotation_start": 0, "annotation_end": 5}
        """
        db_records = []
        span_records = []
        for group_id, annotations in annotator_groups_data.items():
            for ann in annotations:
                db_records.append(
                    {
                        "dataset": ann["dataset"],
                        "split": ann["split"],
                        "setup_id": ann["setup_id"],
                        "example_idx": ann["example_idx"],
                        "annotator_group": group_id,
                        "status": "finished",  # Important for get_common_examples
                    }
                )
                span_records.append(
                    {
                        "campaign_id": campaign_id,
                        "dataset": ann["dataset"],
                        "split": ann["split"],
                        "setup_id": ann["setup_id"],
                        "example_idx": ann["example_idx"],
                        "annotator_group": group_id,
                        "annotator_group_id": format_group_id(campaign_id, group_id),
                        "annotation_type": ann["annotation_type"],
                        "annotation_start": ann["annotation_start"],
                        "annotation_end": ann["annotation_end"],
                        "annotation_text": "text",  # Placeholder
                    }
                )
        mock_campaign = MagicMock()
        mock_campaign.campaign_id = campaign_id

        expected_db_columns = ["dataset", "split", "setup_id", "example_idx", "annotator_group", "status"]
        if not db_records:
            mock_campaign.db = pd.DataFrame(columns=expected_db_columns)
        else:
            mock_campaign.db = pd.DataFrame(db_records)

        mock_campaign.metadata = {"config": {"annotation_span_categories": ["typeA", "typeB"]}}  # Example

        expected_span_columns = [
            "campaign_id",
            "dataset",
            "split",
            "setup_id",
            "example_idx",
            "annotator_group",
            "annotator_group_id",
            "annotation_type",
            "annotation_start",
            "annotation_end",
            "annotation_text",
        ]
        if not span_records:
            final_span_df = pd.DataFrame(columns=expected_span_columns)
        else:
            final_span_df = pd.DataFrame(span_records)
        return mock_campaign, final_span_df

    @patch("f1.create_app")
    @patch("f1.compute_span_index")
    @patch("f1.load_campaign")
    @patch("f1.generate_campaign_index")
    def test_simple_overlap_hard_match(
        self, mock_generate_campaign_index, mock_load_campaign, mock_compute_span_index, mock_create_app
    ):
        # Mock Flask app
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app

        # --- Mock Campaign Data ---
        ref_camp_id = "ref_camp"
        hyp_camp_id = "hyp_camp"

        # Annotations for reference campaign, group 0
        ref_annotations_g0 = [
            {
                "dataset": "d1",
                "split": "s1",
                "setup_id": "setup1",
                "example_idx": 0,
                "annotation_type": "typeA",
                "annotation_start": 0,
                "annotation_end": 10,
            },  # 10 chars
            {
                "dataset": "d1",
                "split": "s1",
                "setup_id": "setup1",
                "example_idx": 1,
                "annotation_type": "typeB",
                "annotation_start": 5,
                "annotation_end": 15,
            },  # 10 chars
        ]
        # Annotations for hypothesis campaign, group 0
        hyp_annotations_g0 = [
            {
                "dataset": "d1",
                "split": "s1",
                "setup_id": "setup1",
                "example_idx": 0,
                "annotation_type": "typeA",
                "annotation_start": 0,
                "annotation_end": 5,
            },  # 5 chars, overlap 5
            {
                "dataset": "d1",
                "split": "s1",
                "setup_id": "setup1",
                "example_idx": 1,
                "annotation_type": "typeA",
                "annotation_start": 0,
                "annotation_end": 10,
            },  # 10 chars, no overlap (wrong type)
        ]

        mock_ref_campaign, ref_spans_df = self.create_mock_campaign_data(ref_camp_id, {0: ref_annotations_g0})
        mock_hyp_campaign, hyp_spans_df = self.create_mock_campaign_data(hyp_camp_id, {0: hyp_annotations_g0})

        all_spans_df = pd.concat([ref_spans_df, hyp_spans_df], ignore_index=True)

        # --- Mock function calls ---
        mock_generate_campaign_index.return_value = {ref_camp_id: mock_ref_campaign, hyp_camp_id: mock_hyp_campaign}

        def side_effect_load_campaign(app, camp_id):
            if camp_id == ref_camp_id:
                return mock_ref_campaign
            if camp_id == hyp_camp_id:
                return mock_hyp_campaign
            return None

        mock_load_campaign.side_effect = side_effect_load_campaign
        mock_compute_span_index.return_value = all_spans_df

        # --- Call compute_f1 ---
        results = compute_f1(
            ref_camp_id=ref_camp_id,
            ref_groups=[0],
            hyp_camp_id=hyp_camp_id,
            hyp_groups=[0],
            match_mode="hard",
            category_breakdown=False,
        )

        # --- Assertions ---
        # Expected:
        # Ref spans: typeA (0-10), typeB (5-15) :: total ref_length = 10 + 10 = 20
        # Hyp spans: typeA (0-5), typeA (0-10) :: total hyp_length = 5 + 10 = 15
        # Overlap (hard match):
        # Example 0: hyp typeA (0-5) with ref typeA (0-10) :: overlap = 5
        # Example 1: hyp typeA (0-10) with ref typeB (5-15) :: overlap = 0 (type mismatch)
        # Total overlap_length = 5
        # Precision = overlap / hyp_length = 5 / 15 = 0.333
        # Recall = overlap / ref_length = 5 / 20 = 0.25
        # F1 = 2 * P * R / (P + R) = 2 * 0.333 * 0.25 / (0.333 + 0.25) = 0.1665 / 0.583 = 0.2856 ~ 0.286

        self.assertIn(f"{ref_camp_id} :: {hyp_camp_id}", results)
        metrics = results[f"{ref_camp_id} :: {hyp_camp_id}"]

        self.assertEqual(metrics["precision"], 0.333)
        self.assertEqual(metrics["recall"], 0.250)
        self.assertEqual(metrics["f1"], 0.286)
        self.assertEqual(metrics["ref_count"], 2)  # Number of ref annotation entries
        self.assertEqual(metrics["hyp_count"], 2)  # Number of hyp annotation entries

    @patch("f1.create_app")
    @patch("f1.compute_span_index")
    @patch("f1.load_campaign")
    @patch("f1.generate_campaign_index")
    def test_no_overlap(
        self, mock_generate_campaign_index, mock_load_campaign, mock_compute_span_index, mock_create_app
    ):
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app

        ref_camp_id = "ref_c"
        hyp_camp_id = "hyp_c"
        ref_annotations = [
            {
                "dataset": "d1",
                "split": "s1",
                "setup_id": "s1",
                "example_idx": 0,
                "annotation_type": "T1",
                "annotation_start": 0,
                "annotation_end": 5,
            }
        ]
        hyp_annotations = [
            {
                "dataset": "d1",
                "split": "s1",
                "setup_id": "s1",
                "example_idx": 0,
                "annotation_type": "T1",
                "annotation_start": 10,
                "annotation_end": 15,
            }
        ]

        mock_ref_campaign, ref_spans_df = self.create_mock_campaign_data(ref_camp_id, {0: ref_annotations})
        mock_hyp_campaign, hyp_spans_df = self.create_mock_campaign_data(hyp_camp_id, {0: hyp_annotations})
        all_spans_df = pd.concat([ref_spans_df, hyp_spans_df], ignore_index=True)

        mock_generate_campaign_index.return_value = {ref_camp_id: mock_ref_campaign, hyp_camp_id: mock_hyp_campaign}

        def side_effect_load_campaign(app, camp_id):
            return mock_ref_campaign if camp_id == ref_camp_id else mock_hyp_campaign

        mock_load_campaign.side_effect = side_effect_load_campaign
        mock_compute_span_index.return_value = all_spans_df

        results = compute_f1(ref_camp_id, [0], hyp_camp_id, [0], match_mode="hard")
        metrics = results[f"{ref_camp_id} :: {hyp_camp_id}"]
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)

    @patch("f1.create_app")
    @patch("f1.compute_span_index")
    @patch("f1.load_campaign")
    @patch("f1.generate_campaign_index")
    def test_perfect_overlap_soft_match(
        self, mock_generate_campaign_index, mock_load_campaign, mock_compute_span_index, mock_create_app
    ):
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app

        ref_camp_id = "ref_c"
        hyp_camp_id = "hyp_c"
        # Different types, but should match in soft mode
        ref_annotations = [
            {
                "dataset": "d1",
                "split": "s1",
                "setup_id": "s1",
                "example_idx": 0,
                "annotation_type": "T1",
                "annotation_start": 0,
                "annotation_end": 10,
            }
        ]
        hyp_annotations = [
            {
                "dataset": "d1",
                "split": "s1",
                "setup_id": "s1",
                "example_idx": 0,
                "annotation_type": "T2",
                "annotation_start": 0,
                "annotation_end": 10,
            }
        ]

        mock_ref_campaign, ref_spans_df = self.create_mock_campaign_data(ref_camp_id, {0: ref_annotations})
        mock_hyp_campaign, hyp_spans_df = self.create_mock_campaign_data(hyp_camp_id, {0: hyp_annotations})
        all_spans_df = pd.concat([ref_spans_df, hyp_spans_df], ignore_index=True)

        mock_generate_campaign_index.return_value = {ref_camp_id: mock_ref_campaign, hyp_camp_id: mock_hyp_campaign}

        def side_effect_load_campaign(app, camp_id):
            return mock_ref_campaign if camp_id == ref_camp_id else mock_hyp_campaign

        mock_load_campaign.side_effect = side_effect_load_campaign
        mock_compute_span_index.return_value = all_spans_df

        results = compute_f1(ref_camp_id, [0], hyp_camp_id, [0], match_mode="soft")
        metrics = results[f"{ref_camp_id} :: {hyp_camp_id}"]
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)

    @patch("f1.create_app")
    @patch("f1.compute_span_index")
    @patch("f1.load_campaign")
    @patch("f1.generate_campaign_index")
    def test_category_breakdown(
        self, mock_generate_campaign_index, mock_load_campaign, mock_compute_span_index, mock_create_app
    ):
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app

        ref_camp_id = "ref_cb"
        hyp_camp_id = "hyp_cb"
        ref_annotations = [
            {
                "dataset": "d1",
                "split": "s1",
                "setup_id": "s1",
                "example_idx": 0,
                "annotation_type": "typeA",
                "annotation_start": 0,
                "annotation_end": 10,
            },
            {
                "dataset": "d1",
                "split": "s1",
                "setup_id": "s1",
                "example_idx": 0,
                "annotation_type": "typeB",
                "annotation_start": 10,
                "annotation_end": 20,
            },
        ]
        hyp_annotations = [
            {
                "dataset": "d1",
                "split": "s1",
                "setup_id": "s1",
                "example_idx": 0,
                "annotation_type": "typeA",
                "annotation_start": 0,
                "annotation_end": 5,
            },  # P=1, R=0.5, F1=0.667 for typeA
            {
                "dataset": "d1",
                "split": "s1",
                "setup_id": "s1",
                "example_idx": 0,
                "annotation_type": "typeB",
                "annotation_start": 15,
                "annotation_end": 20,
            },  # P=1, R=0.5, F1=0.667 for typeB
        ]
        mock_ref_campaign, ref_spans_df = self.create_mock_campaign_data(ref_camp_id, {0: ref_annotations})
        mock_hyp_campaign, hyp_spans_df = self.create_mock_campaign_data(hyp_camp_id, {0: hyp_annotations})

        # Ensure all_spans_df has 'typeA' and 'typeB' for category list creation
        all_spans_df = pd.concat([ref_spans_df, hyp_spans_df], ignore_index=True)
        # Manually add a dummy row if a type is missing to ensure it's in unique() list for compute_f1_scores
        # This is a bit of a hack for the test setup; in reality, span_index would have all relevant types.
        if "typeA" not in all_spans_df["annotation_type"].unique():
            all_spans_df = pd.concat([all_spans_df, pd.DataFrame([{"annotation_type": "typeA"}])], ignore_index=True)
        if "typeB" not in all_spans_df["annotation_type"].unique():
            all_spans_df = pd.concat([all_spans_df, pd.DataFrame([{"annotation_type": "typeB"}])], ignore_index=True)

        mock_generate_campaign_index.return_value = {ref_camp_id: mock_ref_campaign, hyp_camp_id: mock_hyp_campaign}

        def side_effect_load_campaign(app, camp_id):
            return mock_ref_campaign if camp_id == ref_camp_id else mock_hyp_campaign

        mock_load_campaign.side_effect = side_effect_load_campaign
        mock_compute_span_index.return_value = all_spans_df

        results = compute_f1(ref_camp_id, [0], hyp_camp_id, [0], match_mode="hard", category_breakdown=True)
        metrics = results[f"{ref_camp_id} :: {hyp_camp_id}"]
        self.assertIn("categories", metrics)
        self.assertIn("typeA", metrics["categories"])
        self.assertIn("typeB", metrics["categories"])
        self.assertEqual(metrics["categories"]["typeA"]["f1"], 0.667)
        self.assertEqual(metrics["categories"]["typeB"]["f1"], 0.667)


if __name__ == "__main__":
    unittest.main()
