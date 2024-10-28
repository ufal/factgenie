#!/usr/bin/env python3
import os
import json
import glob
import logging
import pandas as pd
import ast

from datetime import datetime
from factgenie import CAMPAIGN_DIR

logger = logging.getLogger(__name__)


class CampaignMode:
    CROWDSOURCING = "crowdsourcing"
    LLM_EVAL = "llm_eval"
    LLM_GEN = "llm_gen"
    EXTERNAL = "external"
    HIDDEN = "hidden"


class CampaignStatus:
    IDLE = "idle"
    RUNNING = "running"
    FINISHED = "finished"


class ExampleStatus:
    FREE = "free"
    ASSIGNED = "assigned"
    FINISHED = "finished"


class Campaign:
    @classmethod
    def get_name(cls):
        return cls.__name__

    def __init__(self, campaign_id):
        self.campaign_id = campaign_id
        self.dir = os.path.join(CAMPAIGN_DIR, campaign_id)
        self.db_path = os.path.join(self.dir, "db.csv")
        self.metadata_path = os.path.join(self.dir, "metadata.json")

        self.load_db()
        self.load_metadata()

    def get_finished_examples(self):
        # load all the JSONL files in the "files" subdirectory
        examples_finished = []

        for jsonl_file in glob.glob(os.path.join(self.dir, "files/*.jsonl")):
            with open(jsonl_file) as f:
                for line in f:
                    example = json.loads(line)
                    examples_finished.append(example)

        return examples_finished

    def update_db(self, db):
        self.db = db
        db.to_csv(self.db_path, index=False)

    def load_db(self):
        dtype_dict = {"annotator_id": str, "start": float, "end": float}

        with open(self.db_path) as f:
            self.db = pd.read_csv(f, dtype=dtype_dict)

    def update_metadata(self):
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=4)

    def load_metadata(self):
        with open(self.metadata_path) as f:
            self.metadata = json.load(f)

    def clear_all_outputs(self):
        # remove files
        for jsonl_file in glob.glob(os.path.join(self.dir, "files/*.jsonl")):
            os.remove(jsonl_file)

        self.db["status"] = ExampleStatus.FREE
        self.db["annotator_id"] = ""
        self.db["start"] = None
        self.db["end"] = None
        self.update_db(self.db)

        self.metadata["status"] = CampaignStatus.IDLE
        self.update_metadata()

    def clear_single_output(self, idx, idx_type="example_idx", annotator_group=None):
        # Identify the rows where idx_type matches idx
        mask = self.db[idx_type] == idx

        if annotator_group:
            mask = mask & (self.db["annotator_group"] == annotator_group)

        # Update the DataFrame using .loc
        self.db.loc[mask, "status"] = ExampleStatus.FREE
        self.db.loc[mask, "annotator_id"] = ""
        self.db.loc[mask, "start"] = None
        self.db.loc[mask, "end"] = None

        self.update_db(self.db)

        if self.metadata.get("status") == CampaignStatus.FINISHED:
            self.metadata["status"] = CampaignStatus.IDLE
            self.update_metadata()

        # remove any outputs from JSONL files
        dataset = self.db.loc[mask, "dataset"].values[0]
        split = self.db.loc[mask, "split"].values[0]
        setup_id = self.db.loc[mask, "setup_id"].values[0]
        example_idx = self.db.loc[mask, idx_type].values[0]

        for jsonl_file in glob.glob(os.path.join(self.dir, "files/*.jsonl")):
            with open(jsonl_file, "r") as f:
                lines = f.readlines()

            with open(jsonl_file, "w") as f:
                for line in lines:
                    data = json.loads(line)
                    if not (
                        data["dataset"] == dataset
                        and data["split"] == split
                        and data["setup_id"] == setup_id
                        and data[idx_type] == example_idx
                        and data["metadata"].get("annotator_group", None) == annotator_group
                    ):
                        f.write(line)

        logger.info(f"Cleared outputs and assignments for {idx}")


class ExternalCampaign(Campaign):
    def get_stats(self):
        return {}


class HumanCampaign(Campaign):
    def __init__(self, campaign_id, scheduler):
        super().__init__(campaign_id)

        scheduler.add_job(
            self.check_idle_time, "interval", minutes=1, id=f"idle_time_{self.campaign_id}", replace_existing=True
        )

    def check_idle_time(self):
        current_time = datetime.now()
        for _, example in self.db.iterrows():
            if (
                example.status == ExampleStatus.ASSIGNED
                and (current_time - datetime.fromtimestamp(example.start)).total_seconds()
                > self.metadata["config"]["idle_time"] * 60
            ):
                logger.info(f"Freeing example {example.example_idx} for {self.campaign_id} due to idle time")
                self.clear_single_output(example.example_idx)

    def get_stats(self):
        # group by batch_idx, keep the first row of each group
        batch_stats = self.db.groupby(["batch_idx", "annotator_group"]).first()

        return {
            "total": len(batch_stats),
            "assigned": len(batch_stats[batch_stats["status"] == ExampleStatus.ASSIGNED]),
            "finished": len(batch_stats[batch_stats["status"] == ExampleStatus.FINISHED]),
            "free": len(batch_stats[batch_stats["status"] == ExampleStatus.FREE]),
        }

    def clear_output(self, idx, annotator_group):
        self.clear_single_output(idx, idx_type="batch_idx", annotator_group=annotator_group)

    def get_overview(self):
        self.load_db()
        df = self.db.copy()
        # replace NaN with empty string
        df = df.where(pd.notnull(df), "")

        # Group by batch_idx and annotator_group
        grouped = df.groupby(["batch_idx", "annotator_group"])

        # Aggregate the necessary columns
        overview_df = grouped.agg(
            example_list=pd.NamedAgg(
                column="example_idx",
                aggfunc=lambda x: x.index.map(
                    lambda idx: {
                        "dataset": df.at[idx, "dataset"],
                        "split": df.at[idx, "split"],
                        "setup_id": df.at[idx, "setup_id"],
                        "example_idx": df.at[idx, "example_idx"],
                    }
                ).tolist(),
            ),
            example_cnt=pd.NamedAgg(column="example_idx", aggfunc="count"),
            status=pd.NamedAgg(column="status", aggfunc="first"),
            annotator_id=pd.NamedAgg(column="annotator_id", aggfunc="first"),
            start=pd.NamedAgg(column="start", aggfunc="first"),
            end=pd.NamedAgg(column="end", aggfunc="first"),
        ).reset_index()

        for col in ["status", "annotator_id", "start", "end"]:
            overview_df[col] = overview_df[col].astype(df[col].dtype)

        return overview_df.to_dict(orient="records")


class LLMCampaign(Campaign):
    def get_stats(self):
        return {
            "total": len(self.db),
            "finished": len(self.db[self.db["status"] == ExampleStatus.FINISHED]),
            "free": len(self.db[self.db["status"] == ExampleStatus.FREE]),
        }

    def clear_output(self, idx):
        self.clear_single_output(idx, idx_type="example_idx")


class LLMCampaignEval(LLMCampaign):
    def get_overview(self):
        self.load_db()
        overview_db = self.db.copy()
        overview_db["output"] = ""

        # get the finished examples
        finished_examples = self.get_finished_examples()
        example_index = {
            (ex["dataset"], ex["split"], ex["setup_id"], ex["example_idx"]): str(ex) for ex in finished_examples
        }
        overview_db["record"] = {}

        for i, row in self.db.iterrows():
            key = (row["dataset"], row["split"], row["setup_id"], row["example_idx"])
            example = ast.literal_eval(example_index.get(key, "{}"))

            annotations = example.get("annotations", [])
            overview_db.at[i, "record"] = str(annotations)

        overview_db = overview_db.to_dict(orient="records")

        return overview_db


class LLMCampaignGen(LLMCampaign):
    # Enables showing the generated outputs on the campaign detail page even though the outputs are not yet exported
    def get_overview(self):
        finished_examples = self.get_finished_examples()

        example_index = {(ex["dataset"], ex["split"], ex["example_idx"]): str(ex) for ex in finished_examples}

        self.load_db()
        overview_db = self.db.copy()
        overview_db["record"] = ""

        for i, row in self.db.iterrows():
            key = (row["dataset"], row["split"], row["example_idx"])
            example = ast.literal_eval(example_index.get(key, "{}"))

            overview_db.at[i, "record"] = str(example.get("output", ""))

        overview_db = overview_db.to_dict(orient="records")
        return overview_db
