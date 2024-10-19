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

        # temporary fix for the old campaigns
        if self.metadata.get("status") in ["new", "paused"]:
            self.metadata["status"] = CampaignStatus.IDLE
            self.update_metadata()

        # if the db does not contain the `end` column, add it
        if "end" not in self.db.columns:
            self.db["end"] = ""
            self.update_db(self.db)

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
        self.update_db(self.db)

        self.metadata["status"] = CampaignStatus.IDLE
        self.update_metadata()

    def clear_single_output(self, idx, idx_type="example_idx"):
        # Identify the rows where idx_type matches idx
        mask = self.db[idx_type] == idx

        # Update the DataFrame using .loc
        self.db.loc[mask, "status"] = ExampleStatus.FREE
        self.db.loc[mask, "annotator_id"] = ""
        self.db.loc[mask, "start"] = None

        self.update_db(self.db)

        if self.metadata.get("status") == CampaignStatus.FINISHED:
            self.metadata["status"] = CampaignStatus.IDLE
            self.update_metadata()

        logger.info(f"Cleared outputs and assignments for {idx}")

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
                    ):
                        f.write(line)


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

    def get_examples_for_batch(self, batch_idx):
        annotator_batch = []

        # find all examples for this batch in self.db
        batch_examples = self.db[self.db["batch_idx"] == batch_idx]

        for _, row in batch_examples.iterrows():
            annotator_batch.append(
                {
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "setup_id": row["setup_id"],
                    "example_idx": row["example_idx"],
                    "annotator_group": row["annotator_group"],
                }
            )
        return annotator_batch

    def get_overview(self):
        self.load_db()
        overview_db = self.db.copy()
        # replace NaN with empty string
        overview_db = overview_db.where(pd.notnull(overview_db), "")

        # group by batch idx
        # add a column with the number of examples for each batch
        # for other columns keep first item
        overview_db = overview_db.groupby("batch_idx").agg(
            {
                "dataset": "first",
                "split": "first",
                "example_idx": "count",
                "setup_id": "first",
                "status": "first",
                "start": "first",
                "end": "first",
                "annotator_id": "first",
                "annotator_group": "first",
            }
        )

        overview_db["example_details"] = overview_db.index.map(lambda batch_idx: self.get_examples_for_batch(batch_idx))

        overview_db = overview_db.rename(columns={"example_idx": "example_cnt"}).reset_index()
        overview_db = overview_db.to_dict(orient="records")

        return overview_db

    def get_stats(self):
        # group by batch_idx, keep the first row of each group
        batch_stats = self.db.groupby("batch_idx").first()

        return {
            "total": len(batch_stats),
            "assigned": len(batch_stats[batch_stats["status"] == ExampleStatus.ASSIGNED]),
            "finished": len(batch_stats[batch_stats["status"] == ExampleStatus.FINISHED]),
            "free": len(batch_stats[batch_stats["status"] == ExampleStatus.FREE]),
        }

    def clear_output(self, idx):
        self.clear_single_output(idx, idx_type="batch_idx")


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
        # pair the examples in db with the finished examples
        # we need to match the examples on (dataset, split, setup, example_idx)
        # add the annotations to the df

        # get the finished examples
        finished_examples = self.get_finished_examples()
        example_index = {
            (ex["dataset"], ex["split"], ex["setup_id"], ex["example_idx"]): str(ex) for ex in finished_examples
        }

        self.load_db()
        overview_db = self.db.copy()
        overview_db["output"] = ""

        for i, row in self.db.iterrows():
            key = (row["dataset"], row["split"], row["setup_id"], row["example_idx"])
            example = ast.literal_eval(example_index.get(key, "{}"))

            annotations = example.get("annotations", [])
            overview_db.at[i, "output"] = str(annotations)

        overview_db = overview_db.to_dict(orient="records")

        return overview_db


class LLMCampaignGen(LLMCampaign):
    def get_overview(self):
        finished_examples = self.get_finished_examples()

        example_index = {(ex["dataset"], ex["split"], ex["example_idx"]): str(ex) for ex in finished_examples}

        self.load_db()
        overview_db = self.db.copy()
        overview_db["output"] = ""

        for i, row in self.db.iterrows():
            key = (row["dataset"], row["split"], row["example_idx"])
            example = ast.literal_eval(example_index.get(key, "{}"))

            overview_db.at[i, "output"] = str(example.get("out", ""))

        overview_db = overview_db.to_dict(orient="records")
        return overview_db
