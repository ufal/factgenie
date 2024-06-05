#!/usr/bin/env python3
#!/usr/bin/env python3
import os
import json
import glob
import time
import logging
import pandas as pd
import random
import time
import coloredlogs
from factgenie.loaders import DATASET_CLASSES
from pathlib import Path

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO", logger=logger, fmt="%(asctime)s %(levelname)s %(message)s")


DIR_PATH = os.path.dirname(__file__)
CROWDSOURCING_DIR = os.path.join(DIR_PATH, "annotations")


def get_campaigns(source):
    campaigns = {}

    # find all subdirs in CROWDSOURCING_DIR
    for campaign_dir in Path(CROWDSOURCING_DIR).iterdir():
        if not campaign_dir.is_dir():
            continue

        metadata = json.load(open(os.path.join(campaign_dir, "metadata.json")))
        campaign_source = metadata.get("source")

        if campaign_source != source:
            continue

        campaign_id = metadata["id"]

        if source == "human":
            campaign = HumanCampaign(campaign_id=campaign_id)
        elif source == "model":
            campaign = ModelCampaign(campaign_id=campaign_id)

        campaigns[campaign_id] = campaign

    return campaigns


class Campaign:
    @classmethod
    def get_name(cls):
        return cls.__name__

    def __init__(self, campaign_id):
        self.campaign_id = campaign_id
        self.dir = os.path.join(CROWDSOURCING_DIR, campaign_id)
        self.db_path = os.path.join(self.dir, "db.csv")
        self.metadata_path = os.path.join(self.dir, "metadata.json")

        with open(self.db_path) as f:
            self.db = pd.read_csv(f)

        with open(self.metadata_path) as f:
            self.metadata = json.load(f)

    def get_examples_for_batch(self, batch_idx):
        annotator_batch = []

        # find all examples for this batch in self.db
        batch_examples = self.db[self.db["batch_idx"] == batch_idx]

        for _, row in batch_examples.iterrows():
            annotator_batch.append(
                {
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "setup": {"id": row["setup_id"]},
                    "example_idx": row["example_idx"],
                }
            )
        return annotator_batch

    def update_db(self, db):
        db.to_csv(self.db_path, index=False)

    def get_stats(self):
        # group by batch_idx, keep the first row of each group
        batch_stats = self.db.groupby("batch_idx").first()

        return batch_stats["status"].value_counts().to_dict()


class HumanCampaign(Campaign):
    pass


class ModelCampaign(Campaign):
    pass
