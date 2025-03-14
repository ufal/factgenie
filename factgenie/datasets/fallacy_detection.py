#!/usr/bin/env python3
import logging
from factgenie.datasets.basic import JSONLDataset

logger = logging.getLogger(__name__)


class FallacyDetection(JSONLDataset):
    def render(self, example):
        return None