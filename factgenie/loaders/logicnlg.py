#!/usr/bin/env python3
import ast
import json
import logging
from datasets import load_dataset

logger = logging.getLogger(__name__)
from factgenie.loaders.dataset import Dataset
from factgenie.loaders.dataset import OUTPUT_DIR

from tinyhtml import h
from collections import defaultdict
from pathlib import Path
from slugify import slugify


class LogicNLG(Dataset):
    def load_examples(self, split, data_path):
        # loading only 100 examples for the sample
        hf_dataset = load_dataset("kasnerz/logicnlg", split=split)

        # we support only test split
        if split != "test":
            return []

        # just use the first json and assume the rest use the same tables
        needed_table_ids = Path(OUTPUT_DIR) / "logicnlg" / "test" / "GPT4-direct-2shotCoT.json"
        with open(needed_table_ids) as f:
            needed_table_ids = json.load(f)

        table_ids = [out["table_id"] for out in needed_table_ids["generated"]]

        tables = {}

        for example in hf_dataset:
            table = ast.literal_eval(example["table"])
            table_title = example["title"]
            table_id = example["table_id"]
            if table_id not in table_ids:
                continue
            tables[table_id] = (table, table_title, table_id)

        examples = [tables[tid] for tid in table_ids]

        return examples

    def render(self, example):
        table_title = f"{example[1]}"
        table_id = f"[{example[2]}]"
        header_el = h("div")(h("i")(table_id), h("h4", klass="")(table_title))

        rows = example[0]
        trs = [h("tr")([h("th")(c) for c in rows[0]])]
        for row in rows[1:]:
            cells = [h("td")(v) for v in row]
            tr_el = h("tr")(cells)
            trs.append(tr_el)

        tbody_el = h("tbody", id="main-table-body")(trs)
        table_el = h(
            "table",
            klass="table table-sm table-bordered caption-top main-table font-mono",
        )(tbody_el)

        html_el = h("div")(header_el, table_el)

        return html_el.render()
