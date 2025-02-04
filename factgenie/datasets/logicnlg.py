#!/usr/bin/env python3
import ast
import json
import logging
import os
import requests

logger = logging.getLogger("factgenie")
from factgenie.datasets.hf_dataset import HFDataset
from tinyhtml import h


class LogicNLG(HFDataset):
    @classmethod
    def download(cls, dataset_id, data_download_dir, out_download_dir, splits, outputs, dataset_config, **kwargs):
        super().download(
            dataset_id=dataset_id,
            data_download_dir=data_download_dir,
            out_download_dir=out_download_dir,
            splits=splits,
            outputs=outputs,
            dataset_config=dataset_config,
            hf_id="kasnerz/logicnlg",
        )
        # we support only the test split
        split = "test"

        # our custom outputs for a subset of tables
        output_url = "https://owncloud.cesnet.cz/index.php/s/8SFNAjr1g3AbUcm/download"

        setup_id = "gpt4-direct-2shotcot"
        out_path = out_download_dir / setup_id
        os.makedirs(out_path / setup_id, exist_ok=True)

        logger.info(f"Downloading {output_url}")
        response = requests.get(output_url)
        out_content = json.loads(response.content)

        with open(out_path / f"{setup_id}.jsonl", "w") as f:
            for i, out in enumerate(out_content["generated"]):
                out["dataset"] = dataset_id
                out["split"] = split
                out["setup_id"] = setup_id
                out["example_idx"] = i

                out["metadata"] = {
                    "table_id": out.pop("table_id"),
                    "claims": out.pop("claims"),
                    "claim_ids": out.pop("claims_ids"),
                }
                out["output"] = out.pop("out")

                f.write(json.dumps(out) + "\n")

        # filter the examples for our subset of tables
        table_ids = [out["metadata"]["table_id"] for out in out_content["generated"]]
        tables = {}

        with open(data_download_dir / split / "dataset.jsonl") as f:
            hf_dataset = [json.loads(line) for line in f]

        for example in hf_dataset:
            table = ast.literal_eval(example["table"])
            table_title = example["title"]
            table_id = example["table_id"]
            if table_id not in table_ids:
                continue
            tables[table_id] = (table, table_title, table_id)

        examples = [tables[tid] for tid in table_ids]

        # save the outputs we will be loading
        with open(data_download_dir / split / "examples.json", "w") as f:
            json.dump(examples, f)

        return examples

    def load_examples(self, split, data_path):
        examples_path = data_path / split / "examples.json"
        if not examples_path.exists():
            examples = []
        else:
            with open(examples_path) as f:
                examples = json.load(f)
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
