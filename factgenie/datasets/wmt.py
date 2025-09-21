import json

from factgenie.datasets.basic import BasicDataset


class WMTDataset(BasicDataset):
    def load_examples(self, split, data_path):
        examples = []

        with open(f"{data_path}/{split}.jsonl") as f:
            for line in f:
                examples.append(json.loads(line))

        return examples

    def render(self, example):
        lang = example["input_idx"].split("|")[2]
        domain = example["input_idx"].split("|")[1].split("_")[0].split("-")[2]
        input_idx = example["input_idx"]

        html = f"<h4>{lang} | {domain}</h5><p>{example['src']}</p><hr><small>{input_idx}</small>"
        return html
