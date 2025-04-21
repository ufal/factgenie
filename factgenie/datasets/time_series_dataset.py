from factgenie.datasets.dataset import Dataset
import base64
import datetime
import io
import json
import markdown
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import re

import logging
logger = logging.getLogger("factgenie")

class TimeSeriesDataset(Dataset):
    def __init__(self, *vargs, **kwargs):
        self.example_shown = False
        super().__init__(*vargs, **kwargs)

    def load_examples(self, split, data_path):
        examples = []

        with open(f"{data_path}/{split}.jsonl") as f:
            lines = f.readlines()
            for line in lines:
                j = json.loads(line)
                is_reversed = j["time"][0] > j["time"][-1]
                features = j["features"]

                if is_reversed:
                    j["time"] = list(reversed(j["time"]))
                    for key in features.keys():
                        features[key] = list(reversed(features[key]))

                def make_table_lines(dims: dict[str, list], fmt: str, connector: str):
                    named_values = [
                        [fmt.format(k, v) for v in values]
                        for k, values
                        in dims.items()
                    ]
                    # [['a = 1', 'a = 2', 'a = 3'], ['b = 1', 'b = 2', 'b = 3']]
                    lines = [connector.join(nv) for nv in zip(*named_values)]
                    # ['a = 1, b = 1', 'a = 2, b = 2']
                    return lines

                def make_data_formats(time: list, features: dict[str, list]):
                    all = {"time": time, **features}

                    # Comma table will look like this:
                    #   Header:
                    #     time: open, close
                    #   Values:
                    #     2025-01-03: 200, 300
                    #     2025-01-04: 300, 400
                    values_comma_connected = make_table_lines(features, "{1}", ", ")
                    values_comma_connected = [f"{t}: {v}"
                                              for t, v
                                              in zip(time, values_comma_connected)]

                    # Line table will look like this:
                    #   Header:
                    #     | time | open | close |
                    #   Separator:
                    #     | ---- | ---- | ----- |
                    #   Values:
                    #     | 2025-01-03 | 200 | 300 |
                    #     | 2025-01-04 | 300 | 400 |
                    values_line_connected = make_table_lines(all, "{1}", " | ")
                    values_line_connected = [f"| {line} |" for line in values_line_connected]
                    line_header = "| time | " + " | ".join(features.keys()) + " |"

                    # Densely labeled will look like this:
                    #   time = 2025-01-03, open = 200, close = 300
                    #   time = 2025-01-04, open = 300, close = 400
                    densely_labeled = make_table_lines(all, "{0} = {1}", ", ")

                    return {
                        "comma-table": {
                            "header": "time: " + ", ".join(features.keys()),
                            "values": "\n".join(values_comma_connected),
                        },
                        "line-table": {
                            "header": line_header,
                            "separator": re.sub(r"[^|\s]", "-", line_header),
                            "values": "\n".join(values_line_connected),
                        },
                        "densely-labeled": "\n".join(densely_labeled),
                    }

                time = j["time"]
                first_key = list(features.keys())[0]
                first_feature = features[first_key]

                j["single-dim"] = make_data_formats(time, {first_key: first_feature})
                j["multi-dim"] = make_data_formats(time, features)
                j["extra-info"] = {
                    "start time": j["time"][0],
                    "end time": j["time"][1],
                }

                # Error: Can't serialize DataFrame
                # j["df"] = pd.DataFrame({"time": j["time"], **j["features"]})

                # DEBUG PRINT
                if not self.example_shown:
                    def tree(example, indent=1) -> str:
                        if type(example) is dict:
                            text = ""
                            for key in example.keys():
                                text += f"\n{indent * ' '}• {key}"
                                text += tree(example[key], indent + 2)
                            return text
                        elif type(example) is list:
                            text = " (list)"
                            text += tree(example[0], indent)
                            return text
                        else:
                            return ""
                    logger.info("time series dataset example structure:" + tree(j))
                    self.example_shown = True

                examples.append(j)
                # summary = self.json_to_markdown_tables(data=j)
                # examples.append(summary)

        return examples

    def render(self, example):
        desc = example["description"]
        unit = example["unit"]
        frequency = example["frequency"]
        time = example["time"]
        features = example["features"]

        df = pd.DataFrame({"time": time, **features})

        # # fig = plt.figure(figsize=(12, 8))
        # matplotlib.use("Agg")
        # fig, ax = plt.subplots()
        # ax.plot(range(len(time)), features['open'])
        # ax.set_xticks(rotation=60,
        #               ticks=range(0, len(time), 4),
        #               labels=time[0:len(time):4])

        # buf = io.BytesIO()
        # fig.savefig(buf, format='png', bbox_inches='tight')
        # plt.close(fig)
        # buf.seek(0)

        # png_base64 = base64.b64encode(buf.read()).decode('utf-8')

        # html_img = f"<img src='data:image/png;base64,{png_base64}'/>"

        # df = px.data.iris()

        fig = px.line(
            df,
            x="time",
            y=list(features.keys()),
            # color="species",
            title=desc,
            # hover_data=["time", *features.keys()]
        )

        html_img = fig.to_html(include_plotlyjs="cdn")

        html = ""

        return (
            """<div id="graph">"""
            + html_img
            + """</div><div class="root" style="margin-top: 40px">"""
            + html
            + """</div>"""
        )
        # return (
        #     """<div id="graph"></div><div class="root" style="margin-top: 40px">"""
        #     + html
        #     + """</div>
        # <script>
        #     if (typeof forecast === 'undefined') {
        #         var forecast = """
        #     + json.dumps(example)
        #     + """;
        #         // var tree = jsonview.create(forecast);
        #     } else {
        #         forecast = """
        #     + json.dumps(example)
        #     + """;
        #        // tree = jsonview.create(forecast);
        #     }

        #     // jsonview.render(tree, document.querySelector('.root'));
        #     // jsonview.expand(tree);
        #     window.meteogram = new Meteogram(forecast, 'meteogram');
        # </script>"""
        # )
