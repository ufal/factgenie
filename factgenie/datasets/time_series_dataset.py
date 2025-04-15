from factgenie.datasets.dataset import Dataset
import base64
import datetime
import io
import json
import matplotlib
import matplotlib.pyplot as plt
import markdown
import pandas as pd
import plotly.express as px
import textwrap


class TimeSeriesDataset(Dataset):
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

                j["all features"] = "\n".join(
                    [f"{key}: {' '.join(map(str, value))}" for key, value in features.items()]
                )
                first_key = list(features.keys())[0]
                j["first time"] = j["time"][0]
                j["last time"] = j["time"][1]
                j["first feature name"] = str(first_key)
                j["first feature values"] = " ".join(map(str, features[first_key]))

                j["df"] = pd.DataFrame({"time": j["time"], **j["features"]})

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
