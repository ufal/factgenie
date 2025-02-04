#!/usr/bin/env python3
import logging

logger = logging.getLogger("factgenie")

import json
from factgenie.datasets.basic import JSONDataset
from factgenie.datasets.quintd import QuintdDataset
import dateutil.parser
from datetime import datetime


class OpenWeather(QuintdDataset, JSONDataset):
    def postprocess_data(self, examples):
        forecasts = examples["forecasts"]
        examples = []

        # using metric system
        units = {
            "temp": "°C",
            "wind": "m/s",
            "pressure": "hPa",
            "rain": "mm",
            "snow": "mm",
        }

        for forecast in forecasts:
            city = forecast["city"]
            lst_filtered = []
            timezone_shift_sec = city["timezone"]

            for key in ["sunrise", "sunset", "population", "timezone", "coord", "id"]:
                city.pop(key, None)

            for i, f in enumerate(forecast["list"]):
                # 6-hour intervals
                if i % 2 != 0:
                    continue
                f = {k: v for k, v in f.items() if k not in ["dt", "pop", "sys", "visibility"]}

                # remove extra keys
                f["main"] = {
                    k: v
                    for k, v in f["main"].items()
                    if k not in ["temp_kf", "humidity", "sea_level", "grnd_level", "temp_max", "temp_min"]
                }

                # convert "dt_txt" to timestamp
                local_datetime = dateutil.parser.parse(f["dt_txt"])
                local_datetime = local_datetime.timestamp()
                # shift timezone
                local_datetime += timezone_shift_sec
                # convert back to "2023-11-28 09:00:00"
                f["dt_txt"] = datetime.fromtimestamp(local_datetime).strftime("%Y-%m-%d %H:%M:%S")
                f["day_of_week"] = datetime.fromtimestamp(local_datetime).strftime("%A")

                lst_filtered.append(f)

            examples.append({"city": city, "units": units, "list": lst_filtered})

        return examples

    def render(self, example):
        html = ""

        return (
            """<div id="meteogram"></div><div class="root" style="margin-top: 40px">"""
            + html
            + """</div>
        <script>
            if (typeof forecast === 'undefined') {
                var forecast = """
            + json.dumps(example)
            + """;
                // var tree = jsonview.create(forecast);
            } else {
                forecast = """
            + json.dumps(example)
            + """;
               // tree = jsonview.create(forecast);
            }
            
            // jsonview.render(tree, document.querySelector('.root'));
            // jsonview.expand(tree);
            window.meteogram = new Meteogram(forecast, 'meteogram');
        </script>"""
        )
