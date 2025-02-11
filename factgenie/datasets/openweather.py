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
            examples.append({"city": city, "units": units, "list": forecast["list"]})

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
