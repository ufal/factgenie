#!/usr/bin/env python3
import logging

logger = logging.getLogger("factgenie")

import json
from pathlib import Path

import dateutil.parser

from factgenie.datasets.dataset import Dataset
from factgenie.datasets.quintd import QuintdDataset


class OurWorldInData(QuintdDataset, Dataset):
    def load_examples(self, split, data_path):
        examples = []
        split_dir = Path(f"{data_path}/{split}")
        filenames = sorted(split_dir.iterdir(), key=lambda x: int(x.stem.split("-")[0]))

        for filename in filenames:
            with open(filename) as f:
                examples.append(f.read())

        return examples

    def render(self, example):
        # parse the csv comments, e.g. 'country: country_name' as Python dict
        lines_starting_with_hash = [
            line[1:].strip().split(": ") for line in example.split("\n") if line.startswith("#")
        ]

        metadata = {k: v for k, v in lines_starting_with_hash}

        title = metadata["title"]
        description = metadata["description"]
        country = metadata["country"]
        unit = metadata["unit"]
        data = []
        csv_lines = [line for line in example.split("\n") if not line.startswith("#")]
        # unit = csv_lines[0].split(",")[1]

        for row in csv_lines:
            if not row or len(row.split(",")) != 2 or "date" in row:
                continue
            date, value = row.split(",")
            # convert date to timestamp
            date = dateutil.parser.parse(date)
            date = date.timestamp() * 1000

            data.append([int(date), float(value)])

        # data per year vs. data per day
        date_format = "%Y" if title in ["Deaths in under-fives", "Life expectancy at birth"] else "%Y-%m-%d"

        return (
            """
        <div id="chartPlaceholder"></div>
        <script>
        if (typeof chartData === 'undefined') {
            var chartData = """
            + json.dumps(data)
            + """;
        } else {
            chartData = """
            + json.dumps(data)
            + """;
        }
        Highcharts.chart('chartPlaceholder', {
            chart: {
                zooming : {
                    enabled: false
                },
                animation: false,
                credits: {
                    enabled: false
                }
            },
            credits: {
                enabled: false
            },
            title: {
                text: '"""
            + f"{country}"
            + """',
                align: 'left'
            },
            subtitle: {
                text: '"""
            + f"{title}. {description}"
            + """',
                align: 'left'
            },
            xAxis: {
                type: 'datetime'
            },
            yAxis: {
                title: {
                    text: '"""
            + unit
            + """'
                }
            },
            legend: {
                enabled: false
            },
            plotOptions: {
                area: {
                    color: '#a6a6a6',
                    fillColor: '#f2f2f2',
                    marker: {
                        radius: 2,
                        fillColor: '#a6a6a6'
                    },
                    lineWidth: 1,
                    tooltip: {
                        dateTimeLabelFormats: {
                            hour: '"""
            + date_format
            + """',
                        }
                    },
                    states: {
                        hover: {
                            lineWidth: 1
                        }
                    },
                    threshold: null
                }
            },

            series: [{
                type: 'area',
                name: '"""
            + country
            + """',
                data: chartData,
                animation: false
            }]
        });
        </script>
        """
        )
