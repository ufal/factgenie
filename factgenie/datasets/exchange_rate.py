#!/usr/bin/env python3
import logging
import json
import dateutil.parser
from factgenie.datasets.basic import JSONDataset
from pathlib import Path

logger = logging.getLogger(__name__)


class ExchangeRate(JSONDataset):
    def postprocess_data(self, examples):
        """Convert JSON examples to CSV files with metadata."""
        for i, example in enumerate(examples):
            from_symbol = example["from_symbol"]
            to_symbol = example["to_symbol"]

            # Create CSV content with metadata and data
            csv_lines = [
                f"# From Symbol: {from_symbol}",
                f"# To Symbol: {to_symbol}",
                f"# Type: {example['type']}",
                f"# UTC Offset (sec): {example['utc_offset_sec']}",
                f"# Interval (sec): {example['interval_sec']}",
                "timestamp,exchange_rate",
            ]

            # Add time series data
            for timestamp, values in example["time_series"].items():
                csv_lines.append(f"{timestamp},{values['exchange_rate']}")

            examples[i] = "\n".join(csv_lines)

        return examples

    def render(self, example):
        """Render exchange rate chart using CSV data."""
        lines = example.split("\n")

        # Parse metadata
        metadata = {}
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith("#"):
                key, value = line[2:].split(":", 1)
                metadata[key.strip()] = value.strip()
                data_start = i + 1
            elif line.startswith("timestamp"):
                data_start = i + 1
                break

        # Parse data
        data = []
        for line in lines[data_start:]:
            if line.strip():
                timestamp, rate = line.strip().split(",")
                date = dateutil.parser.parse(timestamp)
                date_ms = int(date.timestamp() * 1000)
                data.append([date_ms, float(rate)])

        # Sort data by timestamp
        data.sort(key=lambda x: x[0])

        from_symbol = metadata["From Symbol"]
        to_symbol = metadata["To Symbol"]

        # Rest of the Highcharts configuration remains same
        return f"""
        <div id="chartPlaceholder"></div>
        <script>
        if (typeof chartData === 'undefined') {{
            var chartData = {json.dumps(data)};
        }} else {{
            chartData = {json.dumps(data)};
        }}
        Highcharts.chart('chartPlaceholder', {{
            chart: {{
                zooming: {{
                    enabled: true
                }},
                animation: false
            }},
            credits: {{
                enabled: false
            }},
            title: {{
                text: '{from_symbol}/{to_symbol} Exchange Rate',
                align: 'left'
            }},
            subtitle: {{
                text: 'Currency exchange rate over time',
                align: 'left'
            }},
            xAxis: {{
                type: 'datetime'
            }},
            yAxis: {{
                title: {{
                    text: '{to_symbol} per 1 {from_symbol}'
                }},
                labels: {{
                    format: '{{value:.4f}}'
                }}
            }},
            legend: {{
                enabled: false
            }},
            plotOptions: {{
                area: {{
                    color: '#2f7ed8',
                    fillColor: '#ebf2fa',
                    marker: {{
                        radius: 2,
                        fillColor: '#2f7ed8'
                    }},
                    lineWidth: 1,
                    tooltip: {{
                        pointFormat: '<b>{{point.y:.4f}} {to_symbol}</b><br/>',
                        dateTimeLabelFormats: {{
                            day: '%Y-%m-%d',
                            hour: '%Y-%m-%d %H:%M'
                        }}
                    }},
                    states: {{
                        hover: {{
                            lineWidth: 1
                        }}
                    }},
                    threshold: null
                }}
            }},
            series: [{{
                type: 'area',
                name: '{from_symbol}/{to_symbol}',
                data: chartData,
                animation: false
            }}]
        }});
        </script>
        """
