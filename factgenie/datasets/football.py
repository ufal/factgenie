#!/usr/bin/env python3
import logging
from datetime import datetime
from tinyhtml import h
from factgenie.datasets.basic import JSONDataset

logger = logging.getLogger(__name__)


class Football(JSONDataset):
    def postprocess_data(self, examples):
        for match in examples:
            # Move fixture data to top level
            for key in ["date", "timestamp", "referee", "venue", "status"]:
                if key in match.get("fixture", {}):
                    match[key] = match["fixture"][key]

            # Convert timestamp to readable date
            match["datetime"] = datetime.fromtimestamp(match["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")

            # Store venue data as tuple of (name, city)
            if "venue" in match and match["venue"]:
                match["venue_data"] = {"name": match["venue"].get("name", ""), "city": match["venue"].get("city", "")}

            # Pop unnecessary keys
            match.pop("timestamp", None)
            match.pop("venue", None)
            match.pop("fixture", None)
            match.pop("date", None)

            for event in match.get("events", []):
                if "team" in event:
                    event["team"] = event["team"].get("name", "")
                if "player" in event:
                    event["player"] = event["player"].get("name", "")
                if "assist" in event and event["assist"]:
                    event["assist"] = event["assist"].get("name", "")
                else:
                    event["assist"] = "-"

                # Format time with extra time if present
                if "time" in event:
                    elapsed = event["time"].get("elapsed", "")
                    extra = event["time"].get("extra", "")
                    event["time"] = f"{elapsed}'{' +' + str(extra) if extra else ''}"

        return examples

    def render(self, example):
        # Flag images ("logo" - league, "flag" - country), set max width to 50px
        league_flag = h("img", src=example["league"]["logo"], klass="mr-2", style="max-width: 50px")
        flag_el = h("div")(league_flag)

        # League info
        league_info = f"{example['league']['name']} ({example['league']['country']}) | {example['league']['round']}"
        league_el = h("div", klass="h5 mb-2")(league_info)

        # Teams and score
        home_flag = h("img", src=example["teams"]["home"]["logo"], klass="inline-icon")
        away_flag = h("img", src=example["teams"]["away"]["logo"], klass="inline-icon")
        home_name = h("span")(home_flag, example["teams"]["home"]["name"] + " - ")
        away_name = h("span")(away_flag, example["teams"]["away"]["name"])
        teams_el = h("div", klass="h3 mb-3")(home_name, away_name)

        score = example["score"]["fulltime"]
        teams_score = f"Final score: {score['home']} - {score['away']}"
        teams_score_el = h("div", klass="h4 mb-3")(teams_score)

        # Match details
        venue_name = example["venue_data"]["name"]
        venue_city = example["venue_data"]["city"]
        match_info = f"{example['datetime']} at {venue_name}, {venue_city}"
        header_el = h("div", klass="mb-2")(match_info)

        # Rest of the render method stays the same
        status = (
            f"{example['status']['long']} "
            f"({example['status']['elapsed']}'{' +' + str(example['status']['extra']) if example['status']['extra'] else ''})"
        )
        status_el = h("div", klass="mb-2")(h("b")("Status: "), status)
        referee_el = h("div", klass="mb-3")(h("b")("Referee: "), f"{example['referee']}")

        # Events table
        event_headers = ["Time", "Team", "Player", "Type", "Detail", "Assist / Substitute", "Comments"]
        thead_tr = h("tr")(*(h("th")(header) for header in event_headers))

        event_rows = []
        for event in example["events"]:
            row = h("tr")(
                h("td")(event["time"]),
                h("td")(event["team"]),
                h("td")(event["player"]),
                h("td")(event["type"]),
                h("td")(event["detail"]),
                h("td")(event["assist"]),
                h("td")(event["comments"]),
            )
            event_rows.append(row)

        events_table = h("table", klass="table table-sm table-bordered table-striped")(
            h("thead")(thead_tr), h("tbody")(event_rows)
        )

        # Combine all elements
        html_el = h("div")(
            flag_el,
            league_el,
            teams_el,
            teams_score_el,
            header_el,
            status_el,
            referee_el,
            events_table,
        )

        return html_el.render()
