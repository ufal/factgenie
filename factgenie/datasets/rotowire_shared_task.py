from factgenie.datasets.dataset import Dataset
import json
import markdown
import textwrap
import datetime


class RotowireSharedTask(Dataset):
    def load_examples(self, split, data_path):
        examples = []

        with open(f"{data_path}/{split}.jsonl") as f:
            lines = f.readlines()
            for line in lines:
                j = json.loads(line)
                summary = self.json_to_markdown_tables(data=j)
                examples.append(summary)

        return examples

    def render(self, example):
        html = markdown.markdown(example, extensions=["markdown.extensions.tables"])
        html = html.replace("<table>", '<table class="table table-hover table-sm">')
        html = self.add_explanations(html)

        return html

    def create_game_summary_table(self, data):
        home_team = f"{data['home_city']} {data['home_name']}"
        away_team = f"{data['vis_city']} {data['vis_name']}"

        summary_table = textwrap.dedent(
            f"""\
            #### Game Summary: {away_team} (away) @ {home_team} (home)
            | Team        | Quarter 1                            | Quarter 2                            | Quarter 3                            | Quarter 4                            | Final                           |
            | ----------- | ------------------------------------ | ------------------------------------ | ------------------------------------ | ------------------------------------ | ------------------------------- |
            | {away_team} | {data['vis_line']['TEAM-PTS_QTR1']}  | {data['vis_line']['TEAM-PTS_QTR2']}  | {data['vis_line']['TEAM-PTS_QTR3']}  | {data['vis_line']['TEAM-PTS_QTR4']}  | {data['vis_line']['TEAM-PTS']}  |
            | {home_team} | {data['home_line']['TEAM-PTS_QTR1']} | {data['home_line']['TEAM-PTS_QTR2']} | {data['home_line']['TEAM-PTS_QTR3']} | {data['home_line']['TEAM-PTS_QTR4']} | {data['home_line']['TEAM-PTS']} |
        """
        )
        return summary_table

    def create_team_stats_table(self, data):
        home_team = f"{data['home_city']} {data['home_name']}"
        away_team = f"{data['vis_city']} {data['vis_name']}"

        team_stats_table = textwrap.dedent(
            f"""\
            #### Team Statistics
            | Statistic                   | {away_team}                         | {home_team}                          |
            | --------------------------- | ----------------------------------- | ------------------------------------ |
            | Field Goal Percentage       | {data['vis_line']['TEAM-FG_PCT']}%  | {data['home_line']['TEAM-FG_PCT']}%  |
            | Three Point Percentage      | {data['vis_line']['TEAM-FG3_PCT']}% | {data['home_line']['TEAM-FG3_PCT']}% |
            | Free Throw Percentage       | {data['vis_line']['TEAM-FT_PCT']}%  | {data['home_line']['TEAM-FT_PCT']}%  |
            | Rebounds                    | {data['vis_line']['TEAM-REB']}      | {data['home_line']['TEAM-REB']}      |
            | Assists                     | {data['vis_line']['TEAM-AST']}      | {data['home_line']['TEAM-AST']}      |
            | Turnovers                   | {data['vis_line']['TEAM-TOV']}      | {data['home_line']['TEAM-TOV']}      |
            | Wins in the season so far   | {data['vis_line']['TEAM-WINS']}     | {data['home_line']['TEAM-WINS']}     |
            | Losses in the season so far | {data['vis_line']['TEAM-LOSSES']}   | {data['home_line']['TEAM-LOSSES']}   |
        """
        )
        return team_stats_table

    def create_player_stats_tables(self, data):
        def create_single_team_table(team_city, box_score):
            table = textwrap.dedent(
                f"""\
                #### {team_city} Player Statistics
                | Player | Minutes | Points | Rebounds | Assists | Field Goals | Three Pointers | Free Throws | Steals | Blocks | Turnovers |
                | ------ | ------- | ------ | -------- | ------- | ----------- | -------------- | ----------- | ------ | ------ | --------- |\n"""
            )

            for pid in box_score["PLAYER_NAME"].keys():
                if box_score["TEAM_CITY"][pid] == team_city and box_score["MIN"][pid] != "N/A":

                    name = f"{box_score['FIRST_NAME'][pid]} {box_score['SECOND_NAME'][pid]}"
                    fg = f"{box_score['FGM'][pid]}/{box_score['FGA'][pid]}"
                    tpt = f"{box_score['FG3M'][pid]}/{box_score['FG3A'][pid]}"
                    ft = f"{box_score['FTM'][pid]}/{box_score['FTA'][pid]}"

                    table += f"| {name} | {box_score['MIN'][pid]} | {box_score['PTS'][pid]} | "
                    table += f"{box_score['REB'][pid]} | {box_score['AST'][pid]} | "
                    table += f"{fg} | {tpt} | {ft} | "
                    table += f"{box_score['STL'][pid]} | {box_score['BLK'][pid]} | {box_score['TO'][pid]} |\n"

            return table

        home_table = create_single_team_table(data["home_city"], data["box_score"])
        away_table = create_single_team_table(data["vis_city"], data["box_score"])

        return f"{home_table}\n{away_table}"

    def json_to_markdown_tables(self, data):
        date = data["day"].split("_")
        date = datetime.date(2000 + int(date[2]), int(date[0]), int(date[1]))
        day_of_week = date.strftime("%A")
        date = date.strftime("%B %d, %Y")
        markdown = f"## NBA Game Report - {day_of_week}, {date}\n\n"
        markdown += self.create_game_summary_table(data)
        markdown += "\n"
        markdown += self.create_team_stats_table(data)
        markdown += "\n"
        markdown += self.create_player_stats_tables(data)
        return markdown

    def add_explanations(self, html):
        abbr_mappings = {
            "Minutes": "The number of minutes played",
            "Points": "Total points scored",
            "Rebounds": "Total rebounds (offensive + defensive)",
            "Assists": "Passes that directly lead to a made basket",
            "Field Goals": "Shows makes/attempts for all shots except free throws",
            "Three Pointers": "Shows makes/attempts for shots beyond the three-point line",
            "Free Throws": "Shows makes/attempts for uncontested shots awarded after a foul",
            "Steals": "Number of times the player took the ball from the opposing team",
            "Blocks": "Number of opponents' shots that were blocked",
            "Turnovers": "Number of times the player lost the ball to the opposing team",
            "Field Goal Percentage": "The percentage of shots made (excluding free throws)",
            "Three Point Percentage": "The percentage of three-point shots made",
            "Free Throw Percentage": "The percentage of free throws made",
        }

        for term, explanation in abbr_mappings.items():
            html = html.replace(term, f'<abbr title="{explanation}">{term}</abbr>')

        return html
