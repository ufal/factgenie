#!/usr/bin/env python3

# The run.py module is CLI entry point.
# The local imports in individual functions make CLI way faster.
# Use them as much as possible and minimize imports at the top of the file.
import click
from flask.cli import FlaskGroup
from factgenie.app import app
from factgenie.campaign import CampaignMode  # required because of the click args choices


def list_datasets(app):
    """List locally available datasets."""
    from factgenie.workflows import get_local_dataset_overview

    dataset_overview = get_local_dataset_overview(app)

    for dataset_id in dataset_overview:
        print(dataset_id)


def list_downloadable(app):
    from factgenie import workflows, utils

    datasets = workflows.get_local_dataset_overview(app)

    resources = utils.load_resources_config()

    # set as `downloaded` the datasets that are already downloaded
    for dataset_id in resources.keys():
        resources[dataset_id]["downloaded"] = dataset_id in datasets

    for dataset_id, dataset_info in resources.items():
        print(f"{dataset_id} - downloaded: {dataset_info['downloaded']}")


def list_outputs(app):
    """List all available outputs."""
    from factgenie.workflows import get_model_outputs_overview

    model_outputs = get_model_outputs_overview(app, datasets=None)

    max_dataset_len = max(len(combination["dataset"]) for combination in model_outputs) + 2
    max_split_len = max(len(combination["split"]) for combination in model_outputs) + 2
    max_setup_id_len = max(len(combination["setup_id"]) for combination in model_outputs) + 2
    max_output_ids_len = max(len(str(len(combination["output_ids"]))) for combination in model_outputs) + 2

    # Print the header with computed lengths
    print(
        f"{'Dataset':>{max_dataset_len}} {'Split':>{max_split_len}} {'Setup ID':>{max_setup_id_len}} {'# Outputs':>{max_output_ids_len}}"
    )
    print("-" * (max_dataset_len + max_split_len + max_setup_id_len + max_output_ids_len + 3))

    # Print each combination with computed lengths
    for combination in model_outputs:
        print(
            f"{combination['dataset']:>{max_dataset_len}} {combination['split']:>{max_split_len}} {combination['setup_id']:>{max_setup_id_len}}"
            f" {len(combination['output_ids']):>{max_output_ids_len}}"
        )


def list_campaigns(app):
    """List all available campaigns."""
    from factgenie.workflows import get_sorted_campaign_list
    from pprint import pprint as pp

    campaigns = get_sorted_campaign_list(
        app, modes=[CampaignMode.CROWDSOURCING, CampaignMode.LLM_EVAL, CampaignMode.LLM_GEN, CampaignMode.EXTERNAL]
    )

    for campaign_id in campaigns.keys():
        print(campaign_id)


@app.cli.command("list")
@click.argument("output", type=click.Choice(["datasets", "outputs", "campaigns", "downloadable"]))
def list_data(output: str):
    """List available data."""
    if output == "datasets":
        list_datasets(app)
    elif output == "outputs":
        list_outputs(app)
    elif output == "campaigns":
        list_campaigns(app)
    elif output == "downloadable":
        list_downloadable(app)
    else:
        click.echo(list_data.get_help(click.Context(list_data)))


def show_dataset_info(app, dataset_id: str):
    """Show information about a dataset."""

    from factgenie.workflows import get_local_dataset_overview

    dataset_overview = get_local_dataset_overview(app)
    dataset_info = dataset_overview.get(dataset_id)

    if dataset_info is None:
        print(f"Dataset {dataset_id} not found.")

    print(f"{'id:':>15} {dataset_id}")

    for key, value in dataset_info.items():
        print(f"{key:>15}: {value}")


def show_campaign_info(app, campaign_id: str):
    """Show information about a campaign."""
    from factgenie.workflows import load_campaign
    from pprint import pprint as pp

    campaign = load_campaign(app, campaign_id)

    if campaign is None:
        print(f"Campaign {campaign_id} not found.")

    pp({"metadata": campaign.metadata, "stats": campaign.get_stats()})


@app.cli.command("iaa")
@click.argument("first_campaign", type=str)
@click.argument("first_ann_group", type=int)
@click.argument("second_campaign", type=str)
@click.argument("second_ann_group", type=int)
@click.argument("method", type=click.Choice(["ann_cnt_pearson", "gamma_score"]))
@click.option(
    "--gamma_score_alpha",
    default=1,
    show_default=True,
    type=float,
    help="Coefficient weighting the positional dissimilarity value for gamma score (default: 1)",
)
@click.option(
    "--gamma_score_beta",
    default=1,
    show_default=True,
    type=float,
    help="Coefficient weighting the categorical dissimilarity value for gamma score(default: 1)",
)
@click.option(
    "--gamma_score_delta",
    default=1,
    show_default=True,
    type=float,
    help="Empty dissimilarity value for gamma score (default: 1)",
)
@click.option("--gamma_score_soft", is_flag=True, default=False, help="Use soft gamma score")
@click.option("--gamma_save_plots", type=str, help="Save gamma best alignment plots to the specified directory")
@click.option(
    "--gamma_handle_empty_annotations",
    is_flag=True,
    default=False,
    help="Computes a modified gamma score that handles cases where annotations from one or both annotators are missing. Score is computed as 1 / (1 + ann_count), where ann_count is the number of annotations from the existing annotator.",
)
def compute_iaa(first_campaign, first_ann_group, second_campaign, second_ann_group, method, **args):
    """Compute inter-annotator agreement between two annotator groups."""
    from factgenie.workflows import load_campaign, generate_campaign_index
    from factgenie import analysis

    # Load campaigns
    campaigns = generate_campaign_index(app, force_reload=True)

    if first_campaign not in campaigns or second_campaign not in campaigns:
        print("Campaign not found. Available campaigns:")
        for c in campaigns:
            print(f"  {c}")
        return

    first_camp = load_campaign(app, first_campaign)
    second_camp = load_campaign(app, second_campaign)

    # Get available annotator groups
    first_groups = first_camp.db.annotator_group.unique()
    second_groups = second_camp.db.annotator_group.unique()

    if first_ann_group not in first_groups or second_ann_group not in second_groups:
        print(f"Invalid annotator group. Available groups:")
        print(f"Campaign {first_campaign}: {first_groups}")
        print(f"Campaign {second_campaign}: {second_groups}")
        return

    # Find common examples
    combinations = analysis.get_common_examples(first_camp.db, second_camp.db, first_ann_group, second_ann_group)

    if not combinations:
        print("No common examples found between the selected annotator groups")
        return

    selected_campaigns = [first_campaign, second_campaign]

    dfs = analysis.compute_iaa_dfs(app, selected_campaigns, combinations, campaigns)

    dataset_level_counts = dfs["dataset_level_counts"]
    example_level_counts = dfs["example_level_counts"]
    span_index = dfs["span_index"]

    first_group_id = analysis.format_group_id(first_campaign, first_ann_group)
    second_group_id = analysis.format_group_id(second_campaign, second_ann_group)

    # filter the selected annotator groups
    for df in [dataset_level_counts, example_level_counts, span_index]:
        df = df[df["annotator_group_id"].isin([first_group_id, second_group_id])]
        df.reset_index(drop=True, inplace=True)

    if method == "ann_cnt_pearson":
        # Compute Pearson correlation for both levels
        for level, counts_df in [("example", example_level_counts), ("dataset", dataset_level_counts)]:
            correlations = analysis.compute_pearson_r(counts_df, first_group_id, second_group_id)

            print(f"\n{level.title()}-level correlations between {first_group_id} and {second_group_id}")
            print("==============================================")
            print(f"Micro Pearson-r: {correlations['micro']:.3f}")
            print("==============================================")

            for i, corr in enumerate(correlations["category_correlations"]):
                print(f"Category {i}: {corr:.3f}")
            print("----------------------------------------------")
            print(f"Macro Pearson-r: {correlations['macro']:.3f}")
            print("==============================================")

    elif method == "gamma_score":
        gamma = analysis.compute_gamma_score(
            span_index,
            example_level_counts,
            alpha=args["gamma_score_alpha"],
            beta=args["gamma_score_beta"],
            delta_empty=args["gamma_score_delta"],
            soft=args["gamma_score_soft"],
            save_plots=args["gamma_save_plots"],
            handle_empty_annotations=args["gamma_handle_empty_annotations"],
        )

        print("==============================================")
        print(f"Gamma score: {gamma:.3f}")
        print("==============================================")


@app.cli.command("info")
@click.option("-d", "--dataset", type=str, help="Show information about a dataset.")
@click.option("-c", "--campaign", type=str, help="Show information about a campaign.")
def info(dataset: str, campaign: str):
    """Show information about a dataset or campaign."""
    if dataset:
        show_dataset_info(app, dataset)
    elif campaign:
        show_campaign_info(app, campaign)
    else:
        click.echo(info.get_help(click.Context(info)))


@app.cli.command("download")
@click.option(
    "-d",
    "--dataset_id",
    type=str,
    help=(
        "Download dataset input data. "
        "Factgenie does not use references so the inputs define the datasets. "
        "If the dataset class defines model outputs and annotations we download them too."
    ),
)
def download_data(dataset_id: str):
    import factgenie.workflows as workflows

    if dataset_id:
        workflows.download_dataset(app, dataset_id)
    else:
        click.echo(info.get_help(click.Context(info)))


@app.cli.command("create_llm_campaign")
@click.argument(
    "campaign_id",
    type=str,
)
@click.option("-d", "--dataset_ids", required=True, type=str, help="Comma separated dataset identifiers.")
@click.option("-s", "--splits", required=True, type=str, help="Comma separated setups.")
@click.option("-o", "--setup_ids", type=str, help="Comma separated setup ids.")
@click.option("-m", "--mode", required=True, type=click.Choice([CampaignMode.LLM_EVAL, CampaignMode.LLM_GEN]))
@click.option(
    "-c",
    "--config_file",
    required=True,
    type=str,
    help="Path to the YAML configuration file  / name of an existing config (without file suffix).",
)
@click.option("-f", "--overwrite", is_flag=True, default=False, help="Overwrite existing campaign if it exists.")
def create_llm_campaign(
    campaign_id: str, dataset_ids: str, splits: str, setup_ids: str, mode: str, config_file: str, overwrite: bool
):
    """Create a new LLM campaign."""
    import yaml
    from slugify import slugify
    from factgenie.workflows import load_campaign, get_sorted_campaign_list
    from factgenie import workflows, llm_campaign
    from pathlib import Path
    from pprint import pprint as pp

    if mode == CampaignMode.LLM_EVAL and not setup_ids:
        raise ValueError("The `setup_id` argument is required for llm_eval mode.")

    campaigns = get_sorted_campaign_list(
        app, modes=[CampaignMode.CROWDSOURCING, CampaignMode.LLM_EVAL, CampaignMode.LLM_GEN, CampaignMode.EXTERNAL]
    )
    if campaign_id in campaigns and not overwrite:
        raise ValueError(f"Campaign {campaign_id} already exists. Use --overwrite to overwrite.")

    campaign_id = slugify(campaign_id)
    datasets = app.db["datasets_obj"]
    dataset_ids = dataset_ids.split(",")
    splits = splits.split(",")

    if setup_ids:
        setup_ids = setup_ids.split(",")

    if mode == CampaignMode.LLM_EVAL:
        combinations = [
            (dataset_id, split, setup_id) for dataset_id in dataset_ids for split in splits for setup_id in setup_ids
        ]
        dataset_overview = workflows.get_local_dataset_overview(app)
        available_data = workflows.get_model_outputs_overview(app, dataset_overview)
    elif mode == CampaignMode.LLM_GEN:
        combinations = [(dataset_id, split) for dataset_id in dataset_ids for split in splits]
        dataset_overview = workflows.get_local_dataset_overview(app)

        available_data = workflows.get_available_data(app, dataset_overview)

    # drop the `output_ids` key from the available_data
    campaign_data = []

    for c in combinations:
        for data in available_data:
            if (
                c[0] == data["dataset"]
                and c[1] == data["split"]
                and (mode == CampaignMode.LLM_GEN or c[2] == data["setup_id"])
            ):
                data.pop("output_ids")
                campaign_data.append(data)

    if not campaign_data:
        raise ValueError("No valid data combinations found.")

    print(f"Available data combinations:")
    pp(campaign_data)
    print("-" * 80)
    print()

    # if config_file is a path, load the config from the path
    if Path(config_file).exists():
        with open(config_file) as f:
            config = yaml.safe_load(f)
    else:
        if not config_file.endswith(".yaml"):
            config_file = f"{config_file}.yaml"

        configs = workflows.load_configs(mode)
        config = configs.get(config_file)

    if not config:
        config_names = [Path(x).stem for x in configs.keys()]
        raise ValueError(f"Config {config_file} not found. Available configs: {config_names}")

    llm_campaign.create_llm_campaign(app, mode, campaign_id, config, campaign_data, datasets, overwrite=overwrite)

    print(f"Created campaign {campaign_id}")


@app.cli.command("run_llm_campaign")
@click.argument("campaign_id", type=str)
def run_llm_campaign(campaign_id: str):
    """
    Run a LLM campaign by id.
    """
    from factgenie.models import ModelFactory
    from factgenie import llm_campaign
    from factgenie.campaign import CampaignStatus
    from factgenie.workflows import load_campaign

    # mockup object
    announcer = None

    datasets = app.db["datasets_obj"]
    campaign = load_campaign(app, campaign_id)

    if campaign is None:
        raise ValueError(f"Campaign {campaign_id} not found.")

    if campaign.metadata["status"] == CampaignStatus.FINISHED:
        raise ValueError(f"Campaign {campaign_id} is already finished.")

    if campaign.metadata["status"] == CampaignStatus.RUNNING:
        raise ValueError(f"Campaign {campaign_id} is already running.")

    config = campaign.metadata["config"]
    mode = campaign.metadata["mode"]
    model = ModelFactory.from_config(config, mode=mode)
    running_campaigns = app.db["running_campaigns"]

    app.db["running_campaigns"].add(campaign_id)

    return llm_campaign.run_llm_campaign(
        app, mode, campaign_id, announcer, campaign, datasets, model, running_campaigns
    )


@app.cli.command("save_generated_outputs")
@click.argument("campaign_id", type=str)
@click.argument("setup_id", type=str)
def save_generated_outputs(campaign_id: str, setup_id: str):
    """
    Save outputs generated from a campaign under a specific setup ID.

    Args:
        campaign_id: The ID of the campaign containing the generated outputs
        setup_id: The desired setup ID under which to save the outputs
    """
    from factgenie import llm_campaign

    result = llm_campaign.save_generation_outputs(app, campaign_id, setup_id)

    if result.status == "200 OK":
        print(f"Successfully saved outputs from campaign {campaign_id} under setup ID {setup_id}")
    else:
        print(f"Error saving outputs: {result}")


def setup_logging(config):
    import logging
    import coloredlogs
    import os
    import re
    from datetime import datetime

    from factgenie import ROOT_DIR

    class PlainTextFormatter(logging.Formatter):
        def format(self, record):
            msg = super().format(record)
            return remove_ansi_codes(msg)

    def remove_ansi_codes(text):
        """Removes ANSI escape sequences from text."""
        import re

        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        return ansi_escape.sub("", text)

    os.makedirs(f"{ROOT_DIR}/logs", exist_ok=True)
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Define log format
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    # Create loggers
    logger = logging.getLogger("factgenie")

    # Get logging level from config
    logging_level = config.get("logging", {}).get("level", "INFO")
    logger.setLevel(logging_level)

    # File handler for errors and warnings
    error_handler = logging.FileHandler(f"{ROOT_DIR}/logs/{datetime_str}_error.log")
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(PlainTextFormatter(log_format))

    # File handler for info messages only
    info_handler = logging.FileHandler(f"{ROOT_DIR}/logs/{datetime_str}_info.log")
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(lambda record: record.levelno == logging.INFO)
    info_handler.setFormatter(PlainTextFormatter(log_format))

    # # Console handler with colored output
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging_level)
    coloredlogs.install(level=logging_level, logger=logger, fmt=log_format)
    # console_handler.setFormatter(logging.Formatter(log_format))  # ANSI codes preserved

    # Add handlers to logger
    logger.addHandler(error_handler)
    logger.addHandler(info_handler)
    # logger.addHandler(console_handler)

    return logger


def create_app(**kwargs):
    import yaml
    import os
    import shutil
    import logging
    import factgenie.workflows as workflows
    from apscheduler.schedulers.background import BackgroundScheduler
    from datetime import datetime
    from factgenie.utils import check_login
    from factgenie import ROOT_DIR, MAIN_CONFIG_PATH, MAIN_CONFIG_TEMPLATE_PATH, CAMPAIGN_DIR, INPUT_DIR, OUTPUT_DIR

    if not MAIN_CONFIG_PATH.exists():
        print("Activating the default configuration.")
        shutil.copy(MAIN_CONFIG_TEMPLATE_PATH, MAIN_CONFIG_PATH)

    with open(MAIN_CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    logger = setup_logging(config)

    config["host_prefix"] = os.getenv("FACTGENIE_HOST_PREFIX", config["host_prefix"])
    config["login"]["active"] = os.getenv("FACTGENIE_LOGIN_ACTIVE", config["login"]["active"])
    config["login"]["lock_view_pages"] = os.getenv(
        "FACTGENIE_LOCK_VIEW_PAGES", config["login"].get("lock_view_pages", True)
    )
    config["login"]["username"] = os.getenv("FACTGENIE_LOGIN_USERNAME", config["login"]["username"])
    config["login"]["password"] = os.getenv("FACTGENIE_LOGIN_PASSWORD", config["login"]["password"])

    for service, key in config.get("api_keys", {}).items():
        if key != "":
            os.environ[service] = key

    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    app.config["root_dir"] = ROOT_DIR
    app.config.update(config)

    assert check_login(
        app, config["login"]["username"], config["login"]["password"]
    ), "Login should pass for valid user"
    assert not check_login(app, "dummy_non_user_name", "dummy_bad_password"), "Login should fail for dummy user"

    app.db["datasets_obj"] = workflows.instantiate_datasets()
    app.db["scheduler"] = BackgroundScheduler()

    logging.getLogger("apscheduler.scheduler").setLevel(logging.WARNING)
    logging.getLogger("apscheduler.executors.default").setLevel(logging.WARNING)
    app.db["scheduler"].start()

    workflows.generate_campaign_index(app)

    if config.get("logging", {}).get("flask_debug", False) is False:
        logging.getLogger("werkzeug").disabled = True

    logger.info("Application ready")
    app.config.update(SECRET_KEY=os.urandom(24))

    return app


@click.group(cls=FlaskGroup, create_app=create_app)
def run():
    pass


if __name__ == "__main__":
    app = create_app()
    app.run(debug=False)
