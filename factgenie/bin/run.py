#!/usr/bin/env python3

# The run.py module is CLI entry point.
# The local imports in individual functions make CLI way faster.
# Use them as much as possible and minimize imports at the top of the file.
import click
from flask.cli import FlaskGroup
from factgenie.app import app
from factgenie.campaigns import CampaignMode  # required because of the click args choices


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
    setup_ids = setup_ids.split(",")

    combinations = [
        (dataset_id, split, setup_id) for dataset_id in dataset_ids for split in splits for setup_id in setup_ids
    ]
    dataset_overview = workflows.get_local_dataset_overview(app)
    if mode == CampaignMode.LLM_EVAL:
        available_data = workflows.get_model_outputs_overview(app, dataset_overview)
    elif mode == CampaignMode.LLM_GEN:
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
        config_file = workflows.load_config_from_path(config_file)
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
    from factgenie.models import ModelFactory
    from factgenie import llm_campaign
    from factgenie.campaigns import CampaignStatus
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


def create_app(**kwargs):
    import yaml
    import logging
    import coloredlogs
    import os
    import factgenie.workflows as workflows
    from apscheduler.schedulers.background import BackgroundScheduler
    from factgenie.utils import check_login
    from factgenie import ROOT_DIR, MAIN_CONFIG_PATH, CAMPAIGN_DIR, INPUT_DIR, OUTPUT_DIR

    logger = logging.getLogger(__name__)

    file_handler = logging.FileHandler("error.log")
    file_handler.setLevel(logging.ERROR)

    with open(MAIN_CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    logging_level = config.get("logging", {}).get("level", "INFO")
    logging.basicConfig(
        format="%(levelname)s (%(filename)s:%(lineno)d) - %(message)s",
        level=logging_level,
        handlers=[file_handler, logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        level=logging_level,
        logger=logger,
        fmt="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
    )

    config["host_prefix"] = os.getenv("FACTGENIE_HOST_PREFIX", config["host_prefix"])
    config["login"]["active"] = os.getenv("FACTGENIE_LOGIN_ACTIVE", config["login"]["active"])
    config["login"]["username"] = os.getenv("FACTGENIE_LOGIN_USERNAME", config["login"]["username"])
    config["login"]["password"] = os.getenv("FACTGENIE_LOGIN_PASSWORD", config["login"]["password"])

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
