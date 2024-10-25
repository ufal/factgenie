#!/usr/bin/env python3

# The cli is CLI entry point.
# The local imports in individual functions make CLI way faster.
# Use them as much as possible and minimize imports at the top of the file.
import click

from flask.cli import FlaskGroup
from factgenie.campaigns import CampaignMode
from factgenie.utils import load_dataset_config


@click.command()
def list_datasets():
    config = load_dataset_config()
    for dataset_id, _ in config.items():
        print(dataset_id)


@click.command()
@click.option("--campaign_id", required=True, type=str)
@click.option("--dataset_id", required=True, type=str)
@click.option("--split", required=True, type=str)
@click.option("--setup_id", type=str)
@click.option("--mode", required=True, type=click.Choice([CampaignMode.LLM_EVAL, CampaignMode.LLM_GEN]))
@click.option(
    "--llm_metric_config", required=True, type=str, help="Path to the metric config file or just the metric name."
)
@click.option("--overwrite", is_flag=True, default=False, help="Remove existing campaign if it exists.")
def run_llm_campaign(
    campaign_id: str, dataset_id: str, split: str, setup_id: str, mode: str, llm_metric_config: str, overwrite: bool
):
    """Runs the LLM campaign from CLI with no web server."""
    from slugify import slugify
    from factgenie import workflows, llm_campaign
    from factgenie.models import ModelFactory

    campaign_id = slugify(campaign_id)
    campaign_data = [{"dataset": dataset_id, "split": split, "setup_id": setup_id}]

    config = load_dataset_config()
    dataset_config = config[dataset_id]
    datasets = {dataset_id: workflows.instantiate_dataset(dataset_id, dataset_config)}

    if mode == CampaignMode.LLM_EVAL and not setup_id:
        raise ValueError("The `setup_id` argument is required for llm_eval mode.")

    configs = workflows.load_configs(mode)
    metric_config = configs[llm_metric_config]
    campaign = llm_campaign.create_llm_campaign(
        mode, campaign_id, metric_config, campaign_data, datasets, overwrite=overwrite
    )

    # mockup objects
    announcer = None
    running_campaigns = {campaign_id}

    model = ModelFactory.from_config(metric_config, mode=mode)

    return llm_campaign.run_llm_campaign(mode, campaign_id, announcer, campaign, datasets, model, running_campaigns)


def create_app(**kwargs):
    import yaml
    import logging
    import coloredlogs
    import os
    from factgenie.main import app
    from apscheduler.schedulers.background import BackgroundScheduler

    logger = logging.getLogger(__name__)

    file_handler = logging.FileHandler("error.log")
    file_handler.setLevel(logging.ERROR)

    logging.basicConfig(
        format="%(levelname)s (%(filename)s:%(lineno)d) - %(message)s",
        level=app.config.get("logging_level", "INFO"),
        handlers=[file_handler, logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        level=app.config.get("logging_level", "INFO"),
        logger=logger,
        fmt="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
    )

    from factgenie import ROOT_DIR, MAIN_CONFIG_PATH, CAMPAIGN_DIR, INPUT_DIR, OUTPUT_DIR
    import factgenie.workflows as workflows
    from factgenie.utils import check_login

    if not MAIN_CONFIG_PATH.exists():
        raise ValueError(
            f"Invalid path to config.yml {MAIN_CONFIG_PATH=}. "
            "Please copy config_TEMPLATE.yml to config.yml "
            "and change the password, update the host prefix, etc."
        )

    with open(MAIN_CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    app.config.update(config)
    app.config["root_dir"] = ROOT_DIR

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

    if config["debug"] is False:
        logging.getLogger("werkzeug").disabled = True

    logger.info("Application ready")

    app.config.update(SECRET_KEY=os.urandom(24))

    # register CLI commands
    app.cli.add_command(run_llm_campaign)
    app.cli.add_command(list_datasets)

    return app


@click.group(cls=FlaskGroup, create_app=create_app)
def run():
    pass
