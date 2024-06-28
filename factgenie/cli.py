#!/usr/bin/env python3

# local imports in individual functions make CLI way faster
import click
from flask.cli import FlaskGroup


def create_app(**kwargs):
    from factgenie.loaders import DATASET_CLASSES
    import yaml
    import logging
    import coloredlogs
    import os
    from .main import app

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yml")) as f:
        config = yaml.safe_load(f)

    app.config.update(config)
    app.config["root_dir"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)

    app.db["datasets_obj"] = {}

    for dataset_name in DATASET_CLASSES.keys():
        app.db["datasets_obj"][dataset_name] = DATASET_CLASSES[dataset_name]()

    if config["debug"] is False:
        logging.getLogger("werkzeug").disabled = True

    logger = logging.getLogger(__name__)
    logger.info("Application ready")

    file_handler = logging.FileHandler("error.log")
    file_handler.setLevel(logging.ERROR)

    logging.basicConfig(
        format="%(levelname)s - %(message)s",
        level=app.config.get("logging_level", "INFO"),
        handlers=[file_handler, logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    coloredlogs.install(level=app.config.get("logging_level", "INFO"), logger=logger, fmt="%(asctime)s %(levelname)s %(message)s")

    app.config.update(SECRET_KEY=os.urandom(24))

    return app


@click.group(cls=FlaskGroup, create_app=create_app)
def run():
    pass

@click.command()
def run_llm_eval(campaign_name: str, dataset_name: str, split: str, output_name: str, llm_metric_config: str):
    from slugify import slugify
    from factgenie import utils
    from factgenie.loaders import DATASET_CLASSES

    campaign_id = slugify(campaign_name)
    campaign_data = [(dataset_name, split, output_name)]

    metric_name = ...  # ; raise NotImplementedError("load from config")
    error_categories = ... #; raise NotImplementedError("load from config")


    DATASETS = {}
    for dataset_id in DATASET_CLASSES.keys():
        DATASETS[dataset_id] = DATASET_CLASSES[dataset_id]()
    dataset = DATASETS[dataset_name]

    campaign = utils.llm_eval_new(campaign_id, metric, campaign_data, DATASETS)

    metrics_index = utils.generate_metrics_index(dataset)
    metric = metrics_index[metric_name]

    # mockup objects useful for interactivity
    threads = {"threads": {campaign_id: {"running": True}}}
    announcer = None

    return utils.run_llm_eval(campaign_id, announcer, campaign, DATASETS, metric, threads, metric_name)