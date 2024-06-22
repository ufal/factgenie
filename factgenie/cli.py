#!/usr/bin/env python3
import coloredlogs
import click
import os
import yaml
import logging
from flask.cli import FlaskGroup

from factgenie.loaders import DATASET_CLASSES

from .main import app


def create_app(**kwargs):

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
