#!/usr/bin/env python3
import click
import os
import logging
from flask.cli import FlaskGroup
from .main import app

logger = logging.getLogger(__name__)
from factgenie.loaders import DATASET_CLASSES


def create_app(**kwargs):
    import yaml

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yml")) as f:
        config = yaml.safe_load(f)

    app.config.update(config)
    app.config["root_dir"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)

    app.db["datasets_obj"] = {}

    for dataset_name in DATASET_CLASSES.keys():
        app.db["datasets_obj"][dataset_name] = DATASET_CLASSES[dataset_name]()

    if config["debug"] is False:
        logging.getLogger("werkzeug").disabled = True

    logger.info("Application ready")

    return app


@click.group(cls=FlaskGroup, create_app=create_app)
def run():
    pass
