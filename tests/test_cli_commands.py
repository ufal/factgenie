"""
Run cli commands which wrap high level functionalites as a kind of functional tests.
"""

import logging
import subprocess
from pathlib import Path

from factgenie import CAMPAIGN_DIR, INPUT_DIR, OUTPUT_DIR


def test_factgenie_help():
    cmd = "factgenie --help"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logging.debug(f"RUNNING ${cmd}\n{result.stdout.decode()}")
    assert result.returncode == 0
    assert b"Usage: factgenie [OPTIONS] COMMAND [ARGS]..." in result.stdout


def test_factgenie_routes():
    """Routes list endpoints to be called with GET, POST, PUT, DELETE methods"""
    cmd = "factgenie routes"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert result.returncode == 0


# TODO by default there are no datasets in empty fatgenie data/input directory
# download some and create and test download command by that
def test_factgenie_list_datasets():
    """List datasets defined as inputs in factgenie.INPUT_DIR
    factgenie.INPUTDIR is factgenie/data/input"""
    assert Path(INPUT_DIR).exists()
    assert "factgenie/data/input" in str(INPUT_DIR), f"INPUT_DIR: {INPUT_DIR}"
    cmd = "factgenie list datasets"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logging.debug(f"RUNNING ${cmd}\n{result.stdout.decode()}")
    assert result.returncode == 0
    cli_datasets = sorted([d.strip() for d in result.stdout.decode().split("\n") if len(d.strip()) > 0])
    dir_datasets = sorted([d.name for d in Path(INPUT_DIR).iterdir() if d.is_dir()])
    logging.debug(f"CLI: {cli_datasets}")
    logging.debug(f"DIR: {dir_datasets}")
    assert cli_datasets == dir_datasets


def test_factgenie_list_outputs():
    assert Path(OUTPUT_DIR).exists()
    # Skipping the rest of the logic for now. Parsing factgenie list outputs is fragile.


def test_factgenie_list_campaigns():
    """List campaigns defined as inputs in factgenie.CAMPAIGN_DIR
    factgenie.CAMPAIGN_DIR is factgenie/data/campaigns"""
    assert Path(CAMPAIGN_DIR).exists()
    assert "factgenie/campaigns" in str(CAMPAIGN_DIR), f"CAMPAIGN_DIR: {CAMPAIGN_DIR}"
    cmd = "factgenie list campaigns"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logging.debug(f"RUNNING ${cmd}\n{result.stdout.decode()}")
    assert result.returncode == 0
    cli_campaigns = sorted([d.strip() for d in result.stdout.decode().split("\n") if len(d.strip()) > 0])
    dir_campaigns = sorted([d.name for d in Path(CAMPAIGN_DIR).iterdir() if d.is_dir()])
    logging.debug(f"CLI: {cli_campaigns}")
    logging.debug(f"DIR: {dir_campaigns}")
    assert cli_campaigns == dir_campaigns


if __name__ == "__main__":
    # test_factgenie_list_datasets()
    pass
