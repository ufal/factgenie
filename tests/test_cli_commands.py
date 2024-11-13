"""
Run cli commands which wrap high level functionalites as a kind of functional tests.
"""

import logging
import os
from pathlib import Path
import pytest
import shutil
import subprocess

from factgenie import INPUT_DIR, OUTPUT_DIR, CAMPAIGN_DIR, LLM_EVAL_CONFIG_DIR


def remove_data(inputs=True, outputs=True, campaigns=True):
    rm_dirs = []
    if inputs:
        rm_dirs.append(INPUT_DIR)
    if outputs:
        rm_dirs.append(OUTPUT_DIR)
    if campaigns:
        rm_dirs.append(CAMPAIGN_DIR)
    for d in rm_dirs:
        for p in Path(d).iterdir():
            if p.name == ".gitignore":
                continue
            if p.is_file():
                p.unlink()
            else:
                shutil.rmtree(p, ignore_errors=True)


@pytest.fixture
def remove_data_before_after():
    remove_data()
    yield
    remove_data()


@pytest.mark.usefixtures("remove_data_before_after")
def test_download_logicnlg100():
    download_logicnlg100()  # using the function as a test


def download_logicnlg100():
    """Utility function with assets to download logicnlg-100 dataset"""
    cmd = "factgenie download -d logicnlg-100"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logging.debug(f"RUNNING ${cmd}\n{result.stdout.decode()}")
    assert result.returncode == 0
    assert Path(INPUT_DIR).exists(), INPUT_DIR
    logicnlg100_dir = Path(INPUT_DIR) / "logicnlg-100"
    logicnlg100_test_dir = logicnlg100_dir / "test"
    assert logicnlg100_test_dir.exists(), logicnlg100_test_dir
    dataset_jsonl = logicnlg100_test_dir / "dataset.jsonl"

    assert Path(OUTPUT_DIR).exists(), OUTPUT_DIR
    logicnlg100_gpt4_direct_2shotcot_dir = OUTPUT_DIR / "gpt4-direct-2shotcot"
    assert logicnlg100_gpt4_direct_2shotcot_dir.exists(), logicnlg100_gpt4_direct_2shotcot_dir
    ouputs_jsonl = logicnlg100_gpt4_direct_2shotcot_dir / "gpt4-direct-2shotcot.jsonl"
    assert ouputs_jsonl.exists(), ouputs_jsonl

    assert dataset_jsonl.exists(), dataset_jsonl


@pytest.mark.usefixtures("remove_data_before_after")
def test_vllm_campaign():
    download_logicnlg100()
    vllm_config = LLM_EVAL_CONFIG_DIR / "example-vllm-llama3-eval.yaml"
    cmd = f"factgenie create_llm_campaign TEST_VLLM_CAMPAIG_{os.pidid()} -m llm_eval -d logicnlg-100 -s test --setup_ids gpt4-direct-2shotcot --config_file {vllm_config}"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


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
