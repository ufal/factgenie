# conftest.py
import shutil

import pytest

from factgenie import MAIN_CONFIG_PATH


@pytest.fixture(scope="module", autouse=True)
def prepare_testing_config():
    # copy config_TEMPLATE.yml to config.yml
    shutil.copy(MAIN_CONFIG_PATH.with_name("config_TEMPLATE.yml"), MAIN_CONFIG_PATH)
    assert MAIN_CONFIG_PATH.exists()
