"""
conftest.py for fixtures
"""

# pylint: disable=E1128
import pytest
import yaml

from clean_data import load_data
from ml.model import load_model

with open("config.yaml", "rb") as f:
    config = yaml.safe_load(f)


def df_plugin():
    """
    This is pytest fixture boilerplate code.
    """
    return None


# Creating a Dataframe object 'pytest.df' in Namespace
def pytest_configure():
    """
    pytest configure -> This is pytest fixture boilerplate code.
    """
    pytest.df = df_plugin()


@pytest.fixture(scope="session")
def path():
    """
    pytest fixture for path
    """
    return config["CLEANED_DATA_PATH"]


@pytest.fixture(scope="session")
def df(path):
    return load_data(path)


@pytest.fixture(scope="session")
def cat_features():
    return config["CAT_FEATURES"]


@pytest.fixture(scope="session")
def label():
    return config["LABEL"]


@pytest.fixture(scope="session")
def encoder():
    return load_model(config["OHE_PATH"])


@pytest.fixture(scope="session")
def lb():
    return load_model(config["LB_PATH"])
