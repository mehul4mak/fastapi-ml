"""
conftest.py for fixtures
"""

# pylint: disable=E1128
import pytest

from clean_data import load_data
from ml.model import load_model


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
    return "./data/cleaned_census.csv"


@pytest.fixture(scope="session")
def df(path):
    return load_data(path)


@pytest.fixture(scope="session")
def cat_features():
    return [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]


@pytest.fixture(scope="session")
def label():
    return "salary"


@pytest.fixture(scope="session")
def encoder():
    return load_model("./model/onehotencoder.pkl")


@pytest.fixture(scope="session")
def lb():
    return load_model("./model/labelbinarizer.pkl")
