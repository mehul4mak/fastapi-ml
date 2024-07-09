import pytest

from clean_data import load_data


@pytest.fixture(scope="session")
def df():
    return load_data("src/data/cleaned_census.csv")
