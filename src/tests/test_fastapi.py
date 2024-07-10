from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_read_root_get():
    response = client.get("/")
    assert response.status_code == 200


def test_read_root_content():
    response = client.get("/")

    assert response.content == b'"Welcome!"'


def test_predict_negative():
    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    }

    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.content == b'{"prediction":"<=50K"}'


def test_predict_positve():

    data = {
        "workclass": "State-gov",
        "education": "Bachelors",
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "native_country": "United-States",
        "age": 39,
        "fnlgt": 77516,
        "education_num": 13,
        "capital_gain": 10000,
        "capital_loss": 0,
        "hours_per_week": 40,
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.content == b'{"prediction":">50K"}'
