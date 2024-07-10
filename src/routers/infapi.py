"""Inference API Router"""

import pandas as pd
import yaml
from fastapi import APIRouter

from ml.data import process_data
from ml.model import inference, load_model
from schema import InputData, Prediction

# pylint: disable=C0103

with open("./config.yaml", "rb") as f:
    config = yaml.safe_load(f)


model = load_model(config["MODEL_PATH"])
encoder = load_model(config["OHE_PATH"])
lb = load_model(config["LB_PATH"])


infapi = APIRouter(prefix="")


@infapi.post("/predict")
async def predict(input_data: InputData) -> Prediction:
    """Prediction API

    Args:
        data (InputData): Input Data for ML

    Returns:
        Prediction: Prediction result from ML given inputs
    """

    df = pd.DataFrame(input_data.dict(), index=[0])

    X_test, _, _, _ = process_data(
        df,
        categorical_features=config["CAT_FEATURES"],
        training=False,
        encoder=encoder,
    )

    pred = inference(model, X_test)
    pred = lb.inverse_transform(pred)[0]

    return Prediction(**{"prediction": pred})
