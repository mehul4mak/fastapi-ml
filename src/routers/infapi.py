"""Inference API Router"""

import os

import pandas as pd
from fastapi import APIRouter

from ml.data import process_data
from ml.model import inference, load_model
from schema import InputData, Prediction
from train_model import cat_features

infapi = APIRouter(prefix="")
model = load_model("./model/model.pkl")
encoder = load_model("./model/onehotencoder.pkl")
lb = load_model("./model/labelbinarizer.pkl")


@infapi.post("/predict")
async def predict(input_data: InputData) -> Prediction:
    """Prediction API

    Args:
        data (InputData): Input Data for ML

    Returns:
        Prediction: Prediction result from ML given inputs
    """

    df = pd.DataFrame(input_data.dict(), index=[0])

    print(df)

    X_test, y_test, _, _ = process_data(
        df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
    )

    pred = inference(model, X_test)
    pred = lb.inverse_transform(pred)[0]

    return Prediction(**{"prediction": pred})
