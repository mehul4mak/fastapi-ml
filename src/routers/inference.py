"""Inference API Router"""

from fastapi import APIRouter
from schema import InputData, Prediction


inference = APIRouter(prefix="/inference")


@inference.post("/predict")
async def predict(data: InputData) -> Prediction:
    """Prediction API

    Args:
        data (InputData): Input Data for ML

    Returns:
        Prediction: Prediction result from ML given inputs
    """
    _data = data.model_dump_json()

    print(_data)

    return Prediction(**{"prediction": 1.12})
