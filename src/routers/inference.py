"""Inference API Router"""

from fastapi import APIRouter

from src.schema import InputData, Prediction

inference = APIRouter(prefix="/inference")


@inference.post("/predict")
async def predict(input_data: InputData) -> Prediction:
    """Prediction API

    Args:
        data (InputData): Input Data for ML

    Returns:
        Prediction: Prediction result from ML given inputs
    """
    _data = input_data.model_dump_json()

    print(_data)

    return Prediction(**{"prediction": 1.12})
