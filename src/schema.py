"""
Pydantic Models for API Input and Output

This module defines Pydantic models for validating API input data and
output predictions.
"""

from pydantic import BaseModel, Field


class InputData(BaseModel):
    """
    Input Data Schema for Machine Learning Prediction API

    Attributes:
    -----------
    age : int
        Age of the individual.
    workclass : str
        Type of workclass.
    fnlgt : int
        Final weight in the census data.
    education : str
        Level of education.
    education_num : int
        Numeric representation of education level.
    marital_status : str
        Marital status of the individual.
    occupation : str
        Occupation of the individual.
    relationship : str
        Relationship status of the individual.
    race : str
        Ethnicity of the individual.
    sex : str
        Gender of the individual.
    capital_gain : int
        Capital gains of the individual.
    capital_loss : int
        Capital losses of the individual.
    hours_per_week : int
        Number of hours worked per week.
    native_country : str
        Native country of the individual.
    model_feature : str
        Additional feature specific to the model
        (example: 'feature1', 'feature2').
    """

    # Using the first row of census.csv as sample
    age: int = Field(None, example=39)
    workclass: str = Field(None, example="State-gov")
    fnlgt: int = Field(None, example=77516)
    education: str = Field(None, example="Bachelors")
    education_num: int = Field(None, example=13)
    marital_status: str = Field(None, example="Never-married")
    occupation: str = Field(None, example="Adm-clerical")
    relationship: str = Field(None, example="Not-in-family")
    race: str = Field(None, example="White")
    sex: str = Field(None, example="Female")
    capital_gain: int = Field(None, example=2174)
    capital_loss: int = Field(None, example=0)
    hours_per_week: int = Field(None, example=40)
    native_country: str = Field(None, example="United-States")


class Prediction(BaseModel):
    """
    Prediction Schema for Machine Learning Prediction API

    Attributes:
    -----------
    prediction : str
        Predicted outcome from the model.
    """

    prediction: str
