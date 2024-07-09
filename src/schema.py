from pydantic import BaseModel, Field


class InputData(BaseModel):
    """Model Input data Schema"""

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
    """Prediction Schema"""

    prediction: str
