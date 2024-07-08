# Put the code for your API here.
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field

app = FastAPI()


@app.get("/")
async def root():
    return "Hello Mehul!"


class InputData(BaseModel):
    """Model Input data Scehma"""

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


@app.post("/predict")
async def predict(data: InputData):
    return {"pred": 1}


if __name__ == "__main__":
    config = uvicorn.Config(
        "main:app", host="0.0.0.0", reload=True, port=8080, log_level="info"
    )
    server = uvicorn.Server(config)
    server.run()
