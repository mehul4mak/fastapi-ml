# Put the code for your API here.
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root():
    return "Hello Mehul!"


@app.post("/predict")
def predict():
    return 1
