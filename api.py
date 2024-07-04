from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from function import train_model, predict_size
import openai
import os
import requests

app = FastAPI(
    title="Clothing Size Prediction API",
    description="""
    This API allows you to train a machine learning model on a dataset of clothing sizes
    and make predictions about clothing sizes based on user input.
    The API also provides a model information endpoint using HuggingFace models.
    """,
    version="1.0.0"
)

class TrainingData(BaseModel):
    data_path: str

class PredictionData(BaseModel):
    weight: float
    age: int
    height: float

    class Config:
        schema_extra = {
            "example": {
                "weight": 70.0,
                "age": 34,
                "height": 180.0
            }
        }

@app.post("/training", tags=["Model Training"], description="Train a model with the provided dataset")
def train():
    try:
        train_model('sizes.csv')
        return {"message": "Model trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict", tags=["Prediction"], description="Predict clothing size using the trained model")
def predict(data: PredictionData):
    """
    Predict clothing size using the trained model.

    This endpoint receives the weight, age, and height of a person and returns the predicted clothing size.

    Example request:
    {
        "weight": 70.0,
        "age": 34,
        "height": 180.0
    }

    Returns:
        JSON response containing the predicted clothing size.

    Raises:
        HTTPException: If there is an error with the prediction process.
    """
    try:
        size = predict_size(data.weight, data.age, data.height)
        return {"predicted_size": size}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model", tags=["Model Information"], description="Get information about a HuggingFace model.")
def get_model():
    """
    Retrieve information about the HuggingFace model `distilbert-base-uncased`.

    This endpoint queries the HuggingFace API to retrieve metadata and details about the specified model.
    It requires a valid HuggingFace API token to authenticate the request.

    Returns:
        JSON response containing model information.

    Raises:
        HTTPException: If there is an error with the API request or the token is invalid.
    """
    try:
        api_token = "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"
        headers = {"Authorization": f"Bearer {api_token}"}
        response = requests.get("https://api-inference.huggingface.co/models/distilbert-base-uncased", headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.json())
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
