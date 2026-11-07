
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import (
    CustomerInput,
    PredictionResponse,
    BatchPredictRequest,
    BatchPredictionResponse
)

from api.preprocessing import clean_prediction_data
from api.model_loader import ModelLoader
from api.inference_logger import log_inference


loader = ModelLoader()


# Load the trained model when the API starts
@asynccontextmanager
async def lifespan(app: FastAPI):
    loader.load_model()
    yield


app = FastAPI(
    title="Adult Income Prediction API",
    description="Predict whether annual income is above or below $50K",
    version="1.0",
    lifespan=lifespan
)


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Adult Income Prediction API",
        "status": "running"
    }


# Health check endpoint
@app.get("/health")
def health():

    if loader.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "metadata": loader.get_metadata()
    }


# Convert API column names to trained model column names
COLUMN_MAPPING = {
    "education-num": "education.num",
    "marital-status": "marital.status",
    "capital-gain": "capital.gain",
    "capital-loss": "capital.loss",
    "hours-per-week": "hours.per.week",
    "native-country": "native.country"
}


# Common prediction function
def make_prediction(customer):

    # Convert Pydantic input into dictionary
    data = customer.model_dump(by_alias=True)

    # Create DataFrame
    df = pd.DataFrame([data])

    # Rename columns to match the trained preprocessor
    df = df.rename(columns=COLUMN_MAPPING)

    # Clean input data
    df = clean_prediction_data(df)

    # Apply trained preprocessing
    processed_data = loader.get_preprocessor().transform(df)

    # Predict income class
    prediction = int(
        loader.get_model().predict(processed_data)[0]
    )

    # Convert prediction into readable output
    income_class = ">50K" if prediction == 1 else "<=50K"

    # Save inference log
    log_inference(data, prediction)

    return PredictionResponse(
        prediction=prediction,
        income_class=income_class
    )


# Single customer prediction endpoint
@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(customer: CustomerInput):

    try:
        return make_prediction(customer)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# Batch customer prediction endpoint
@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse
)
def batch_predict(request: BatchPredictRequest):

    try:
        results = [
            make_prediction(customer)
            for customer in request.customers
        ]

        return BatchPredictionResponse(
            count=len(results),
            predictions=results
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )
