
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


@app.get("/")
def home():
    return {
        "message": "Adult Income Prediction API",
        "status": "running"
    }


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


def make_prediction(customer):

    data = customer.model_dump(by_alias=True)

    df = pd.DataFrame([data])
    df = clean_prediction_data(df)

    processed_data = loader.get_preprocessor().transform(df)

    prediction = int(
        loader.get_model().predict(processed_data)[0]
    )

    income_class = ">50K" if prediction == 1 else "<=50K"

    log_inference(data, prediction)

    return PredictionResponse(
        prediction=prediction,
        income_class=income_class
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):

    try:
        return make_prediction(customer)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


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
