
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


class CustomerInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    age: int = Field(ge=17, le=100)
    workclass: str
    fnlwgt: float
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: Literal["Male", "Female"]
    capital_gain: float = Field(alias="capital-gain", ge=0)
    capital_loss: float = Field(alias="capital-loss", ge=0)
    hours_per_week: int = Field(alias="hours-per-week", ge=1, le=100)
    native_country: str = Field(alias="native-country")


class PredictionResponse(BaseModel):
    prediction: int
    income_class: str


class BatchPredictRequest(BaseModel):
    customers: list[CustomerInput] = Field(min_length=1)


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]
