import pandas as pd

from src.schemas import REQUIRED_FIELDS


def preprocess_input(data, preprocessor):

    if hasattr(data, "model_dump"):
        data = data.model_dump()
    elif hasattr(data, "dict"):
        data = data.dict()

    if not isinstance(data, dict):
        raise ValueError("Input must be a dictionary")

    missing_fields = [
        field for field in REQUIRED_FIELDS
        if field not in data
    ]

    if missing_fields:
        raise ValueError(f"Missing fields: {missing_fields}")

    input_df = pd.DataFrame([data])

    return preprocessor.transform(input_df)