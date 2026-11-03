
import pandas as pd


def clean_prediction_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = [
        column.strip().replace("_", "-")
        for column in df.columns
    ]

    # Clean whitespace in categorical columns
    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].str.strip()

    # Replace missing-value markers
    df = df.replace("?", pd.NA)

    return df
