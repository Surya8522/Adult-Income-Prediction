import os
import pandas as pd


def validate_data():

    print("=========================================")
    print("Validating Adult Income Dataset")
    print("=========================================")

    path = "data/raw/adult.csv"

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    required_columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education.num",
        "marital.status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital.gain",
        "capital.loss",
        "hours.per.week",
        "native.country",
        "income"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if len(df) == 0:
        raise ValueError(
            "Dataset is empty."
        )

    print(
        f"Dataset shape: {df.shape}"
    )

    print(
        "All required columns are present."
    )

    print(
        "Adult Income dataset validation passed."
    )


if __name__ == "__main__":
    validate_data()