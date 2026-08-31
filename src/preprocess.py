import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(file_path):
    """Load the Adult Income dataset."""
    df = pd.read_csv(file_path)

    # Replace unknown values with NaN
    df = df.replace("?", np.nan)

    # Remove duplicate rows
    df = df.drop_duplicates()

    return df


def prepare_data(df):
    """Separate features and target."""

    X = df.drop("income", axis=1)

    y = df["income"].map({
        "<=50K": 0,
        ">50K": 1
    })

    return X, y


def create_preprocessor(X):
    """Create preprocessing pipeline."""

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    return preprocessor


def split_data(X, y):
    """Split data into training and testing sets."""

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )