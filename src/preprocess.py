import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def run_preprocessing():

    print("=========================================")
    print("Starting Adult Income Preprocessing")
    print("=========================================")

    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------
    data_path = "data/raw/adult.csv"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    print(f"Dataset shape: {df.shape}")

    # --------------------------------------------------
    # 2. Clean column names
    # --------------------------------------------------
    df.columns = df.columns.str.strip()

    # --------------------------------------------------
    # 3. Replace '?' with NaN
    # --------------------------------------------------
    df = df.replace("?", np.nan)

    # Remove accidental whitespace from string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # --------------------------------------------------
    # 4. Target encoding
    # --------------------------------------------------
    if "income" not in df.columns:
        raise ValueError("Target column 'income' not found.")

    df["income"] = (
        df["income"]
        .astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
    )

    df["income"] = df["income"].map({
        "<=50K": 0,
        ">50K": 1
    })

    if df["income"].isna().any():
        raise ValueError("Unexpected values found in income column.")

    # --------------------------------------------------
    # 5. Separate features and target
    # --------------------------------------------------
    X = df.drop(columns=["income"])
    y = df["income"].astype(int)

    # --------------------------------------------------
    # 6. Identify feature types
    # --------------------------------------------------
    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    print("Numerical features:")
    print(numerical_features)

    print("\nCategorical features:")
    print(categorical_features)

    # --------------------------------------------------
    # 7. Numerical pipeline
    # --------------------------------------------------
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # --------------------------------------------------
    # 8. Categorical pipeline
    # --------------------------------------------------
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    # --------------------------------------------------
    # 9. Combined preprocessing
    # --------------------------------------------------
    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    # --------------------------------------------------
    # 10. Train/test split
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # --------------------------------------------------
    # 11. Fit preprocessing only on training data
    # --------------------------------------------------
    X_train_final = preprocessor.fit_transform(X_train)
    X_test_final = preprocessor.transform(X_test)

    print(f"Processed training shape: {X_train_final.shape}")
    print(f"Processed testing shape: {X_test_final.shape}")

    # --------------------------------------------------
    # 12. Create directories
    # --------------------------------------------------
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # --------------------------------------------------
    # 13. Save processed data
    # --------------------------------------------------
    np.save(
        "data/processed/X_train_final.npy",
        X_train_final
    )

    np.save(
        "data/processed/X_test_final.npy",
        X_test_final
    )

    np.save(
        "data/processed/y_train.npy",
        y_train.to_numpy()
    )

    np.save(
        "data/processed/y_test.npy",
        y_test.to_numpy()
    )

    # --------------------------------------------------
    # 14. Save preprocessor
    # --------------------------------------------------
    joblib.dump(
        preprocessor,
        "models/preprocessor.pkl"
    )

    # --------------------------------------------------
    # 15. Save metadata
    # --------------------------------------------------
    metadata = {
        "dataset": "Adult Income",
        "target": "income",
        "target_mapping": {
            "<=50K": 0,
            ">50K": 1
        },
        "original_shape": list(df.shape),
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "test_size": 0.20,
        "random_state": 42
    }

    with open(
        "data/processed/dataset_metadata.json",
        "w"
    ) as f:
        json.dump(metadata, f, indent=4)

    print("\n=========================================")
    print("Preprocessing completed successfully!")
    print("=========================================")


if __name__ == "__main__":
    run_preprocessing()