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

    print("[INFO] Starting Sklearn Pipeline Preprocessing...")

    # ==================================================
    # 1. Load Adult Income dataset
    # ==================================================

    data_path = "data/raw/adult.csv"

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    df = pd.read_csv(data_path)

    print(f"[INFO] Dataset shape: {df.shape}")

    # ==================================================
    # 2. Clean column names
    # ==================================================

    df.columns = df.columns.str.strip()

    # ==================================================
    # 3. Replace ? with NaN
    # ==================================================

    df = df.replace("?", np.nan)

    # Remove whitespace from object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # ==================================================
    # 4. Validate target
    # ==================================================

    if "income" not in df.columns:
        raise ValueError(
            "Target column 'income' not found."
        )

    # ==================================================
    # 5. Encode target
    # ==================================================

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
        raise ValueError(
            "Invalid values found in income column."
        )

    # ==================================================
    # 6. Separate X and y
    # ==================================================

    X = df.drop(
        columns=["income"]
    )

    y = df["income"].astype(int)

    # ==================================================
    # 7. Identify numerical and categorical columns
    # ==================================================

    numerical_features = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    print(
        f"[INFO] Numerical features: "
        f"{numerical_features}"
    )

    print(
        f"[INFO] Categorical features: "
        f"{categorical_features}"
    )

    # ==================================================
    # 8. Numerical preprocessing
    # ==================================================

    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    # ==================================================
    # 9. Categorical preprocessing
    # ==================================================

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    # ==================================================
    # 10. Combined preprocessing
    # ==================================================

    preprocessor = ColumnTransformer([
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])

    # ==================================================
    # 11. Train/Test split
    # ==================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(
        f"[INFO] Training samples: {len(X_train)}"
    )

    print(
        f"[INFO] Testing samples: {len(X_test)}"
    )

    # ==================================================
    # 12. Fit only on training data
    # ==================================================

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    print(
        f"[INFO] Processed training shape: "
        f"{X_train_processed.shape}"
    )

    print(
        f"[INFO] Processed testing shape: "
        f"{X_test_processed.shape}"
    )

    # ==================================================
    # 13. Create directories
    # ==================================================

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    os.makedirs(
        "models",
        exist_ok=True
    )

    os.makedirs(
        "reports",
        exist_ok=True
    )

    # ==================================================
    # 14. Save processed datasets
    # ==================================================

    np.save(
        "data/processed/X_train_final.npy",
        X_train_processed
    )

    np.save(
        "data/processed/X_test_final.npy",
        X_test_processed
    )

    np.save(
        "data/processed/y_train.npy",
        y_train.to_numpy()
    )

    np.save(
        "data/processed/y_test.npy",
        y_test.to_numpy()
    )

    # ==================================================
    # 15. Save preprocessing object
    # ==================================================

    joblib.dump(
        preprocessor,
        "models/preprocessor.pkl"
    )

    # ==================================================
    # 16. Save metadata
    # ==================================================

    metadata = {
        "dataset": "Adult Income",
        "dataset_path": data_path,
        "target": "income",

        "target_mapping": {
            "<=50K": 0,
            ">50K": 1
        },

        "original_rows": int(df.shape[0]),
        "original_columns": int(df.shape[1]),

        "train_rows": int(
            X_train_processed.shape[0]
        ),

        "test_rows": int(
            X_test_processed.shape[0]
        ),

        "processed_features": int(
            X_train_processed.shape[1]
        ),

        "numerical_features":
            numerical_features,

        "categorical_features":
            categorical_features,

        "test_size": 0.20,
        "random_state": 42
    }

    with open(
        "data/processed/dataset_metadata.json",
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    print(
        "[INFO] Metadata saved."
    )

    print(
        "[SUCCESS] Sklearn preprocessing completed!"
    )


if __name__ == "__main__":
    run_preprocessing()