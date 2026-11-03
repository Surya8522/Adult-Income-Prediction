
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA = BASE_DIR / "data" / "raw" / "adult.csv"
OUTPUT_DIR = BASE_DIR / "data" / "monitoring" / "drift"


def main():

    print("=" * 65)
    print("LAB 12 - PREPARE ADULT INCOME MONITORING DATA")
    print("=" * 65)

    if not RAW_DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA}")

    df = pd.read_csv(RAW_DATA)

    # Apply the same cleaning as the original preprocessing pipeline.
    df.columns = df.columns.str.strip()
    df = df.replace("?", np.nan)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

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
        raise ValueError("Invalid income target values found.")

    X = df.drop(columns=["income"])
    y = df["income"].astype(int)

    # Use exactly the original split configuration.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save raw features, before preprocessing.
    X_train.to_csv(
        OUTPUT_DIR / "baseline_train_features.csv",
        index=False
    )

    X_test.to_csv(
        OUTPUT_DIR / "baseline_test_features.csv",
        index=False
    )

    np.save(
        OUTPUT_DIR / "baseline_train_labels.npy",
        y_train.to_numpy()
    )

    np.save(
        OUTPUT_DIR / "baseline_test_labels.npy",
        y_test.to_numpy()
    )

    metadata = {
        "project": "Adult Income Prediction",
        "target": "income",
        "target_mapping": {
            "<=50K": 0,
            ">50K": 1
        },
        "original_dataset_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "feature_count": int(X.shape[1]),
        "test_size": 0.20,
        "random_state": 42,
        "stratified": True
    }

    with open(
        OUTPUT_DIR / "baseline_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\nBASELINE DATA CREATED")
    print("-" * 65)
    print("Training features:", X_train.shape)
    print("Testing features :", X_test.shape)
    print("Training labels  :", y_train.shape)
    print("Testing labels   :", y_test.shape)

    print("\nSaved files:")
    for file in sorted(OUTPUT_DIR.iterdir()):
        print(file.name)

    print("\n[SUCCESS] Baseline monitoring data prepared.")


if __name__ == "__main__":
    main()
