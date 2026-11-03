
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import mlflow

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "monitoring" / "drift"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs" / "retraining"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_TRAIN_FEATURES = DATA_DIR / "baseline_train_features.csv"
BASELINE_TRAIN_LABELS = DATA_DIR / "baseline_train_labels.npy"

BASELINE_TEST_FEATURES = DATA_DIR / "baseline_test_features.csv"
BASELINE_TEST_LABELS = DATA_DIR / "baseline_test_labels.npy"

PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"

EVOLVED_TRAIN_PATH = DATA_DIR / "evolved_train_labeled_data.csv"
EVOLVED_TRAIN_REPORT = DATA_DIR / "evolved_train_generation_report.json"

V2_MODEL_PATH = MODEL_DIR / "retrained_model_v2.pkl"
REPORT_PATH = OUTPUT_DIR / "retrained_model_v2_report.json"

MODEL_NAME = "Adult_Income_Prediction_V2"


# --------------------------------------------------
# GENERATE EVOLVED TRAINING DATA
# --------------------------------------------------

def generate_evolved_training_data(X_train, y_train):

    evolved = X_train.copy()

    # Simulate changes in numeric features
    if "age" in evolved.columns:
        evolved["age"] = (evolved["age"] + 5).clip(upper=90)

    if "hours.per.week" in evolved.columns:
        evolved["hours.per.week"] = (
            evolved["hours.per.week"] + 5
        ).clip(upper=99)

    if "capital.gain" in evolved.columns:
        evolved["capital.gain"] = (
            evolved["capital.gain"] * 1.5
        ).round()

    # Simulate a categorical distribution change
    if "workclass" in evolved.columns:
        evolved["workclass"] = evolved["workclass"].replace(
            {"Private": "Self-emp-not-inc"}
        )

    # Simulate concept drift by flipping 20% of training labels
    evolved_labels = y_train.copy()

    rng = np.random.default_rng(42)

    number_to_flip = int(len(evolved_labels) * 0.20)

    indices_to_flip = rng.choice(
        len(evolved_labels),
        size=number_to_flip,
        replace=False
    )

    evolved_labels[indices_to_flip] = (
        1 - evolved_labels[indices_to_flip]
    )

    evolved["income"] = evolved_labels

    evolved.to_csv(EVOLVED_TRAIN_PATH, index=False)

    report = {
        "project": "Adult Income Prediction",
        "dataset": "Evolved training dataset",
        "original_training_rows": int(len(X_train)),
        "evolved_training_rows": int(len(evolved)),
        "labels_flipped": int(number_to_flip),
        "label_flip_percentage": 20,
        "random_state": 42,
        "features_changed": [
            "age",
            "hours.per.week",
            "capital.gain",
            "workclass"
        ],
        "test_data_used_for_training": False
    }

    with open(EVOLVED_TRAIN_REPORT, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nEvolved training dataset generated.")
    print("Rows:", len(evolved))
    print("Labels flipped:", number_to_flip)
    print("Saved:", EVOLVED_TRAIN_PATH)

    return evolved


# --------------------------------------------------
# METRIC CALCULATION
# --------------------------------------------------

def calculate_metrics(model, X, y):

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return {
        "accuracy": float(
            accuracy_score(y, predictions)
        ),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y, probabilities)
        )
    }


# --------------------------------------------------
# MAIN RETRAINING FUNCTION
# --------------------------------------------------

def main():

    print("=" * 65)
    print("LAB 12 - ADULT INCOME V2 MODEL RETRAINING")
    print("=" * 65)

    # Check required files
    required_files = [
        BASELINE_TRAIN_FEATURES,
        BASELINE_TRAIN_LABELS,
        BASELINE_TEST_FEATURES,
        BASELINE_TEST_LABELS,
        PREPROCESSOR_PATH
    ]

    for path in required_files:

        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # Load original training data
    X_train = pd.read_csv(BASELINE_TRAIN_FEATURES)

    y_train = np.load(
        BASELINE_TRAIN_LABELS
    ).astype(int)

    # Load original untouched test data
    X_test = pd.read_csv(BASELINE_TEST_FEATURES)

    y_test = np.load(
        BASELINE_TEST_LABELS
    ).astype(int)

    # Validate dimensions
    if len(X_train) != len(y_train):
        raise ValueError("Training feature and label counts differ.")

    if len(X_test) != len(y_test):
        raise ValueError("Testing feature and label counts differ.")

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Training and testing feature columns differ.")

    print("\nOriginal training rows:", len(X_train))
    print("Original testing rows:", len(X_test))

    # Generate evolved data using TRAINING data only
    evolved_data = generate_evolved_training_data(
        X_train,
        y_train
    )

    # Separate evolved features and labels
    X_evolved = evolved_data.drop(columns=["income"])

    y_evolved = evolved_data["income"].astype(int).to_numpy()

    # Load existing preprocessing object
    # Do not overwrite the production preprocessor
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Transform evolved training data
    X_train_transformed = preprocessor.transform(X_evolved)

    # Transform original untouched test data
    X_test_transformed = preprocessor.transform(X_test)

    print("\nTransformed training shape:", X_train_transformed.shape)
    print("Transformed testing shape:", X_test_transformed.shape)

    # --------------------------------------------------
    # TRAIN V2 MODEL
    # --------------------------------------------------

    params = {
        "n_estimators": 150,
        "max_depth": 12,
        "random_state": 42,
        "class_weight": "balanced",
        "training_data": "evolved_training_data_only"
    }

    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"],
        class_weight=params["class_weight"],
        n_jobs=-1
    )

    mlflow.set_experiment(
        "Adult_Income_Lab12_Retraining"
    )

    with mlflow.start_run(
        run_name="Adult_Income_Retrained_V2"
    ):

        print("\nTraining V2 model...")

        model.fit(
            X_train_transformed,
            y_evolved
        )

        # Evaluate on the original untouched test set
        metrics = calculate_metrics(
            model,
            X_test_transformed,
            y_test
        )

        # Log parameters and metrics
        mlflow.log_params(params)

        mlflow.log_param("model_version", "V2")
        mlflow.log_param("project", "Adult Income Prediction")
        mlflow.log_param("test_data", "Original untouched baseline test set")

        mlflow.log_metrics(metrics)

        # Save model separately
        joblib.dump(
            model,
            V2_MODEL_PATH
        )

        # Log saved model as an artifact
        # This avoids MLflow's skops serialization route
        mlflow.log_artifact(
            str(V2_MODEL_PATH),
            artifact_path="model"
        )

        # Save report
        report = {
            "project": "Adult Income Prediction",
            "model": MODEL_NAME,
            "version": "V2",
            "training_rows": int(len(X_evolved)),
            "testing_rows": int(len(X_test)),
            "training_data": "Evolved original training data",
            "test_data": "Original untouched baseline test data",
            "test_data_used_for_training": False,
            "metrics": metrics,
            "model_file": str(V2_MODEL_PATH),
            "preprocessor_file": str(PREPROCESSOR_PATH),
            "mlflow_run_id": mlflow.active_run().info.run_id
        }

        with open(
            REPORT_PATH,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                report,
                file,
                indent=4
            )

        print("\n" + "=" * 50)
        print("V2 MODEL METRICS")
        print("=" * 50)

        for metric, value in metrics.items():

            print(f"{metric}: {value:.4f}")

        print(
            "\nMLflow Run ID:",
            mlflow.active_run().info.run_id
        )

    print("\nV2 model saved:", V2_MODEL_PATH)
    print("Report saved:", REPORT_PATH)

    print("\n[SUCCESS] V2 model retraining completed.")


if __name__ == "__main__":
    main()
