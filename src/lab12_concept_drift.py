
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "monitoring" / "drift"

MODEL_PATH = (
    BASE_DIR / "artifacts" / "model_registry"
    / "production_model.pkl"
)

PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"

BASELINE_FEATURES = DATA_DIR / "baseline_test_features.csv"
BASELINE_LABELS = DATA_DIR / "baseline_test_labels.npy"
EVOLVED_DATA = DATA_DIR / "evolved_labeled_data.csv"

OUTPUT_DIR = BASE_DIR / "outputs" / "drift"
REPORT_PATH = OUTPUT_DIR / "concept_drift_report.json"

def calculate_metrics(y_true, predictions, probabilities):
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities))
    }


def main():
    print("=" * 65)
    print("LAB 12 - ADULT INCOME CONCEPT DRIFT EVALUATION")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in [
        MODEL_PATH,
        PREPROCESSOR_PATH,
        BASELINE_FEATURES,
        BASELINE_LABELS,
        EVOLVED_DATA
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Baseline dataset
    baseline_features = pd.read_csv(BASELINE_FEATURES)
    baseline_labels = np.load(BASELINE_LABELS).astype(int)

    # Evolved dataset
    evolved_data = pd.read_csv(EVOLVED_DATA)

    evolved_features = evolved_data.drop(columns=["income"])
    evolved_labels = evolved_data["income"].astype(int).to_numpy()

    if len(baseline_features) != len(baseline_labels):
        raise ValueError("Baseline feature and label counts do not match.")

    if list(baseline_features.columns) != list(evolved_features.columns):
        raise ValueError("Baseline and evolved feature columns do not match.")

    # Apply the existing production preprocessing
    baseline_transformed = preprocessor.transform(baseline_features)
    evolved_transformed = preprocessor.transform(evolved_features)

    # Generate predictions
    baseline_predictions = model.predict(baseline_transformed)
    evolved_predictions = model.predict(evolved_transformed)

    baseline_probabilities = model.predict_proba(
        baseline_transformed
    )[:, 1]

    evolved_probabilities = model.predict_proba(
        evolved_transformed
    )[:, 1]

    # Calculate metrics
    baseline_metrics = calculate_metrics(
        baseline_labels,
        baseline_predictions,
        baseline_probabilities
    )

    evolved_metrics = calculate_metrics(
        evolved_labels,
        evolved_predictions,
        evolved_probabilities
    )

    metric_changes = {}

    for metric in baseline_metrics:
        metric_changes[metric] = round(
            evolved_metrics[metric] - baseline_metrics[metric],
            6
        )

    summary = {
        "project": "Adult Income Prediction",
        "model": "Production Model Version 5",
        "baseline_metrics": baseline_metrics,
        "evolved_metrics": evolved_metrics,
        "metric_changes": metric_changes,
        "concept_drift_simulated": True,
        "labels_flipped": 1302,
        "labels_flip_percentage": 20
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\nBASELINE PERFORMANCE")
    for metric, value in baseline_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nEVOLVED DATA PERFORMANCE")
    for metric, value in evolved_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nMETRIC CHANGES (EVOLVED - BASELINE)")
    for metric, value in metric_changes.items():
        print(f"{metric}: {value:+.4f}")

    print("\nSaved report:", REPORT_PATH)
    print("\n[SUCCESS] Concept drift evaluation completed.")


if __name__ == "__main__":
    main()
