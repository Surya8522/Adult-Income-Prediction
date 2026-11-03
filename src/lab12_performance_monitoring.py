
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
OUTPUT_DIR = BASE_DIR / "outputs" / "monitoring"

MODEL_PATH = (
    BASE_DIR / "artifacts" / "model_registry"
    / "production_model.pkl"
)
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"

BASELINE_FEATURES = DATA_DIR / "baseline_test_features.csv"
BASELINE_LABELS = DATA_DIR / "baseline_test_labels.npy"
EVOLVED_DATA = DATA_DIR / "evolved_labeled_data.csv"

REPORT_PATH = OUTPUT_DIR / "performance_monitoring_report.json"

THRESHOLDS = {
    "accuracy": 0.70,
    "precision": 0.50,
    "recall": 0.70,
    "f1_score": 0.60,
    "roc_auc": 0.75
}


def evaluate_model(model, preprocessor, features, labels):
    transformed = preprocessor.transform(features)

    predictions = model.predict(transformed)
    probabilities = model.predict_proba(transformed)[:, 1]

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(labels, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(labels, probabilities))
    }


def main():
    print("=" * 65)
    print("LAB 12 - ADULT INCOME PERFORMANCE MONITORING")
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

    baseline_features = pd.read_csv(BASELINE_FEATURES)
    baseline_labels = np.load(BASELINE_LABELS).astype(int)

    evolved_data = pd.read_csv(EVOLVED_DATA)
    evolved_features = evolved_data.drop(columns=["income"])
    evolved_labels = evolved_data["income"].astype(int).to_numpy()

    baseline_metrics = evaluate_model(
        model, preprocessor, baseline_features, baseline_labels
    )

    evolved_metrics = evaluate_model(
        model, preprocessor, evolved_features, evolved_labels
    )

    baseline_gate = {
        metric: value >= THRESHOLDS[metric]
        for metric, value in baseline_metrics.items()
    }

    evolved_gate = {
        metric: value >= THRESHOLDS[metric]
        for metric, value in evolved_metrics.items()
    }

    summary = {
        "project": "Adult Income Prediction",
        "model": "Production Model Version 5",
        "thresholds": THRESHOLDS,
        "baseline_metrics": baseline_metrics,
        "evolved_metrics": evolved_metrics,
        "baseline_quality_gate": baseline_gate,
        "evolved_quality_gate": evolved_gate,
        "baseline_passed": bool(all(baseline_gate.values())),
        "evolved_passed": bool(all(evolved_gate.values())),
        "monitoring_status": (
            "DEGRADATION_DETECTED"
            if not all(evolved_gate.values())
            else "HEALTHY"
        )
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\nBASELINE METRICS")
    for metric, value in baseline_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nEVOLVED METRICS")
    for metric, value in evolved_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nQUALITY GATE RESULTS")
    for metric in THRESHOLDS:
        print(
            f"{metric}: "
            f"baseline={baseline_gate[metric]}, "
            f"evolved={evolved_gate[metric]}"
        )

    print("\nMonitoring status:", summary["monitoring_status"])
    print("Report saved:", REPORT_PATH)
    print("\n[SUCCESS] Performance monitoring completed.")


if __name__ == "__main__":
    main()
