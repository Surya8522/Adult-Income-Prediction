
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

EVOLVED_DATA = DATA_DIR / "evolved_labeled_data.csv"
REPORT_PATH = OUTPUT_DIR / "performance_trend_report.json"


def evaluate_window(model, preprocessor, features, labels):
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
    print("LAB 12 - ADULT INCOME PERFORMANCE TREND MONITORING")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(PREPROCESSOR_PATH)

    if not EVOLVED_DATA.exists():
        raise FileNotFoundError(EVOLVED_DATA)

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    data = pd.read_csv(EVOLVED_DATA)

    features = data.drop(columns=["income"])
    labels = data["income"].astype(int).to_numpy()

    # Divide the data into three sequential windows
    windows = np.array_split(np.arange(len(data)), 3)

    trend_results = []

    for index, window_indices in enumerate(windows, start=1):
        window_features = features.iloc[window_indices]
        window_labels = labels[window_indices]

        metrics = evaluate_window(
            model,
            preprocessor,
            window_features,
            window_labels
        )

        result = {
            "window": index,
            "rows": int(len(window_indices)),
            "metrics": metrics
        }

        trend_results.append(result)

        print(f"\nWINDOW {index}")
        print("Rows:", len(window_indices))

        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    # Compare first and last windows
    first_metrics = trend_results[0]["metrics"]
    last_metrics = trend_results[-1]["metrics"]

    metric_changes = {
        metric: round(
            last_metrics[metric] - first_metrics[metric],
            6
        )
        for metric in first_metrics
    }

    summary = {
        "project": "Adult Income Prediction",
        "model": "Production Model Version 5",
        "number_of_windows": 3,
        "window_results": trend_results,
        "first_to_last_metric_changes": metric_changes,
        "trend_status": (
            "PERFORMANCE_DECLINING"
            if any(value < 0 for value in metric_changes.values())
            else "STABLE_OR_IMPROVING"
        )
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\nFIRST TO LAST WINDOW CHANGES")

    for metric, change in metric_changes.items():
        print(f"{metric}: {change:+.4f}")

    print("\nTrend status:", summary["trend_status"])
    print("Report saved:", REPORT_PATH)

    print("\n[SUCCESS] Performance trend monitoring completed.")


if __name__ == "__main__":
    main()
