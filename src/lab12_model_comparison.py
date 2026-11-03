
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
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs" / "retraining"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

V1_MODEL_PATH = (
    BASE_DIR / "artifacts" / "model_registry"
    / "production_model.pkl"
)

V2_MODEL_PATH = MODEL_DIR / "retrained_model_v2.pkl"

PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"

TEST_FEATURES_PATH = DATA_DIR / "baseline_test_features.csv"
TEST_LABELS_PATH = DATA_DIR / "baseline_test_labels.npy"

V1_METADATA_PATH = (
    BASE_DIR / "artifacts" / "model_registry"
    / "production_model_metadata.json"
)

REPORT_PATH = OUTPUT_DIR / "model_comparison_v1_v2.json"

# Quality gate thresholds
THRESHOLDS = {
    "accuracy": 0.70,
    "precision": 0.50,
    "recall": 0.70,
    "f1_score": 0.60,
    "roc_auc": 0.75
}


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


def main():

    print("=" * 65)
    print("LAB 12 - MODEL COMPARISON: PRODUCTION V5 VS RETRAINED V2")
    print("=" * 65)

    required_files = [
        V1_MODEL_PATH,
        V2_MODEL_PATH,
        PREPROCESSOR_PATH,
        TEST_FEATURES_PATH,
        TEST_LABELS_PATH,
        V1_METADATA_PATH
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # Load both models
    v1_model = joblib.load(V1_MODEL_PATH)
    v2_model = joblib.load(V2_MODEL_PATH)

    # Load existing preprocessor
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Load the same untouched test dataset
    X_test = pd.read_csv(TEST_FEATURES_PATH)
    y_test = np.load(TEST_LABELS_PATH).astype(int)

    X_test_transformed = preprocessor.transform(X_test)

    # Evaluate both models on identical test data
    v1_metrics = calculate_metrics(
        v1_model,
        X_test_transformed,
        y_test
    )

    v2_metrics = calculate_metrics(
        v2_model,
        X_test_transformed,
        y_test
    )

    # Calculate V2 improvement over V1
    improvements = {}

    for metric in v1_metrics:

        improvements[metric] = (
            v2_metrics[metric] - v1_metrics[metric]
        )

    # Check V2 quality gates
    quality_gate = {}

    for metric, threshold in THRESHOLDS.items():

        quality_gate[metric] = (
            v2_metrics[metric] >= threshold
        )

    all_gates_passed = all(quality_gate.values())

    # Staging eligibility requires all quality gates
    # and an improvement in F1-score over V1.
    f1_improved = (
        v2_metrics["f1_score"] > v1_metrics["f1_score"]
    )

    eligible_for_staging = (
        all_gates_passed and f1_improved
    )

    # Build comparison report
    report = {
        "project": "Adult Income Prediction",
        "comparison": "Production V5 vs Retrained V2",
        "test_rows": int(len(X_test)),
        "same_test_dataset": True,
        "production_v1_metrics": v1_metrics,
        "retrained_v2_metrics": v2_metrics,
        "v2_improvement_over_v1": improvements,
        "quality_thresholds": THRESHOLDS,
        "v2_quality_gate": quality_gate,
        "all_quality_gates_passed": all_gates_passed,
        "f1_improved": f1_improved,
        "eligible_for_staging": eligible_for_staging,
        "production_model_changed": False
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(report, file, indent=4)

    # Display results
    print("\nMODEL PERFORMANCE COMPARISON")
    print("-" * 65)

    print(
        f"{'Metric':<15}"
        f"{'Production V5':<20}"
        f"{'Retrained V2':<20}"
        f"{'Difference':<15}"
    )

    for metric in v1_metrics:

        print(
            f"{metric:<15}"
            f"{v1_metrics[metric]:<20.4f}"
            f"{v2_metrics[metric]:<20.4f}"
            f"{improvements[metric]:+.4f}"
        )

    print("\nV2 QUALITY GATE")

    for metric, passed in quality_gate.items():

        status = "PASS" if passed else "FAIL"

        print(
            f"{metric}: {status} "
            f"(required: {THRESHOLDS[metric]:.2f})"
        )

    print("\nAll quality gates passed:", all_gates_passed)
    print("F1-score improved:", f1_improved)

    print(
        "Eligible for Staging:",
        eligible_for_staging
    )

    print("\nProduction V5 changed: NO")

    print("\nComparison report saved:")
    print(REPORT_PATH)

    print("\n[SUCCESS] Model comparison completed.")


if __name__ == "__main__":
    main()
