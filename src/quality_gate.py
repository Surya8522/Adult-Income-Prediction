
import json
import joblib
import numpy as np

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "artifacts"
    / "model_registry"
    / "production_model.pkl"
)

X_TEST_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "X_test_final.npy"
)

Y_TEST_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "y_test.npy"
)

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = REPORT_DIR / "quality_gate_report.json"


# ==========================================
# QUALITY THRESHOLDS
# ==========================================

MIN_ACCURACY = 0.78
MIN_PRECISION = 0.70
MIN_RECALL = 0.70
MIN_F1 = 0.70


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    print("======================================")
    print("ADULT INCOME MODEL QUALITY GATE")
    print("======================================")

    # Load model
    print("\nLoading production model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    # Load test data
    print("Loading test dataset...")

    if not X_TEST_PATH.exists():
        raise FileNotFoundError(X_TEST_PATH)

    if not Y_TEST_PATH.exists():
        raise FileNotFoundError(Y_TEST_PATH)

    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    # Generate predictions
    print("Generating predictions...")

    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),

        "precision": float(
            precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )
        ),

        "f1_score": float(
            f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )
        )
    }

    # Check thresholds
    checks = {
        "accuracy": (
            metrics["accuracy"] >= MIN_ACCURACY
        ),

        "precision": (
            metrics["precision"] >= MIN_PRECISION
        ),

        "recall": (
            metrics["recall"] >= MIN_RECALL
        ),

        "f1_score": (
            metrics["f1_score"] >= MIN_F1
        )
    }

    passed = all(checks.values())

    # Create report
    report = {
        "model": MODEL_PATH.name,
        "metrics": metrics,

        "thresholds": {
            "accuracy": MIN_ACCURACY,
            "precision": MIN_PRECISION,
            "recall": MIN_RECALL,
            "f1_score": MIN_F1
        },

        "checks": checks,

        "quality_gate": (
            "PASSED" if passed else "FAILED"
        )
    }

    # Save JSON report
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

    # Display results
    print("\n========== MODEL METRICS ==========")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\n========== QUALITY CHECKS ==========")

    for name, status in checks.items():

        result = "PASS" if status else "FAIL"

        print(f"{name}: {result}")

    print("\n======================================")

    print(
        "QUALITY GATE:",
        report["quality_gate"]
    )

    print("Report saved:", REPORT_PATH)

    print("======================================")

    # Fail CI when quality requirements are not met
    if not passed:
        raise SystemExit(1)


# ==========================================
# PROGRAM ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()
