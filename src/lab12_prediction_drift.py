
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "monitoring" / "drift"
MODEL_PATH = BASE_DIR / "artifacts" / "model_registry" / "production_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"

OUTPUT_DIR = BASE_DIR / "outputs" / "drift"

BASELINE_PATH = DATA_DIR / "baseline_test_features.csv"
EVOLVED_PATH = DATA_DIR / "evolved_data.csv"

REPORT_PATH = OUTPUT_DIR / "prediction_drift_report.json"

DRIFT_THRESHOLD = 0.10


def main():
    print("=" * 65)
    print("LAB 12 - ADULT INCOME PREDICTION DRIFT DETECTION")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in [
        BASELINE_PATH,
        EVOLVED_PATH,
        MODEL_PATH,
        PREPROCESSOR_PATH
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    baseline = pd.read_csv(BASELINE_PATH)
    evolved = pd.read_csv(EVOLVED_PATH)

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    baseline_transformed = preprocessor.transform(baseline)
    evolved_transformed = preprocessor.transform(evolved)

    baseline_predictions = model.predict(baseline_transformed)
    evolved_predictions = model.predict(evolved_transformed)

    baseline_probabilities = model.predict_proba(
        baseline_transformed
    )[:, 1]

    evolved_probabilities = model.predict_proba(
        evolved_transformed
    )[:, 1]

    prediction_statistic, prediction_pvalue = ks_2samp(
        baseline_predictions,
        evolved_predictions
    )

    probability_statistic, probability_pvalue = ks_2samp(
        baseline_probabilities,
        evolved_probabilities
    )

    prediction_drift = prediction_statistic >= DRIFT_THRESHOLD
    probability_drift = probability_statistic >= DRIFT_THRESHOLD

    summary = {
        "project": "Adult Income Prediction",
        "baseline_rows": int(len(baseline)),
        "evolved_rows": int(len(evolved)),
        "threshold": DRIFT_THRESHOLD,
        "prediction_drift": {
            "ks_statistic": float(prediction_statistic),
            "p_value": float(prediction_pvalue),
            "drift_detected": bool(prediction_drift)
        },
        "probability_drift": {
            "ks_statistic": float(probability_statistic),
            "p_value": float(probability_pvalue),
            "drift_detected": bool(probability_drift)
        },
        "baseline_positive_prediction_rate": float(
            np.mean(baseline_predictions)
        ),
        "evolved_positive_prediction_rate": float(
            np.mean(evolved_predictions)
        ),
        "baseline_mean_probability": float(
            np.mean(baseline_probabilities)
        ),
        "evolved_mean_probability": float(
            np.mean(evolved_probabilities)
        ),
        "overall_prediction_drift": bool(
            prediction_drift or probability_drift
        )
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\nPREDICTION DRIFT RESULTS")
    print(json.dumps(summary, indent=4))

    print("\nSaved report:", REPORT_PATH)
    print("\n[SUCCESS] Prediction drift detection completed.")


if __name__ == "__main__":
    main()
