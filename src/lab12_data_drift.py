
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "monitoring" / "drift"
OUTPUT_DIR = BASE_DIR / "outputs" / "drift"

BASELINE_PATH = DATA_DIR / "baseline_test_features.csv"
EVOLVED_PATH = DATA_DIR / "evolved_data.csv"

REPORT_PATH = OUTPUT_DIR / "data_drift_report.csv"
SUMMARY_PATH = OUTPUT_DIR / "data_drift_summary.json"

NUMERIC_FEATURES = [
    "age",
    "hours.per.week",
    "capital.gain"
]

CATEGORICAL_FEATURES = [
    "workclass"
]

DRIFT_THRESHOLD = 0.10


def calculate_js_divergence(baseline, evolved):
    baseline_counts = baseline.value_counts(normalize=True)
    evolved_counts = evolved.value_counts(normalize=True)

    categories = baseline_counts.index.union(evolved_counts.index)

    p = baseline_counts.reindex(categories, fill_value=0).to_numpy()
    q = evolved_counts.reindex(categories, fill_value=0).to_numpy()

    # Jensen-Shannon distance is the square root of JS divergence.
    distance = jensenshannon(p, q)

    return float(distance ** 2)


def main():
    print("=" * 65)
    print("LAB 12 - ADULT INCOME DATA DRIFT DETECTION")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(BASELINE_PATH)

    if not EVOLVED_PATH.exists():
        raise FileNotFoundError(EVOLVED_PATH)

    baseline = pd.read_csv(BASELINE_PATH)
    evolved = pd.read_csv(EVOLVED_PATH)

    if list(baseline.columns) != list(evolved.columns):
        raise ValueError("Baseline and evolved columns do not match.")

    results = []

    for feature in NUMERIC_FEATURES:
        statistic, p_value = ks_2samp(
            baseline[feature].dropna(),
            evolved[feature].dropna()
        )

        drift_detected = statistic >= DRIFT_THRESHOLD

        results.append({
            "feature": feature,
            "feature_type": "numeric",
            "drift_score": float(statistic),
            "p_value": float(p_value),
            "threshold": DRIFT_THRESHOLD,
            "drift_detected": bool(drift_detected)
        })

    for feature in CATEGORICAL_FEATURES:
        score = calculate_js_divergence(
            baseline[feature].fillna("Missing").astype(str),
            evolved[feature].fillna("Missing").astype(str)
        )

        drift_detected = score >= DRIFT_THRESHOLD

        results.append({
            "feature": feature,
            "feature_type": "categorical",
            "drift_score": score,
            "p_value": None,
            "threshold": DRIFT_THRESHOLD,
            "drift_detected": bool(drift_detected)
        })

    report = pd.DataFrame(results)
    report.to_csv(REPORT_PATH, index=False)

    drifted_features = report.loc[
        report["drift_detected"], "feature"
    ].tolist()

    summary = {
        "project": "Adult Income Prediction",
        "baseline_rows": int(len(baseline)),
        "evolved_rows": int(len(evolved)),
        "features_checked": int(len(report)),
        "drifted_features": drifted_features,
        "drift_detected": bool(len(drifted_features) > 0),
        "threshold": DRIFT_THRESHOLD,
        "report_file": str(REPORT_PATH)
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\nDATA DRIFT RESULTS")
    print(report.to_string(index=False))

    print("\nDrifted features:", drifted_features)
    print("\nSaved report:", REPORT_PATH)
    print("Saved summary:", SUMMARY_PATH)

    print("\n[SUCCESS] Data drift detection completed.")


if __name__ == "__main__":
    main()
