
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "monitoring" / "drift"

FEATURES_PATH = DATA_DIR / "evolved_data.csv"
LABELS_PATH = DATA_DIR / "baseline_test_labels.npy"

OUTPUT_PATH = DATA_DIR / "evolved_labeled_data.csv"
REPORT_PATH = DATA_DIR / "concept_drift_generation_report.json"

FLIP_PERCENTAGE = 0.20
RANDOM_STATE = 42


def main():
    print("=" * 65)
    print("LAB 12 - GENERATE ADULT INCOME CONCEPT DRIFT DATA")
    print("=" * 65)

    features = pd.read_csv(FEATURES_PATH)
    labels = np.load(LABELS_PATH).astype(int)

    if len(features) != len(labels):
        raise ValueError("Feature and label row counts do not match.")

    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Income labels must be binary: 0 or 1.")

    evolved_labels = labels.copy()

    rng = np.random.default_rng(RANDOM_STATE)
    flip_count = int(len(evolved_labels) * FLIP_PERCENTAGE)

    selected_indices = rng.choice(
        len(evolved_labels),
        size=flip_count,
        replace=False
    )

    evolved_labels[selected_indices] = (
        1 - evolved_labels[selected_indices]
    )

    result = features.copy()
    result["income"] = evolved_labels

    result.to_csv(OUTPUT_PATH, index=False)

    report = {
        "project": "Adult Income Prediction",
        "purpose": "Simulated concept drift for Lab 12",
        "total_rows": int(len(result)),
        "labels_flipped": int(flip_count),
        "flip_percentage": FLIP_PERCENTAGE,
        "random_state": RANDOM_STATE,
        "baseline_positive_labels": int(np.sum(labels)),
        "evolved_positive_labels": int(np.sum(evolved_labels)),
        "output_file": str(OUTPUT_PATH)
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nOriginal rows:", len(features))
    print("Labels changed:", flip_count)
    print("Evolved dataset shape:", result.shape)
    print("\nSaved:", OUTPUT_PATH)
    print("Report:", REPORT_PATH)

    print("\n[SUCCESS] Concept drift dataset generated.")


if __name__ == "__main__":
    main()
