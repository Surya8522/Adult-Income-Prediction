
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "monitoring" / "drift"

BASELINE_PATH = DATA_DIR / "baseline_test_features.csv"
EVOLVED_PATH = DATA_DIR / "evolved_data.csv"
REPORT_PATH = DATA_DIR / "evolved_data_generation_report.json"


def main():

    print("=" * 65)
    print("LAB 12 - GENERATE ADULT INCOME DATA DRIFT")
    print("=" * 65)

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(BASELINE_PATH)

    df = pd.read_csv(BASELINE_PATH)

    evolved = df.copy()

    # Simulate an increase in age.
    evolved["age"] = (
        evolved["age"] + 5
    ).clip(upper=90)

    # Simulate increased working hours.
    evolved["hours.per.week"] = (
        evolved["hours.per.week"] + 5
    ).clip(upper=99)

    # Simulate changes in capital gain.
    evolved["capital.gain"] = (
        evolved["capital.gain"] * 1.5
    ).round()

    # Simulate a categorical distribution change.
    private_mask = evolved["workclass"] == "Private"
    evolved.loc[private_mask, "workclass"] = "Self-emp-not-inc"

    evolved.to_csv(EVOLVED_PATH, index=False)

    report = {
        "project": "Adult Income Prediction",
        "purpose": "Simulated feature drift for Lab 12",
        "baseline_rows": int(len(df)),
        "evolved_rows": int(len(evolved)),
        "changed_features": [
            "age",
            "hours.per.week",
            "capital.gain",
            "workclass"
        ],
        "changes": {
            "age": "Increased by 5 years, capped at 90",
            "hours.per.week": "Increased by 5, capped at 99",
            "capital.gain": "Multiplied by 1.5",
            "workclass": "Private changed to Self-emp-not-inc"
        }
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nBaseline shape:", df.shape)
    print("Evolved shape :", evolved.shape)

    print("\nChanged features:")
    for feature in report["changed_features"]:
        print("-", feature)

    print("\nSaved:", EVOLVED_PATH)
    print("Report:", REPORT_PATH)

    print("\n[SUCCESS] Evolved monitoring dataset generated.")


if __name__ == "__main__":
    main()
