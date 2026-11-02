
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from validate_data import validate_data
from validate_outputs import validate_outputs


BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = REPORT_DIR / "lab9_failure_report.json"

REQUIRED_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education.num",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
    "native.country",
    "income"
]


def run_failure_test(name, expected_exception, test_function):
    print(f"\nRunning: {name}")

    try:
        test_function()

        print("[FAIL] Expected failure was not detected.")

        return {
            "test": name,
            "status": "FAIL",
            "details": "Expected exception was not raised."
        }

    except expected_exception as error:
        print(f"[PASS] Expected failure detected: {error}")

        return {
            "test": name,
            "status": "PASS",
            "details": str(error)
        }

    except Exception as error:
        print(f"[FAIL] Unexpected exception: {error}")

        return {
            "test": name,
            "status": "FAIL",
            "details": f"Unexpected exception: {error}"
        }


def create_dataset_folder():
    Path("data/raw").mkdir(parents=True, exist_ok=True)


def test_missing_dataset():
    with tempfile.TemporaryDirectory() as temp:
        old_dir = os.getcwd()

        try:
            os.chdir(temp)
            create_dataset_folder()

            validate_data()

        finally:
            os.chdir(old_dir)


def test_missing_columns():
    with tempfile.TemporaryDirectory() as temp:
        old_dir = os.getcwd()

        try:
            os.chdir(temp)
            create_dataset_folder()

            df = pd.DataFrame({
                "age": [25],
                "income": ["<=50K"]
            })

            df.to_csv("data/raw/adult.csv", index=False)

            validate_data()

        finally:
            os.chdir(old_dir)


def test_empty_dataset():
    with tempfile.TemporaryDirectory() as temp:
        old_dir = os.getcwd()

        try:
            os.chdir(temp)
            create_dataset_folder()

            df = pd.DataFrame(columns=REQUIRED_COLUMNS)

            df.to_csv("data/raw/adult.csv", index=False)

            validate_data()

        finally:
            os.chdir(old_dir)


def test_missing_processed_outputs():
    with tempfile.TemporaryDirectory() as temp:
        old_dir = os.getcwd()

        try:
            os.chdir(temp)

            validate_outputs()

        finally:
            os.chdir(old_dir)


def test_train_test_size_mismatch():
    with tempfile.TemporaryDirectory() as temp:
        old_dir = os.getcwd()

        try:
            os.chdir(temp)

            Path("data/processed").mkdir(parents=True)
            Path("models").mkdir(parents=True)

            np.save(
                "data/processed/X_train_final.npy",
                np.array([[1, 2], [3, 4], [5, 6]])
            )

            np.save(
                "data/processed/X_test_final.npy",
                np.array([[7, 8], [9, 10]])
            )

            np.save(
                "data/processed/y_train.npy",
                np.array([0, 1])
            )

            np.save(
                "data/processed/y_test.npy",
                np.array([0, 1])
            )

            Path("data/processed/dataset_metadata.json").touch()
            Path("models/preprocessor.pkl").touch()

            validate_outputs()

        finally:
            os.chdir(old_dir)


def main():
    print("=" * 55)
    print("ADULT INCOME - LAB 9 FAILURE TESTING")
    print("=" * 55)

    results = []

    results.append(
        run_failure_test(
            "Missing dataset",
            FileNotFoundError,
            test_missing_dataset
        )
    )

    results.append(
        run_failure_test(
            "Missing required dataset columns",
            ValueError,
            test_missing_columns
        )
    )

    results.append(
        run_failure_test(
            "Empty dataset",
            ValueError,
            test_empty_dataset
        )
    )

    results.append(
        run_failure_test(
            "Missing preprocessing outputs",
            FileNotFoundError,
            test_missing_processed_outputs
        )
    )

    results.append(
        run_failure_test(
            "Training X/y size mismatch",
            ValueError,
            test_train_test_size_mismatch
        )
    )

    passed = sum(
        result["status"] == "PASS"
        for result in results
    )

    failed = len(results) - passed

    report = {
        "project": "Adult Income Prediction",
        "lab": "Lab 9 - Failure Testing",
        "total_tests": len(results),
        "passed": passed,
        "failed": failed,
        "overall_status": "PASSED" if failed == 0 else "FAILED",
        "test_results": results
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 55)
    print("LAB 9 FAILURE TEST SUMMARY")
    print("=" * 55)

    print(f"Total tests : {len(results)}")
    print(f"Passed      : {passed}")
    print(f"Failed      : {failed}")
    print(f"Status      : {report['overall_status']}")

    print(f"\nReport saved: {REPORT_PATH}")

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
