import json
import numpy as np


def validate_reproducibility():

    print("=========================================")
    print("Checking Reproducibility")
    print("=========================================")

    files = [
        "data/processed/X_train_final.npy",
        "data/processed/X_test_final.npy",
        "data/processed/y_train.npy",
        "data/processed/y_test.npy"
    ]

    report = {}

    for file in files:

        try:
            data = np.load(file)

            report[file] = {
                "exists": True,
                "shape": list(data.shape)
            }

            print(
                f"[OK] {file} -> {data.shape}"
            )

        except Exception as e:

            report[file] = {
                "exists": False,
                "error": str(e)
            }

            print(
                f"[ERROR] {file}: {e}"
            )

    report["random_state"] = 42
    report["dataset"] = "Adult Income"

    with open(
        "reports/reproducibility_report.json",
        "w"
    ) as f:
        json.dump(report, f, indent=4)

    print("\nReproducibility validation completed.")


if __name__ == "__main__":
    validate_reproducibility()