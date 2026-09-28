import os
import numpy as np


def validate_outputs():

    print("=========================================")
    print("Validating Preprocessing Outputs")
    print("=========================================")

    required_files = [
        "data/processed/X_train_final.npy",
        "data/processed/X_test_final.npy",
        "data/processed/y_train.npy",
        "data/processed/y_test.npy",
        "data/processed/dataset_metadata.json",
        "models/preprocessor.pkl"
    ]

    all_valid = True

    for file in required_files:

        if os.path.exists(file):
            print(f"[OK] {file}")
        else:
            print(f"[ERROR] Missing: {file}")
            all_valid = False

    if not all_valid:
        raise FileNotFoundError(
            "One or more required outputs are missing."
        )

    X_train = np.load(
        "data/processed/X_train_final.npy"
    )

    X_test = np.load(
        "data/processed/X_test_final.npy"
    )

    y_train = np.load(
        "data/processed/y_train.npy"
    )

    y_test = np.load(
        "data/processed/y_test.npy"
    )

    if len(X_train) != len(y_train):
        raise ValueError(
            "Training X/y size mismatch."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "Testing X/y size mismatch."
        )

    print("\nAll preprocessing outputs are valid.")


if __name__ == "__main__":
    validate_outputs()