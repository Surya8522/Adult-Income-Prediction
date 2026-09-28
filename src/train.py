import os
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier


def train_model():

    print("=========================================")
    print("Training Adult Income Model")
    print("=========================================")

    X_train = np.load(
        "data/processed/X_train_final.npy"
    )

    y_train = np.load(
        "data/processed/y_train.npy"
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        model,
        "models/random_forest_baseline.pkl"
    )

    print("Model saved:")
    print("models/random_forest_baseline.pkl")


if __name__ == "__main__":
    train_model()