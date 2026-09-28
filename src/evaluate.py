import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


def evaluate_model():

    print("=========================================")
    print("Evaluating Adult Income Model")
    print("=========================================")

    X_test = np.load(
        "data/processed/X_test_final.npy"
    )

    y_test = np.load(
        "data/processed/y_test.npy"
    )

    model = joblib.load(
        "models/random_forest_baseline.pkl"
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test, y_prob
        )
    }

    print("\nModel Metrics:")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        ax=ax
    )

    ax.set_title("Adult Income - Confusion Matrix")

    fig.savefig(
        "outputs/confusion_matrix.png",
        bbox_inches="tight"
    )

    plt.close(fig)

    # ROC curve
    fig, ax = plt.subplots(figsize=(6, 5))

    RocCurveDisplay.from_predictions(
        y_test,
        y_prob,
        ax=ax
    )

    ax.set_title("Adult Income - ROC Curve")

    fig.savefig(
        "outputs/roc_curve.png",
        bbox_inches="tight"
    )

    plt.close(fig)

    with open(
        "reports/evaluation_report.json",
        "w"
    ) as f:
        json.dump(metrics, f, indent=4)

    print("\nEvaluation completed.")


if __name__ == "__main__":
    evaluate_model()