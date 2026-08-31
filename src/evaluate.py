import sys
from pathlib import Path

import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from configs.config import (
    PROCESSED_DATA,
    MODEL_PATH,
    CONFUSION_MATRIX,
    CLASSIFICATION_REPORT
)


def evaluate_model():

    # Load processed data
    df = pd.read_csv(PROCESSED_DATA)

    X = df.drop("income", axis=1)

    y = df["income"].map({
        "<=50K": 0,
        ">50K": 1
    })

    # Load model
    model = joblib.load(MODEL_PATH)

    # Predictions
    predictions = model.predict(X)

    # Save predictions
    prediction_df = pd.DataFrame({
        "Actual": y,
        "Predicted": predictions
    })

    prediction_df.to_csv(
        Path(__file__).resolve().parent.parent / "outputs" / "predictions.csv",
        index=False
    )

    # Classification report
    report = classification_report(
        y,
        predictions,
        target_names=["<=50K", ">50K"]
    )

    print(report)

    # Save classification report
    with open(CLASSIFICATION_REPORT, "w") as file:
        file.write(report)

    # Confusion matrix
    cm = confusion_matrix(
        y,
        predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["<=50K", ">50K"]
    )

    display.plot()

    plt.title("Confusion Matrix")

    plt.savefig(
        CONFUSION_MATRIX,
        bbox_inches="tight"
    )

    plt.close()

    print("Evaluation completed.")


if __name__ == "__main__":
    evaluate_model()