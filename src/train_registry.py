import os
import joblib
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


MODEL_NAME = "Adult_Income_Production_Model"


def train_and_register():

    print("=========================================")
    print("Adult Income - Model Registry")
    print("=========================================")

    # --------------------------------------------------
    # 1. Check processed files
    # --------------------------------------------------
    required_files = [
        "data/processed/X_train_final.npy",
        "data/processed/X_test_final.npy",
        "data/processed/y_train.npy",
        "data/processed/y_test.npy"
    ]

    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(
                f"Required file not found: {file}\n"
                "Run Lab 3 first."
            )

    # --------------------------------------------------
    # 2. Load data
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 3. MLflow experiment
    # --------------------------------------------------
    mlflow.set_experiment(
        "Adult_Income_Prediction"
    )

    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "class_weight": "balanced"
    }

    # --------------------------------------------------
    # 4. Start run
    # --------------------------------------------------
    with mlflow.start_run(
        run_name="Adult_Income_Registry_Model"
    ):

        print("\nTraining production model...")

        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"],
            class_weight=params["class_weight"],
            n_jobs=-1
        )

        model.fit(
            X_train,
            y_train
        )

        # --------------------------------------------------
        # 5. Predictions
        # --------------------------------------------------
        y_pred = model.predict(
            X_test
        )

        y_prob = model.predict_proba(
            X_test
        )[:, 1]

        # --------------------------------------------------
        # 6. Metrics
        # --------------------------------------------------
        metrics = {
            "accuracy": accuracy_score(
                y_test,
                y_pred
            ),

            "precision": precision_score(
                y_test,
                y_pred,
                zero_division=0
            ),

            "recall": recall_score(
                y_test,
                y_pred,
                zero_division=0
            ),

            "f1_score": f1_score(
                y_test,
                y_pred,
                zero_division=0
            ),

            "roc_auc": roc_auc_score(
                y_test,
                y_prob
            )
        }

        # --------------------------------------------------
        # 7. Log parameters
        # --------------------------------------------------
        mlflow.log_params(params)

        mlflow.log_param(
            "dataset",
            "Adult Income"
        )

        mlflow.log_param(
            "target",
            "income"
        )

        mlflow.log_param(
            "model_family",
            "RandomForest"
        )

        # --------------------------------------------------
        # 8. Log metrics
        # --------------------------------------------------
        mlflow.log_metrics(
            metrics
        )

        print("\nModel Metrics:")

        for name, value in metrics.items():
            print(
                f"{name}: {value:.4f}"
            )

        # --------------------------------------------------
        # 9. Register model
        # --------------------------------------------------
        print(
            "\nRegistering model with MLflow..."
        )

        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=MODEL_NAME,
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        # --------------------------------------------------
        # 10. Save local model
        # --------------------------------------------------
        os.makedirs(
            "models",
            exist_ok=True
        )

        joblib.dump(
            model,
            "models/random_forest_model.pkl"
        )

        print(
            "\nLocal model saved:"
        )

        print(
            "models/random_forest_model.pkl"
        )

        # --------------------------------------------------
        # 11. Run information
        # --------------------------------------------------
        run = mlflow.active_run()

        if run is not None:

            print("\n=========================================")
            print("Registry Run Information")
            print("=========================================")

            print(
                f"Run ID: {run.info.run_id}"
            )

            print(
                f"Model Name: {MODEL_NAME}"
            )

    print("\n=========================================")
    print("Model Registration Completed")
    print("=========================================")


if __name__ == "__main__":
    train_and_register()