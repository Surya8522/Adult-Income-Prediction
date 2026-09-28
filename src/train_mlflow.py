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


def train_and_track(
    run_name="RandomForest_Adult_Income",
    params=None
):

    print("=========================================")
    print("Adult Income - MLflow Tracking")
    print("=========================================")

    # --------------------------------------------------
    # 1. Model parameters
    # --------------------------------------------------
    if params is None:
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "class_weight": "balanced"
        }

    # --------------------------------------------------
    # 2. Check processed data
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
                "Run Lab 3 preprocessing first."
            )

    # --------------------------------------------------
    # 3. Load processed data
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

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape : {y_test.shape}")

    # --------------------------------------------------
    # 4. Set MLflow experiment
    # --------------------------------------------------
    mlflow.set_experiment(
        "Adult_Income_Prediction"
    )

    print(f"\nStarting MLflow run: {run_name}")

    # --------------------------------------------------
    # 5. Start MLflow run
    # --------------------------------------------------
    with mlflow.start_run(
        run_name=run_name
    ):

        # ----------------------------------------------
        # 6. Log parameters
        # ----------------------------------------------
        mlflow.log_params(params)

        mlflow.log_param(
            "model_family",
            "RandomForest"
        )

        mlflow.log_param(
            "dataset",
            "Adult Income"
        )

        mlflow.log_param(
            "target",
            "income"
        )

        # ----------------------------------------------
        # 7. Create model
        # ----------------------------------------------
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"],
            class_weight=params["class_weight"],
            n_jobs=-1
        )

        # ----------------------------------------------
        # 8. Train
        # ----------------------------------------------
        print("\nTraining Random Forest...")

        model.fit(
            X_train,
            y_train
        )

        print("Training completed.")

        # ----------------------------------------------
        # 9. Predictions
        # ----------------------------------------------
        y_pred = model.predict(X_test)

        y_prob = model.predict_proba(
            X_test
        )[:, 1]

        # ----------------------------------------------
        # 10. Calculate metrics
        # ----------------------------------------------
        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }

        # ----------------------------------------------
        # 11. Print metrics
        # ----------------------------------------------
        print("\n=========================================")
        print("Model Performance")
        print("=========================================")

        print(
            f"Accuracy : {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall   : {recall:.4f}"
        )

        print(
            f"F1 Score : {f1:.4f}"
        )

        print(
            f"ROC-AUC  : {roc_auc:.4f}"
        )

        # ----------------------------------------------
        # 12. Log metrics to MLflow
        # ----------------------------------------------
        mlflow.log_metrics(metrics)

        # ----------------------------------------------
        # 13. Save local model
        # ----------------------------------------------
        os.makedirs(
            "models",
            exist_ok=True
        )

        local_model_path = (
            "models/random_forest_model.pkl"
        )

        joblib.dump(
            model,
            local_model_path
        )

        print(
            f"\nLocal model saved: "
            f"{local_model_path}"
        )

        # ----------------------------------------------
        # 14. Log preprocessing metadata
        # ----------------------------------------------
        metadata_path = (
            "data/processed/"
            "dataset_metadata.json"
        )

        if os.path.exists(metadata_path):

            mlflow.log_artifact(
                metadata_path,
                artifact_path="metadata"
            )

            print(
                "Dataset metadata logged."
            )

        # ----------------------------------------------
        # 15. Log model to MLflow
        # ----------------------------------------------
        #
        # IMPORTANT:
        # RandomForest contains sklearn.tree._tree.Tree.
        # New MLflow versions may require this type
        # to be explicitly trusted by skops.
        #
        print("\nLogging model to MLflow...")

        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        print(
            "Model successfully logged to MLflow."
        )

        # ----------------------------------------------
        # 16. Display run information
        # ----------------------------------------------
        run = mlflow.active_run()

        if run is not None:

            print("\n=========================================")
            print("MLflow Run Information")
            print("=========================================")

            print(
                f"Run ID: {run.info.run_id}"
            )

            print(
                f"Experiment ID: "
                f"{run.info.experiment_id}"
            )

            print(
                f"Run Name: "
                f"{run.data.tags.get('mlflow.runName')}"
            )

    print("\n=========================================")
    print("MLflow Tracking Completed Successfully")
    print("=========================================")


if __name__ == "__main__":

    train_and_track(
        run_name="RandomForest_Adult_Income"
    )