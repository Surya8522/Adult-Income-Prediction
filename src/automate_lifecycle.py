import os
import json
import joblib
import numpy as np
import pandas as pd
import mlflow

from mlflow.tracking import MlflowClient

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)

REGISTRY_DIR = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "model_registry"
)

REPORT_DIR = os.path.join(
    PROJECT_ROOT,
    "reports"
)


# Create required directories

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    REGISTRY_DIR,
    exist_ok=True
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)


# ============================================================
# 2. MLflow MODEL REGISTRY CONFIGURATION
# ============================================================

MODEL_NAME = "Adult_Income_Production_Model"

print("\n" + "=" * 60)
print("ADULT INCOME MODEL LIFECYCLE")
print("=" * 60)

print("\nMLflow Registered Model:")
print(MODEL_NAME)


# ============================================================
# 3. LOAD TEST DATA
# ============================================================

print("\n" + "=" * 60)
print("STEP 1: LOADING TEST DATA")
print("=" * 60)


X_test_path = os.path.join(
    DATA_DIR,
    "X_test_final.npy"
)

y_test_path = os.path.join(
    DATA_DIR,
    "y_test.npy"
)


if not os.path.exists(X_test_path):

    print(
        "[ERROR] X_test_final.npy not found."
    )

    raise SystemExit(1)


if not os.path.exists(y_test_path):

    print(
        "[ERROR] y_test.npy not found."
    )

    raise SystemExit(1)


X_test = np.load(
    X_test_path
)

y_test = np.load(
    y_test_path
)


print(
    "X_test shape :",
    X_test.shape
)

print(
    "y_test shape :",
    y_test.shape
)


# ============================================================
# 4. LOAD TRAINED MODEL
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: LOADING TRAINED MODEL")
print("=" * 60)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "random_forest_model.pkl"
)


if not os.path.exists(
    MODEL_PATH
):

    print(
        "[ERROR] Random Forest model not found:"
    )

    print(
        MODEL_PATH
    )

    raise SystemExit(1)


model = joblib.load(
    MODEL_PATH
)


print(
    "Random Forest loaded successfully."
)

print(
    "Model path:",
    MODEL_PATH
)


# ============================================================
# 5. MODEL PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: GENERATING PREDICTIONS")
print("=" * 60)


predictions = model.predict(
    X_test
)


print(
    "Predictions generated successfully."
)

print(
    "Number of predictions:",
    len(predictions)
)


# ============================================================
# 6. PREDICTION PROBABILITIES
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: GENERATING PREDICTION PROBABILITIES")
print("=" * 60)


if hasattr(
    model,
    "predict_proba"
):

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    print(
        "Prediction probabilities generated."
    )

else:

    probabilities = None

    print(
        "[WARNING] Model does not support predict_proba()."
    )


# ============================================================
# 7. CALCULATE CLASSIFICATION METRICS
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: CALCULATING MODEL METRICS")
print("=" * 60)


accuracy = accuracy_score(
    y_test,
    predictions
)


precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)


recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)


f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)


if probabilities is not None:

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

else:

    roc_auc = 0.0


print(
    "\nModel Metrics:"
)

print(
    "Accuracy :",
    round(
        accuracy,
        4
    )
)

print(
    "Precision:",
    round(
        precision,
        4
    )
)

print(
    "Recall   :",
    round(
        recall,
        4
    )
)

print(
    "F1 Score :",
    round(
        f1,
        4
    )
)

print(
    "ROC-AUC  :",
    round(
        roc_auc,
        4
    )
)


# ============================================================
# 8. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: GENERATING CLASSIFICATION REPORT")
print("=" * 60)


classification_report_text = classification_report(
    y_test,
    predictions,
    zero_division=0
)


print(
    classification_report_text
)


classification_report_path = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt"
)


with open(
    classification_report_path,
    "w"
) as file:

    file.write(
        classification_report_text
    )


print(
    "Classification report saved:"
)

print(
    classification_report_path
)


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: GENERATING CONFUSION MATRIX")
print("=" * 60)


cm = confusion_matrix(
    y_test,
    predictions
)


print(
    "\nConfusion Matrix:"
)

print(
    cm
)


# ============================================================
# 10. MODEL COMPARISON TABLE
# ============================================================

print("\n" + "=" * 60)
print("STEP 8: CREATING MODEL COMPARISON")
print("=" * 60)


results = [

    {
        "model":
            "Random Forest",

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1_score":
            f1,

        "roc_auc":
            roc_auc
    }

]


results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    by="f1_score",
    ascending=False
)


print(
    "\nModel Comparison:"
)

print(
    results_df.to_string(
        index=False
    )
)


comparison_path = os.path.join(
    OUTPUT_DIR,
    "model_comparison.csv"
)


results_df.to_csv(
    comparison_path,
    index=False
)


print(
    "\nComparison table saved:"
)

print(
    comparison_path
)


# ============================================================
# 11. CONNECT TO MLFLOW
# ============================================================

print("\n" + "=" * 60)
print("STEP 9: CONNECTING TO MLFLOW MODEL REGISTRY")
print("=" * 60)


client = MlflowClient()


print(
    "MLflow Model Registry connected."
)


# ============================================================
# 12. CHECK REGISTERED MODEL
# ============================================================

print("\n" + "=" * 60)
print("STEP 10: CHECKING REGISTERED MODEL")
print("=" * 60)


try:

    registered_model = client.get_registered_model(
        MODEL_NAME
    )

    print(
        "Registered model found:"
    )

    print(
        "Name:",
        registered_model.name
    )

except Exception:

    print(
        "[ERROR] Registered model not found."
    )

    print(
        "Expected model name:",
        MODEL_NAME
    )

    print(
        "Run train_registry.py first."
    )

    raise SystemExit(1)


# ============================================================
# 13. GET MODEL VERSIONS
# ============================================================

print("\n" + "=" * 60)
print("STEP 11: GETTING MODEL VERSIONS")
print("=" * 60)


versions = client.search_model_versions(
    f"name='{MODEL_NAME}'"
)


if not versions:

    print(
        "[ERROR] No model versions found."
    )

    raise SystemExit(1)


print(
    "Registered model versions:"
)


for version in versions:

    print(
        "Version:",
        version.version,
        "| Stage:",
        version.current_stage,
        "| Run ID:",
        version.run_id
    )


# ============================================================
# 14. FIND LATEST VERSION
# ============================================================

print("\n" + "=" * 60)
print("STEP 12: FINDING LATEST MODEL VERSION")
print("=" * 60)


latest_version = max(
    versions,
    key=lambda v: int(v.version)
)


print(
    "Latest model version:",
    latest_version.version
)


print(
    "Current stage:",
    latest_version.current_stage
)


print(
    "Run ID:",
    latest_version.run_id
)


# ============================================================
# 15. FIND CURRENT PRODUCTION MODEL
# ============================================================

print("\n" + "=" * 60)
print("STEP 13: CHECKING PRODUCTION MODEL")
print("=" * 60)


production_versions = [

    version

    for version in versions

    if version.current_stage == "Production"

]


if production_versions:

    current_production = max(
        production_versions,
        key=lambda v: int(v.version)
    )

    print(
        "Current Production version:",
        current_production.version
    )

else:

    current_production = None

    print(
        "No model is currently in Production."
    )


# ============================================================
# 16. PROMOTE MODEL TO PRODUCTION
# ============================================================

print("\n" + "=" * 60)
print("STEP 14: MODEL LIFECYCLE TRANSITION")
print("=" * 60)


if current_production is None:

    print(
        "Promoting latest model to Production..."
    )

    client.transition_model_version_stage(

        name=MODEL_NAME,

        version=latest_version.version,

        stage="Production",

        archive_existing_versions=False

    )

    print(
        "[SUCCESS] Model promoted to Production."
    )


elif int(
    latest_version.version
) > int(
    current_production.version
):

    print(
        "Newer model version detected."
    )

    print(
        "Current Production:",
        current_production.version
    )

    print(
        "New version:",
        latest_version.version
    )


    print(
        "\nPromoting new version..."
    )


    client.transition_model_version_stage(

        name=MODEL_NAME,

        version=latest_version.version,

        stage="Production",

        archive_existing_versions=False

    )


    print(
        "[SUCCESS] New version promoted."
    )


    print(
        "\nArchiving previous Production model..."
    )


    client.transition_model_version_stage(

        name=MODEL_NAME,

        version=current_production.version,

        stage="Archived",

        archive_existing_versions=False

    )


    print(
        "[SUCCESS] Previous Production model archived."
    )


else:

    print(
        "Latest version is already Production."
    )


# ============================================================
# 17. VERIFY PRODUCTION MODEL
# ============================================================

print("\n" + "=" * 60)
print("STEP 15: VERIFYING PRODUCTION MODEL")
print("=" * 60)


updated_versions = client.search_model_versions(
    f"name='{MODEL_NAME}'"
)


production_versions = [

    version

    for version in updated_versions

    if version.current_stage == "Production"

]


if not production_versions:

    print(
        "[ERROR] No model found in Production stage!"
    )

    raise SystemExit(1)


production_model = max(
    production_versions,
    key=lambda v: int(v.version)
)


print(
    "\nProduction Model Verified:"
)


print(
    "Model Name :",
    MODEL_NAME
)


print(
    "Version    :",
    production_model.version
)


print(
    "Stage      :",
    production_model.current_stage
)


print(
    "Run ID     :",
    production_model.run_id
)


print(
    "Source     :",
    production_model.source
)


# ============================================================
# 18. SAVE LOCAL PRODUCTION MODEL
# ============================================================

print("\n" + "=" * 60)
print("STEP 16: SAVING LOCAL PRODUCTION MODEL")
print("=" * 60)


production_model_path = os.path.join(
    REGISTRY_DIR,
    "production_model.pkl"
)


joblib.dump(
    model,
    production_model_path
)


print(
    "Production model saved:"
)


print(
    production_model_path
)


# ============================================================
# 19. CREATE MODEL METADATA
# ============================================================

print("\n" + "=" * 60)
print("STEP 17: CREATING MODEL METADATA")
print("=" * 60)


metadata = {

    "project":
        "Adult Income Prediction",

    "task":
        "Binary Classification",

    "model_name":
        MODEL_NAME,

    "model_type":
        "Random Forest",

    "version":
        int(
            production_model.version
        ),

    "stage":
        "Production",

    "run_id":
        production_model.run_id,

    "accuracy":
        float(
            accuracy
        ),

    "precision":
        float(
            precision
        ),

    "recall":
        float(
            recall
        ),

    "f1_score":
        float(
            f1
        ),

    "roc_auc":
        float(
            roc_auc
        ),

    "model_file":
        "production_model.pkl"

}


metadata_path = os.path.join(
    REGISTRY_DIR,
    "production_model_metadata.json"
)


with open(
    metadata_path,
    "w"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


print(
    "Metadata saved:"
)


print(
    metadata_path
)


# ============================================================
# 20. CREATE LIFECYCLE REPORT
# ============================================================

print("\n" + "=" * 60)
print("STEP 18: CREATING MODEL LIFECYCLE REPORT")
print("=" * 60)


lifecycle_report = {

    "project":
        "Adult Income Prediction",

    "task":
        "Binary Classification",

    "registered_model":
        MODEL_NAME,

    "production_version":
        int(
            production_model.version
        ),

    "production_stage":
        production_model.current_stage,

    "model_type":
        "Random Forest",

    "run_id":
        production_model.run_id,

    "metrics": {

        "accuracy":
            float(
                accuracy
            ),

        "precision":
            float(
                precision
            ),

        "recall":
            float(
                recall
            ),

        "f1_score":
            float(
                f1
            ),

        "roc_auc":
            float(
                roc_auc
            )

    },

    "files": {

        "model":
            production_model_path,

        "comparison":
            comparison_path,

        "classification_report":
            classification_report_path,

        "metadata":
            metadata_path

    },

    "status":
        "Production"


}


lifecycle_report_path = os.path.join(
    REPORT_DIR,
    "model_lifecycle_report.json"
)


with open(
    lifecycle_report_path,
    "w"
) as file:

    json.dump(
        lifecycle_report,
        file,
        indent=4
    )


print(
    "Lifecycle report saved:"
)


print(
    lifecycle_report_path
)


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ADULT INCOME MODEL LIFECYCLE COMPLETED")
print("=" * 60)


print(
    "\nRegistered Model :",
    MODEL_NAME
)


print(
    "Production Version:",
    production_model.version
)


print(
    "Stage             :",
    production_model.current_stage
)


print(
    "Model Type        :",
    "Random Forest"
)


print(
    "Accuracy          :",
    round(
        accuracy,
        4
    )
)


print(
    "Precision         :",
    round(
        precision,
        4
    )
)


print(
    "Recall            :",
    round(
        recall,
        4
    )
)


print(
    "F1 Score          :",
    round(
        f1,
        4
    )
)


print(
    "ROC-AUC           :",
    round(
        roc_auc,
        4
    )
)


print(
    "\nMLflow Production Model:"
)

print(
    f"models:/{MODEL_NAME}/"
    f"{production_model.version}"
)


print(
    "\nLocal Production Model:"
)

print(
    production_model_path
)


print(
    "\nLifecycle Report:"
)

print(
    lifecycle_report_path
)


print(
    "\nStatus: SUCCESS"
)

print(
    "=" * 60
)