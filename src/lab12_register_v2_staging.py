
import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

REPORT_DIR = BASE_DIR / "outputs" / "retraining"

COMPARISON_REPORT = (
    REPORT_DIR / "model_comparison_v1_v2.json"
)

V2_REPORT = (
    REPORT_DIR / "retrained_model_v2_report.json"
)

MODEL_FILE = (
    BASE_DIR / "models" / "retrained_model_v2.pkl"
)

MODEL_NAME = "Adult_Income_Production_Model"

PRODUCTION_VERSION = "5"

# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main():

    print("=" * 65)
    print("LAB 12 - REGISTER V2 AS STAGING CANDIDATE")
    print("=" * 65)

    client = MlflowClient()

    # Check required files
    for path in [
        COMPARISON_REPORT,
        V2_REPORT,
        MODEL_FILE
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # Load comparison report
    with open(COMPARISON_REPORT, "r", encoding="utf-8") as file:
        comparison = json.load(file)

    # Load V2 training report
    with open(V2_REPORT, "r", encoding="utf-8") as file:
        v2_report = json.load(file)

    # Verify quality gate
    if not comparison.get("all_quality_gates_passed", False):
        raise RuntimeError(
            "V2 failed quality gates. Registration stopped."
        )

    if not comparison.get("f1_improved", False):
        raise RuntimeError(
            "V2 did not improve F1-score. Registration stopped."
        )

    if not comparison.get("eligible_for_staging", False):
        raise RuntimeError(
            "V2 is not eligible for Staging."
        )

    run_id = v2_report.get("mlflow_run_id")

    if not run_id:
        raise ValueError(
            "MLflow Run ID missing from V2 report."
        )

    print("\nQuality checks passed.")
    print("V2 Run ID:", run_id)

    # --------------------------------------------------
    # VERIFY EXISTING PRODUCTION MODEL
    # --------------------------------------------------

    versions = client.search_model_versions(
        filter_string=f"name='{MODEL_NAME}'"
    )

    production_versions = [
        version for version in versions
        if version.current_stage == "Production"
    ]

    if len(production_versions) != 1:
        raise RuntimeError(
            "Expected exactly one Production version. "
            "Registry update stopped."
        )

    production = production_versions[0]

    if str(production.version) != PRODUCTION_VERSION:
        raise RuntimeError(
            f"Expected Production V5, found V{production.version}. "
            "Registry update stopped."
        )

    original_production_run_id = production.run_id

    print("\nExisting Production model verified.")
    print("Production version:", production.version)
    print("Production run ID:", original_production_run_id)

    # --------------------------------------------------
    # VERIFY V2 MLflow ARTIFACT
    # --------------------------------------------------

    run = client.get_run(run_id)

    artifacts = client.list_artifacts(
        run_id,
        "model"
    )

    expected_artifact = "retrained_model_v2.pkl"

    artifact_exists = any(
        artifact.path.endswith(expected_artifact)
        for artifact in artifacts
    )

    if not artifact_exists:
        raise FileNotFoundError(
            "V2 model artifact was not found in MLflow."
        )

    artifact_uri = (
        run.info.artifact_uri.rstrip("/")
        + "/model/"
        + expected_artifact
    )

    print("\nV2 MLflow artifact verified:")
    print(artifact_uri)

    # --------------------------------------------------
    # CHECK EXISTING V2 REGISTRATION
    # --------------------------------------------------

    existing_v2 = [
        version for version in versions
        if version.run_id == run_id
    ]

    if existing_v2:

        v2_version = existing_v2[0]

        print(
            "\nV2 is already registered as version:",
            v2_version.version
        )

    else:

        # Do not overwrite or archive another Staging candidate
        staging_versions = [
            version for version in versions
            if version.current_stage == "Staging"
        ]

        if staging_versions:
            raise RuntimeError(
                "Another model is already in Staging. "
                "Inspect the registry before proceeding."
            )

        print("\nRegistering V2 as a new model version...")

        v2_version = client.create_model_version(
            name=MODEL_NAME,
            source=artifact_uri,
            run_id=run_id,
            description=(
                "Adult Income V2 retrained model. "
                "Passed comparison quality gates. "
                "Registered as Staging candidate only."
            ),
            tags={
                "project": "Adult Income Prediction",
                "model_version": "V2",
                "candidate_stage": "Staging",
                "quality_gate": "PASSED"
            }
        )

        client.set_model_version_tag(
            MODEL_NAME,
            v2_version.version,
            "model_version",
            "V2"
        )

        client.set_model_version_tag(
            MODEL_NAME,
            v2_version.version,
            "quality_gate",
            "PASSED"
        )

        client.set_model_version_tag(
            MODEL_NAME,
            v2_version.version,
            "candidate_stage",
            "Staging"
        )

        client.set_model_version_tag(
            MODEL_NAME,
            v2_version.version,
            "project",
            "Adult Income Prediction"
        )

    # Wait for registration to finish
    client.get_model_version(
        MODEL_NAME,
        v2_version.version
    )

    # --------------------------------------------------
    # MOVE V2 TO STAGING
    # --------------------------------------------------

    current_v2 = client.get_model_version(
        MODEL_NAME,
        v2_version.version
    )

    if current_v2.current_stage != "Staging":

        print("\nMoving V2 to Staging...")

        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=v2_version.version,
            stage="Staging",
            archive_existing_versions=False
        )

    # --------------------------------------------------
    # FINAL VERIFICATION
    # --------------------------------------------------

    final_versions = client.search_model_versions(
        filter_string=f"name='{MODEL_NAME}'"
    )

    final_production = [
        version for version in final_versions
        if version.current_stage == "Production"
    ]

    final_v2 = client.get_model_version(
        MODEL_NAME,
        v2_version.version
    )

    if len(final_production) != 1:
        raise RuntimeError(
            "Final verification failed: Production version count changed."
        )

    if (
        str(final_production[0].version) != PRODUCTION_VERSION
        or final_production[0].run_id != original_production_run_id
    ):
        raise RuntimeError(
            "Production V5 changed unexpectedly!"
        )

    if final_v2.current_stage != "Staging":
        raise RuntimeError(
            "V2 did not reach Staging."
        )

    # --------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL MODEL REGISTRY STATUS")
    print("=" * 65)

    for version in final_versions:

        print(
            "Version:", version.version,
            "| Stage:", version.current_stage,
            "| Run ID:", version.run_id
        )

    print("\nV2 registered successfully.")
    print("V2 version:", final_v2.version)
    print("V2 stage:", final_v2.current_stage)

    print("\nProduction V5 remains unchanged.")
    print("\n[SUCCESS] Safe MLflow registry update completed.")


if __name__ == "__main__":
    main()
