import os
import json
import mlflow
from mlflow.tracking import MlflowClient


def generate_registry_report():

    print("=========================================")
    print("Generating Model Registry Report")
    print("=========================================")

    client = MlflowClient()

    # Adult Income registered model
    model_name = "Adult_Income_Production_Model"

    try:

        # =====================================================
        # 1. Search for the Model
        # =====================================================

        production_models = client.search_model_versions(
            f"name='{model_name}'"
        )

        champion = next(
            (
                mv
                for mv in production_models
                if mv.current_stage == "Production"
            ),
            None
        )

        if not champion:

            print(
                "[ERROR] No model found in Production stage!"
            )

            return

        # =====================================================
        # 2. Get Run Details
        # =====================================================

        run = client.get_run(champion.run_id)

        # =====================================================
        # 3. Build Deployment-Ready Report
        # =====================================================

        report = {

            "registry_status":
                "READY_FOR_DEPLOYMENT",

            "model_lineage": {

                "registered_name":
                    model_name,

                "version":
                    int(champion.version),

                "current_stage":
                    champion.current_stage,

                "run_id":
                    champion.run_id,

                "artifact_uri":
                    champion.source
            },

            "performance_metrics": {

                "recall":
                    run.data.metrics.get("recall"),

                "f1_score":
                    run.data.metrics.get("f1_score"),

                "roc_auc":
                    run.data.metrics.get("roc_auc"),

                "accuracy":
                    run.data.metrics.get("accuracy"),

                "precision":
                    run.data.metrics.get("precision")
            },

            "hyperparameters":
                run.data.params,

            "preprocessing_dependency":
                "models/preprocessor.pkl"
        }

        # =====================================================
        # 4. Save Report
        # =====================================================

        os.makedirs(
            "reports",
            exist_ok=True
        )

        report_path = (
            "reports/production_model_report.json"
        )

        with open(
            report_path,
            "w"
        ) as f:

            json.dump(
                report,
                f,
                indent=4
            )

        # =====================================================
        # 5. Success Message
        # =====================================================

        print(
            f"[SUCCESS] Report successfully generated "
            f"for Version {champion.version}"
        )

        print(
            f"[INFO] Saved to: {report_path}"
        )

        print("=========================================")
        print("Registry Report Generation Completed")
        print("=========================================")

    except Exception as e:

        print(
            f"[ERROR] Failed to generate report: {e}"
        )


if __name__ == "__main__":
    generate_registry_report()