import os
import json
import mlflow


MODEL_NAME = "Adult_Income_Production_Model"


def generate_report():

    print("=========================================")
    print("Generating Model Registry Report")
    print("=========================================")

    client = mlflow.MlflowClient()

    report = {
        "model_name": MODEL_NAME,
        "dataset": "Adult Income",
        "status": "registered"
    }

    try:

        versions = client.search_model_versions(
            f"name='{MODEL_NAME}'"
        )

        if versions:

            latest = max(
                versions,
                key=lambda v: int(v.version)
            )

            report["version"] = latest.version
            report["run_id"] = latest.run_id
            report["source"] = latest.source

    except Exception as e:

        report["error"] = str(e)

    os.makedirs(
        "reports",
        exist_ok=True
    )

    with open(
        "reports/production_model_report.json",
        "w"
    ) as f:
        json.dump(
            report,
            f,
            indent=4
        )

    print(
        "\nReport saved to:"
    )

    print(
        "reports/production_model_report.json"
    )


if __name__ == "__main__":
    generate_report()