import mlflow


MODEL_NAME = "Adult_Income_Production_Model"


def automate_lifecycle():

    print("=========================================")
    print("Adult Income Model Lifecycle")
    print("=========================================")

    client = mlflow.MlflowClient()

    try:
        versions = client.search_model_versions(
            f"name='{MODEL_NAME}'"
        )

        if not versions:
            print(
                "No registered model version found."
            )
            return

        latest = max(
            versions,
            key=lambda v: int(v.version)
        )

        print(
            f"Latest model version: "
            f"{latest.version}"
        )

        print(
            "Model successfully registered."
        )

    except Exception as e:

        print(
            f"Lifecycle error: {e}"
        )


if __name__ == "__main__":
    automate_lifecycle()