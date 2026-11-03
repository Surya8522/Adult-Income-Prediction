
from pathlib import Path
import shutil
import json

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
REGISTRY_DIR = BASE_DIR / "artifacts" / "model_registry"
MODELS_DIR = BASE_DIR / "models"
DEPLOYMENT_DIR = BASE_DIR / "deployment"

# Source files
MODEL_SOURCE = REGISTRY_DIR / "production_model.pkl"
PREPROCESSOR_SOURCE = MODELS_DIR / "preprocessor.pkl"
METADATA_SOURCE = REGISTRY_DIR / "production_model_metadata.json"

# Deployment files
MODEL_DEST = DEPLOYMENT_DIR / "model.pkl"
PREPROCESSOR_DEST = DEPLOYMENT_DIR / "preprocessor.pkl"
METADATA_DEST = DEPLOYMENT_DIR / "model_metadata.json"


def deploy_production():

    print("=" * 55)
    print("ADULT INCOME - PRODUCTION DEPLOYMENT")
    print("=" * 55)

    # Check source files
    required_files = [
        MODEL_SOURCE,
        PREPROCESSOR_SOURCE,
        METADATA_SOURCE
    ]

    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(
                f"Required file missing: {file}"
            )

    DEPLOYMENT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Copy production model
    shutil.copy2(
        MODEL_SOURCE,
        MODEL_DEST
    )

    # Copy preprocessor
    shutil.copy2(
        PREPROCESSOR_SOURCE,
        PREPROCESSOR_DEST
    )

    # Copy metadata
    shutil.copy2(
        METADATA_SOURCE,
        METADATA_DEST
    )

    # Verify deployment files
    deployment_files = [
        MODEL_DEST,
        PREPROCESSOR_DEST,
        METADATA_DEST
    ]

    print("\nDeployment verification:")

    for file in deployment_files:
        if not file.exists():
            raise FileNotFoundError(file)

        print(f"[SUCCESS] {file.name}")

    # Display metadata
    with open(
        METADATA_DEST,
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    print("\nProduction Model Details:")
    print("Model:", metadata.get("model_name"))
    print("Version:", metadata.get("version"))
    print("Stage:", metadata.get("stage"))
    print("Accuracy:", metadata.get("accuracy"))

    print("\nProduction deployment completed!")


if __name__ == "__main__":
    deploy_production()
