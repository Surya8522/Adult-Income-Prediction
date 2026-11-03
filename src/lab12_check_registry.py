
import mlflow
from mlflow.tracking import MlflowClient

print("=" * 60)
print("LAB 12 - MLFLOW MODEL REGISTRY INSPECTION")
print("=" * 60)

client = MlflowClient()

print("\nTracking URI:")
print(mlflow.get_tracking_uri())

print("\nRegistered Models:")

registered_models = client.search_registered_models()

if not registered_models:
    print("No registered models found.")

for model in registered_models:
    print("\nModel Name:", model.name)

print("\nProduction Model Versions:")

model_name = "Adult_Income_Production_Model"

versions = client.search_model_versions(
    filter_string=f"name='{model_name}'"
)

if not versions:
    print("No versions found for", model_name)

for version in versions:
    print("--------------------------------")
    print("Version:", version.version)
    print("Stage:", version.current_stage)
    print("Run ID:", version.run_id)
    print("Status:", version.status)

print("\n[SUCCESS] Registry inspection completed.")
