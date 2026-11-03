
from pathlib import Path
import joblib
import json


class ModelLoader:

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[1]
        self.deployment_dir = self.project_root / "deployment"

        self.model = None
        self.preprocessor = None
        self.metadata = {}

    def load_model(self):

        model_path = self.deployment_dir / "model.pkl"
        preprocessor_path = (
            self.deployment_dir / "preprocessor.pkl"
        )
        metadata_path = (
            self.deployment_dir / "model_metadata.json"
        )

        for path in [
            model_path,
            preprocessor_path,
            metadata_path
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"File not found: {path}"
                )

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

        with open(metadata_path, "r", encoding="utf-8") as file:
            self.metadata = json.load(file)

        print("[SUCCESS] Adult Income model loaded")
        print("[SUCCESS] Preprocessor loaded")
        print("[SUCCESS] Model metadata loaded")

    def get_model(self):
        return self.model

    def get_preprocessor(self):
        return self.preprocessor

    def get_metadata(self):
        return self.metadata
