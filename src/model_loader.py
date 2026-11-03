
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR / "artifacts" /
    "model_registry" / "production_model.pkl"
)

PREPROCESSOR_PATH = (
    BASE_DIR / "models" / "preprocessor.pkl"
)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


def load_preprocessor():
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found: {PREPROCESSOR_PATH}"
        )

    return joblib.load(PREPROCESSOR_PATH)
