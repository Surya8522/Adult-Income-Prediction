from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
RAW_DATA = BASE_DIR / "data" / "raw" / "adult.csv"
PROCESSED_DATA = BASE_DIR / "data" / "processed" / "adult_cleaned.csv"

# Model path
MODEL_PATH = BASE_DIR / "models" / "final_model.pkl"

# Output paths
MODEL_COMPARISON = BASE_DIR / "outputs" / "model_comparison.csv"
CONFUSION_MATRIX = BASE_DIR / "outputs" / "confusion_matrix.png"
CLASSIFICATION_REPORT = BASE_DIR / "outputs" / "classification_report.txt"
PREDICTIONS = BASE_DIR / "outputs" / "predictions.csv"

# Random state
RANDOM_STATE = 42

# Test size
TEST_SIZE = 0.20