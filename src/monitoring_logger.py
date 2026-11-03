
import json
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent.parent

LOG_PATH = (
    BASE_DIR / "logs" / "inference" / "inference_log.jsonl"
)


def log_prediction(input_data, prediction, probability):

    LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": input_data,
        "prediction": int(prediction),
        "predicted_income": (
            ">50K" if int(prediction) == 1 else "<=50K"
        ),
        "probability_above_50k": float(probability)
    }

    with open(LOG_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")
