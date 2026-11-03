from flask import Flask, request, jsonify

from src.model_loader import load_model, load_preprocessor
from src.preprocessing import preprocess_input
from src.schemas import validate_input
from src.monitoring_logger import log_prediction


# Initialize Flask application
app = Flask(__name__)


# Load model and preprocessor
try:
    model = load_model()
    preprocessor = load_preprocessor()

    print("[SUCCESS] Adult Income model loaded")
    print("[SUCCESS] Preprocessor loaded")

except Exception as e:
    print(f"[ERROR] Model loading failed: {e}")
    raise


# Home endpoint
@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "project": "Adult Income Prediction",
        "status": "Running",
        "message": "Welcome to Adult Income Prediction API"
    }), 200


# Health check endpoint
@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }), 200


# Single prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "A valid JSON object is required"
            }), 400

        # Validate input
        validated_data = validate_input(data)

        # Preprocess input
        processed_data = preprocess_input(
            validated_data,
            preprocessor
        )

        # Make prediction
        prediction = int(model.predict(processed_data)[0])

        # Calculate probability
        if hasattr(model, "predict_proba"):
            probability = float(
                model.predict_proba(processed_data)[0][1]
            )
        else:
            probability = float(prediction)

        # Prepare result
        result = {
            "prediction": prediction,
            "income_class": ">50K" if prediction == 1 else "<=50K",
            "probability": probability
        }

        # Log prediction - three required arguments
        log_prediction(
            data,
            prediction,
            probability
        )

        return jsonify(result), 200

    except Exception as e:
        print(f"[PREDICTION ERROR] {e}")

        return jsonify({
            "error": str(e)
        }), 400


# Batch prediction endpoint
@app.route("/predict/batch", methods=["POST"])
def predict_batch():

    try:
        data = request.get_json(silent=True)

        if not isinstance(data, list) or not data:
            return jsonify({
                "error": "Provide a non-empty JSON list"
            }), 400

        results = []

        for item in data:

            if not isinstance(item, dict):
                return jsonify({
                    "error": "Every batch item must be a JSON object"
                }), 400

            # Validate input
            validated_data = validate_input(item)

            # Preprocess input
            processed_data = preprocess_input(
                validated_data,
                preprocessor
            )

            # Make prediction
            prediction = int(
                model.predict(processed_data)[0]
            )

            # Calculate probability
            if hasattr(model, "predict_proba"):
                probability = float(
                    model.predict_proba(processed_data)[0][1]
                )
            else:
                probability = float(prediction)

            # Prepare result
            result = {
                "prediction": prediction,
                "income_class": ">50K" if prediction == 1 else "<=50K",
                "probability": probability
            }

            # Log prediction - three required arguments
            log_prediction(
                item,
                prediction,
                probability
            )

            results.append(result)

        return jsonify({
            "total_predictions": len(results),
            "results": results
        }), 200

    except Exception as e:
        print(f"[BATCH PREDICTION ERROR] {e}")

        return jsonify({
            "error": str(e)
        }), 400


# Run Flask application
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )