from __future__ import annotations

import logging
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from price_model import ensure_price_model
from project import predict_car

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PORT = 5050

logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")
price_model, training_metrics = ensure_price_model()


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "price_model_trained_now": training_metrics is not None,
        }
    )


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return ("", 204)

    if "image" not in request.files:
        return jsonify({"error": "Nie przesłano pliku obrazu."}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "Wybrany plik jest pusty."}), 400

    image_bytes = image_file.read()

    try:
        prediction = predict_car(image_bytes)
        estimated_price = price_model.predict_from_label(prediction.label)
    except Exception:
        logger.exception("Prediction failed")
        return jsonify({"error": "Błąd podczas predykcji. Spróbuj innego zdjęcia."}), 500

    return jsonify(
        {
            "label": prediction.label,
            "brand": prediction.brand,
            "model": prediction.model,
            "year": prediction.year,
            "confidence": prediction.confidence,
            "price_reference_table": prediction.price_from_table,
            "price_model_estimate": estimated_price,
        }
    )


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    app.run(debug=debug_mode, port=int(os.environ.get("PORT", DEFAULT_PORT)))
