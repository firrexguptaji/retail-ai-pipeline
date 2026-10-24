import base64

import cv2
import numpy as np
from flask import Flask, jsonify, request

from common.logger import setup_logger

from .detector import detect_products, load_model

app = Flask(__name__)
logger = setup_logger("detector_service")

# -------------------------------
# Startup (load model once)
# -------------------------------
load_model()


# -------------------------------
# Health & Ready
# -------------------------------
@app.route("/health", methods=["GET"])
def health():
    return {"service": "detector", "status": "ok"}, 200


@app.route("/ready", methods=["GET"])
def ready():
    return {"service": "detector", "status": "ready"}, 200


# -------------------------------
# Utils
# -------------------------------
def encode_image(image):
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")


# -------------------------------
# Detection Endpoint
# -------------------------------
@app.route("/detect", methods=["POST"])
def detect():
    try:
        logger.info("/detect request received")

        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]

        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        boxes, crops = detect_products(image)

        encoded_crops = [encode_image(crop) for crop in crops]
        encoded_image = encode_image(image)

        return jsonify({
            "boxes": boxes,
            "crops": encoded_crops,
            "image": encoded_image
        })

    except Exception as e:
        logger.exception("Error in /detect")
        return jsonify({"error": str(e)}), 500


# -------------------------------
# Entry
# -------------------------------
if __name__ == "__main__":
    logger.info("Starting Detector Service on port 8001")

    app.run(
        host="0.0.0.0",
        port=8001,
        debug=True,
        use_reloader=False
    )