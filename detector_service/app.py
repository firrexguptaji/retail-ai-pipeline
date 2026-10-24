import base64

import cv2
import numpy as np
from flask import Flask, jsonify, request

from common.errors import ServiceError, register_error_handlers
from common.logger import setup_logger
from common.validation import max_upload_bytes, validate_image_bytes

from .detector import detect_products, load_model

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = max_upload_bytes()
logger = setup_logger("detector_service")
register_error_handlers(app, logger)

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
    """Errors are turned into JSON responses by the registered error handlers."""
    logger.info("/detect request received")

    file = request.files.get("image")
    if file is None or file.filename == "":
        raise ServiceError("missing_image", "No image provided", 400)

    data = file.read()
    validate_image_bytes(data)  # this port is reachable directly, so check here too

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ServiceError("invalid_image", "Uploaded file is not a valid image", 400)

    boxes, crops = detect_products(image)

    encoded_crops = [encode_image(crop) for crop in crops]
    encoded_image = encode_image(image)

    return jsonify({
        "boxes": boxes,
        "crops": encoded_crops,
        "image": encoded_image
    })


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