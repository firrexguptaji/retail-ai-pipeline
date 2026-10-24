import os
import time

import requests
from flask import Flask, jsonify, render_template, request, send_from_directory

from common.errors import ServiceError, error_response, register_error_handlers
from common.logger import setup_logger
from common.validation import max_upload_bytes, validate_image_bytes

from .client import DownstreamError, DownstreamTimeout, call_detector, call_grouping
from .config import DETECTOR_READY_URL, GROUPING_READY_URL, OUTPUT_DIR

# -------------------------------
# Setup
# -------------------------------

logger = setup_logger("gateway")
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = max_upload_bytes()
os.makedirs(OUTPUT_DIR, exist_ok=True)
register_error_handlers(app, logger)


@app.errorhandler(DownstreamTimeout)
def _downstream_timeout(e: DownstreamTimeout):
    return error_response("upstream_timeout", f"{e.service} service timed out", 504)


@app.errorhandler(DownstreamError)
def _downstream_error(e: DownstreamError):
    return error_response("upstream_error", f"{e.service} service is unavailable", 502)


# -------------------------------
# Utility: Wait for services (DEV ONLY)
# -------------------------------

def wait_for_service(url, name, timeout=30):
    logger.info(f"[WAITING] {name}...")
    

    for i in range(timeout):
        try:
            res = requests.get(url, timeout=2)
            if res.status_code == 200:
                logger.info(f"[READY] {name}")
                return
        except Exception:
            logger.info(f"Checking URL: {url}")
            logger.warning(f"{name} not ready yet... ({i+1}/{timeout})")

        time.sleep(1)

    raise Exception(f"{name} not available after {timeout}s")


# -------------------------------
# Routes
# -------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return {"service": "gateway", "status": "alive"}, 200


@app.route("/ready", methods=["GET"])
def ready():
    try:
        d = requests.get(DETECTOR_READY_URL, timeout=2)
        g = requests.get(GROUPING_READY_URL, timeout=2)

        if d.status_code != 200 or g.status_code != 200:
            return {"status": "not ready"}, 503

    except Exception:
        return {"status": "not ready", "reason": "dependency failure"}, 503

    return {"service": "gateway", "status": "ready"}, 200


@app.route("/outputs/<filename>")
def get_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/predict", methods=["POST"])
def predict():
    """Errors are turned into JSON responses by the registered error handlers."""
    logger.info("/predict request received")

    file = request.files.get("image")
    if file is None or file.filename == "":
        raise ServiceError("missing_image", "No image provided", 400)

    file_bytes = file.read()
    validate_image_bytes(file_bytes)  # 400 / 413 / 415 before any downstream call

    # Step 1: Detector
    det_data = call_detector(file_bytes)
    logger.info("Detector response received")

    # Step 2: Grouping
    grp_data = call_grouping(det_data)
    logger.info("Grouping response received")

    return jsonify(grp_data)


# -------------------------------
# Dev Mode Entry ONLY
# -------------------------------

if __name__ == "__main__":
    logger.info("Starting Gateway Service (DEV MODE)")
    logger.info(f"GATEWAY OUTPUT_DIR = {OUTPUT_DIR}")

    # Only for local dev (NOT for Docker/gunicorn)
    wait_for_service(DETECTOR_READY_URL, "Detector")
    wait_for_service(GROUPING_READY_URL, "Grouping")

    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)