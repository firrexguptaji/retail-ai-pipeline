import base64
import binascii

import cv2
import numpy as np
from flask import Flask, jsonify, request

from common.errors import ServiceError, register_error_handlers
from common.logger import setup_logger
from common.validation import max_payload_bytes

from .grouping import group_products, load_model
from .utils import draw_boxes, save_output

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = max_payload_bytes()
logger = setup_logger("grouping_service")
register_error_handlers(app, logger)

# Load the embedding model once at startup, not on the first request
load_model()


@app.route("/health", methods=["GET"])
def health():
    return {"service": "grouping", "status": "ok"}, 200


@app.route("/ready")
def ready():
    return {"service": "grouping", "status": "ready"}, 200

def decode_image(base64_str, field="image"):
    """Decode a base64 JPEG/PNG string; raise ServiceError(400) if it is not a valid image."""
    try:
        img_bytes = base64.b64decode(base64_str, validate=True)
        image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    except (binascii.Error, ValueError, TypeError, cv2.error):
        image = None
    if image is None:
        raise ServiceError("invalid_image", f"'{field}' is not a valid base64-encoded image", 400)
    return image


def _is_box(box):
    return (
        isinstance(box, list)
        and len(box) == 4
        and all(isinstance(v, int) and not isinstance(v, bool) for v in box)
        and box[2] > box[0]
        and box[3] > box[1]
    )


def parse_group_request(data):
    """Validate the detector payload; return (boxes, crops, image) or raise ServiceError(400)."""
    if not isinstance(data, dict):
        raise ServiceError("invalid_payload", "Request body must be a JSON object", 400)

    boxes, crop_data, image_data = data.get("boxes"), data.get("crops"), data.get("image")

    if not isinstance(boxes, list) or not isinstance(crop_data, list):
        raise ServiceError("invalid_payload", "'boxes' and 'crops' must be lists", 400)
    if len(boxes) != len(crop_data):
        raise ServiceError("invalid_payload", "'boxes' and 'crops' must have the same length", 400)
    if not isinstance(image_data, str) or not image_data:
        raise ServiceError("invalid_payload", "'image' is required", 400)
    for i, box in enumerate(boxes):
        if not _is_box(box):
            raise ServiceError("invalid_payload", f"boxes[{i}] must be [x1, y1, x2, y2] integers with x2>x1, y2>y1", 400)
    if not all(isinstance(c, str) for c in crop_data):
        raise ServiceError("invalid_payload", "'crops' must be base64 strings", 400)

    image = decode_image(image_data, "image")
    crops = [decode_image(c, f"crops[{i}]") for i, c in enumerate(crop_data)]
    return boxes, crops, image


@app.route("/group", methods=["POST"])
def group():
    """Errors are turned into JSON responses by the registered error handlers."""
    logger.info("/group request received")
    boxes, crops, image = parse_group_request(request.get_json(silent=True))

    # Grouping
    labels = group_products(crops, boxes, image.shape)
    logger.info(f"Grouping completed: {len(labels)} labels")

    # Handle empty case
    if len(labels) == 0:
        return jsonify({
            "request_id": None,
            "results": [],
            "output_image": None
        })

    # Visualization
    vis = draw_boxes(image.copy(), boxes, labels)

    filename, request_id = save_output(vis)
    logger.info(f"Saved output image: {filename} (request_id={request_id})")

    results = [{"bbox": box, "group_id": int(label)} for box, label in zip(boxes, labels)]

    return jsonify({
        "request_id": request_id,
        "results": results,
        "output_image": f"outputs/{filename}"
    })


if __name__ == "__main__":
    logger.info("Starting Grouping Service on port 8002")
    app.run(port=8002, debug=True, use_reloader=False)