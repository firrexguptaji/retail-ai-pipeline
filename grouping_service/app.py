import base64

import cv2
import numpy as np
from flask import Flask, jsonify, request

from common.logger import setup_logger

from .grouping import group_products
from .utils import draw_boxes, save_output

app = Flask(__name__)
logger = setup_logger("grouping_service")


@app.route("/health", methods=["GET"])
def health():
    return {"service": "grouping", "status": "ok"}, 200


@app.route("/ready")
def ready():
    return {"service": "grouping", "status": "ready"}, 200

def decode_image(base64_str):
    img_bytes = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


@app.route("/group", methods=["POST"])
def group():
    try:
        logger.info("/group request received")
        data = request.json

        boxes = data.get("boxes", [])
        crop_data = data.get("crops", [])
        image_data = data.get("image", None)

        # Decode crops
        crops = [decode_image(c) for c in crop_data]

        # Decode full image
        image = decode_image(image_data) if image_data else None

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

        results = []
        for box, label in zip(boxes, labels):
            results.append({
                "bbox": box,
                "group_id": int(label)
            })

        return jsonify({
            "request_id": request_id,
            "results": results,
            "output_image": f"outputs/{filename}"
        })

    except Exception as e:
        logger.exception("Error in /group endpoint")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logger.info("Starting Grouping Service on port 8002")
    app.run(port=8002, debug=True, use_reloader=False)