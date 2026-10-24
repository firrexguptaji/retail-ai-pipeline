import os
import uuid

import cv2

from common.logger import setup_logger

from .config import OUTPUT_DIR

logger = setup_logger("grouping.utils")


def draw_boxes(image, boxes, labels):
    colors = {}

    for i, label in enumerate(labels):
        if label not in colors:
            colors[label] = (
                int(50 + label * 40) % 255,
                int(80 + label * 70) % 255,
                int(120 + label * 90) % 255
            )

        x1, y1, x2, y2 = boxes[i]
        color = colors[label]

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"ID:{label}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image


def save_output(image):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    request_id = str(uuid.uuid4())
    filename = f"result_{request_id}.jpg"
    path = os.path.join(OUTPUT_DIR, filename)

    success = cv2.imwrite(path, image)

    if not success:
        raise Exception("Failed to write output image")

    logger.info(f"Wrote output image to {path}")

    return filename, request_id