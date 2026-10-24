import time

import cv2
import torch
from torchvision.ops import nms
from ultralytics import YOLO

from common.logger import setup_logger

from .config import (
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    MAX_AREA_RATIO,
    MIN_AREA_RATIO,
    MODEL_PATH,
    OVERLAP,
    SLICE_SIZE,
)

logger = setup_logger("detector")

# -------------------------------
# Model (single source of truth)
# -------------------------------
model = None


def load_model():
    global model

    if model is None:
        # Fail fast: Ultralytics would otherwise silently download weights for a missing path.
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"YOLO weights not found at {MODEL_PATH}. Set MODEL_PATH (absolute, or relative "
                "to the repo root) or place the file in models/. See .env.example."
            )
        logger.info(f"Loading YOLO model from {MODEL_PATH}")
        model = YOLO(str(MODEL_PATH))
        logger.info(f"Model loaded successfully (CONF_THRESHOLD={CONF_THRESHOLD})")


# -------------------------------
# Sliding Window
# -------------------------------
def sliding_window(image):
    h, w, _ = image.shape
    step = int(SLICE_SIZE * (1 - OVERLAP))

    for y in range(0, h, step):
        for x in range(0, w, step):

            y2 = min(y + SLICE_SIZE, h)
            x2 = min(x + SLICE_SIZE, w)

            crop = image[y:y2, x:x2]

            if crop.shape[0] < 50 or crop.shape[1] < 50:
                continue

            yield x, y, crop


# -------------------------------
# Detection
# -------------------------------
def detect_products(image):
    global model

    if model is None:
        raise RuntimeError("Model not loaded")

    start = time.time()

    h_img, w_img, _ = image.shape
    img_area = h_img * w_img

    all_boxes = []
    windows = 0

    # Sliding window inference
    for x_offset, y_offset, window in sliding_window(image):
        windows += 1

        results = model(window, conf=CONF_THRESHOLD, verbose=False)[0]

        if len(results.boxes) == 0:
            continue

        for box in results.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)

            x1 += x_offset
            x2 += x_offset
            y1 += y_offset
            y2 += y_offset

            all_boxes.append([x1, y1, x2, y2])

    # Area filtering
    filtered_boxes = []
    for x1, y1, x2, y2 in all_boxes:
        area = (x2 - x1) * (y2 - y1)

        if MIN_AREA_RATIO * img_area < area < MAX_AREA_RATIO * img_area:
            filtered_boxes.append([x1, y1, x2, y2])

    if not filtered_boxes:
        return [], []

    # NMS (prefer smaller boxes)
    boxes_tensor = torch.tensor(filtered_boxes, dtype=torch.float32)

    areas = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (
        boxes_tensor[:, 3] - boxes_tensor[:, 1]
    )
    scores = 1 / (areas + 1e-6)

    keep = nms(boxes_tensor, scores, iou_threshold=IOU_THRESHOLD)
    filtered_boxes = boxes_tensor[keep].int().tolist()

    # Remove duplicates
    unique_boxes = []
    for box in filtered_boxes:
        x1, y1, x2, y2 = box

        if not any(abs(x1 - bx1) < 10 and abs(y1 - by1) < 10 for bx1, by1, _, _ in unique_boxes):
            unique_boxes.append(box)

    # Aspect ratio filter
    clean_boxes = []
    for x1, y1, x2, y2 in unique_boxes:
        width = x2 - x1
        height = y2 - y1

        if height > 0 and (width / height) < 3:
            clean_boxes.append([x1, y1, x2, y2])

    # Crops
    crops = []
    final_boxes = []

    for (x1, y1, x2, y2) in clean_boxes:
        pad = 5
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w_img, x2 + pad)
        y2 = min(h_img, y2 + pad)

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop = cv2.resize(crop, (224, 224))

        crops.append(crop)
        final_boxes.append([x1, y1, x2, y2])

    elapsed = time.time() - start

    logger.info(
        f"windows={windows}, raw={len(all_boxes)}, final={len(final_boxes)}, time={elapsed:.2f}s"
    )

    return final_boxes, crops