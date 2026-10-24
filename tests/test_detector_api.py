"""Detector /detect: input validation and sanitised errors (model is stubbed)."""

from io import BytesIO

import numpy as np

import detector_service.app as det_app


def post_image(c, data, filename="shelf.png"):
    return c.post("/detect", data={"image": (BytesIO(data), filename)},
                  content_type="multipart/form-data")


def test_valid_image_returns_boxes_crops_image(detector_client, png, monkeypatch):
    crop = np.zeros((8, 8, 3), np.uint8)
    monkeypatch.setattr(det_app, "detect_products", lambda image: ([[1, 2, 9, 10]], [crop]))
    r = post_image(detector_client, png)
    body = r.get_json()
    assert r.status_code == 200
    assert body["boxes"] == [[1, 2, 9, 10]]
    assert len(body["crops"]) == 1 and body["image"]


def test_missing_image(detector_client):
    r = detector_client.post("/detect", data={}, content_type="multipart/form-data")
    assert r.status_code == 400
    assert r.get_json()["code"] == "missing_image"


def test_corrupt_image_is_400_not_500(detector_client):
    r = post_image(detector_client, b"\x89PNG\r\n\x1a\n garbage")
    assert r.status_code == 400
    assert r.get_json()["code"] == "invalid_image"


def test_model_failure_is_500_without_exception_text(detector_client, png, monkeypatch):
    def boom(image):
        raise RuntimeError("CUDA out of memory at /secret/path")
    monkeypatch.setattr(det_app, "detect_products", boom)
    r = post_image(detector_client, png)
    assert r.status_code == 500
    assert r.get_json()["code"] == "internal_error"
    assert "secret" not in r.get_data(as_text=True)
