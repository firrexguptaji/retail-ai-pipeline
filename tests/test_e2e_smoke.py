"""End-to-end smoke test: real YOLO + real ResNet18, gateway -> detector -> grouping in-process.

Marked `slow` and excluded from the default run. Run with:  pytest -m slow
Needs models/yolov8n.pt and the ResNet18 ImageNet weights (downloaded by torchvision on
first use and cached). Skipped if the dependencies or the YOLO weights are missing.
"""

import importlib
import os
import sys
from io import BytesIO
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

pytest.importorskip("torch")
pytest.importorskip("ultralytics")
pytest.importorskip("sklearn")

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE = REPO_ROOT / "sample_images" / "2019-12-12T152726.jpg"  # 1024 x 768, 12 windows
REAL_MODULES = ["detector_service.detector", "detector_service.app",
                "grouping_service.grouping", "grouping_service.app"]


class ClientResponse:
    """Adapts a Flask test response to the subset of requests.Response the gateway uses."""

    def __init__(self, r):
        self.status_code = r.status_code
        self._json = r.get_json(silent=True)
        self.text = r.get_data(as_text=True)

    def json(self):
        if self._json is None:
            raise ValueError("not json")
        return self._json


@pytest.fixture(scope="module")
def pipeline():
    """Gateway test client whose downstream calls go to real detector and grouping apps."""
    import detector_service.config as det_cfg
    if not det_cfg.MODEL_PATH.is_file():
        pytest.skip(f"YOLO weights not found at {det_cfg.MODEL_PATH}")

    saved = {name: sys.modules.pop(name, None) for name in REAL_MODULES}  # drop conftest stubs
    try:
        det_app = importlib.import_module("detector_service.app").app
        grp_app = importlib.import_module("grouping_service.app").app
        det, grp = det_app.test_client(), grp_app.test_client()

        import gateway.app as gw
        import gateway.client as client

        def route(url, files=None, json=None, **kwargs):
            if url == client.DETECTOR_URL:
                name, data, mimetype = files["image"]
                return ClientResponse(det.post("/detect", data={"image": (BytesIO(data), name)},
                                               content_type="multipart/form-data"))
            if url == client.GROUPING_URL:
                return ClientResponse(grp.post("/group", json=json))
            raise AssertionError(f"unexpected URL {url}")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(client.requests, "post", route)
            yield gw.app.test_client(), gw.OUTPUT_DIR
    finally:
        for name, module in saved.items():
            if module is not None:
                sys.modules[name] = module
            else:
                sys.modules.pop(name, None)


def test_sample_image_end_to_end(pipeline):
    client, output_dir = pipeline
    data = SAMPLE.read_bytes()

    r = client.post("/predict", data={"image": (BytesIO(data), SAMPLE.name)},
                    content_type="multipart/form-data")

    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert set(body) == {"request_id", "results", "output_image"}

    results = body["results"]
    assert len(results) >= 2, "expected several products on a shelf image"
    for item in results:
        x1, y1, x2, y2 = item["bbox"]
        assert all(isinstance(v, int) for v in item["bbox"])
        assert 0 <= x1 < x2 <= 768 and 0 <= y1 < y2 <= 1024
        assert isinstance(item["group_id"], int) and item["group_id"] >= 0

    groups = {item["group_id"] for item in results}
    assert 1 <= len(groups) <= len(results)

    out_file = os.path.join(output_dir, os.path.basename(body["output_image"]))
    assert os.path.isfile(out_file) and os.path.getsize(out_file) > 0
    os.remove(out_file)


def test_corrupt_upload_rejected_with_real_services(pipeline):
    client, _ = pipeline
    r = client.post("/predict", data={"image": (BytesIO(b"not an image"), "x.jpg")},
                    content_type="multipart/form-data")
    assert r.status_code == 400
    assert r.get_json()["code"] == "invalid_image"
