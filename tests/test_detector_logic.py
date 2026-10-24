"""Real detector module (not the stub): fail-fast weights check and model call arguments.

Needs ultralytics and torch (requirements-dev.txt); skipped otherwise. No weights are
loaded: YOLO is replaced by a fake.
"""

import importlib
import sys

import numpy as np
import pytest

pytest.importorskip("ultralytics")
pytest.importorskip("torch")


@pytest.fixture
def detector(monkeypatch):
    """Import the real detector module, bypassing the conftest stub."""
    monkeypatch.delitem(sys.modules, "detector_service.detector", raising=False)
    mod = importlib.import_module("detector_service.detector")
    monkeypatch.setattr(mod, "model", None)
    return mod


class FakeResult:
    boxes = []  # no detections


class FakeYOLO:
    def __init__(self, path=None):
        self.path = path
        self.calls = []

    def __call__(self, image, **kwargs):
        self.calls.append(kwargs)
        return [FakeResult()]


def test_missing_weights_fail_fast_without_download(detector, monkeypatch, tmp_path):
    created = []
    monkeypatch.setattr(detector, "MODEL_PATH", tmp_path / "missing.pt")
    monkeypatch.setattr(detector, "YOLO", lambda p: created.append(p))
    with pytest.raises(FileNotFoundError, match="missing.pt"):
        detector.load_model()
    assert created == []  # YOLO never called, so nothing is downloaded


def test_existing_weights_loaded_once(detector, monkeypatch, tmp_path):
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"x")
    created = []
    monkeypatch.setattr(detector, "MODEL_PATH", weights)
    monkeypatch.setattr(detector, "YOLO", lambda p: created.append(p) or FakeYOLO(p))
    detector.load_model()
    detector.load_model()
    assert created == [str(weights)]


def test_model_called_with_conf_threshold_and_quiet(detector, monkeypatch):
    fake = FakeYOLO()
    monkeypatch.setattr(detector, "model", fake)
    monkeypatch.setattr(detector, "CONF_THRESHOLD", 0.2)
    boxes, crops = detector.detect_products(np.zeros((600, 600, 3), np.uint8))
    assert (boxes, crops) == ([], [])
    assert fake.calls and all(c == {"conf": 0.2, "verbose": False} for c in fake.calls)
