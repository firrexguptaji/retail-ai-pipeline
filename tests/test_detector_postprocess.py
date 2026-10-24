"""Detector logic: sliding window and post-processing, with a fake YOLO (no weights).

These are characterisation tests: they pin down current behaviour, including known
quirks (NMS prefers smaller boxes; near-duplicates are judged by the top-left corner
only; returned boxes include 5 px padding), so M4 experiments change it deliberately.
"""

import importlib
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")


class FakeBoxes:
    def __init__(self, xyxy):
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4)

    def __len__(self):
        return self.xyxy.shape[0]


class FakeResult:
    def __init__(self, xyxy):
        self.boxes = FakeBoxes(xyxy)


class FakeYOLO:
    """Returns the boxes listed for the n-th call (window), in window coordinates."""

    def __init__(self, per_call=None):
        self.per_call = per_call or {}
        self.calls = 0

    def __call__(self, window, **kwargs):
        boxes = self.per_call.get(self.calls, [])
        self.calls += 1
        return [FakeResult(boxes)]


@pytest.fixture
def det(monkeypatch):
    """Real detector module (bypassing the conftest stub) with fixed, explicit settings."""
    monkeypatch.delitem(sys.modules, "detector_service.detector", raising=False)
    mod = importlib.import_module("detector_service.detector")
    for name, value in {"SLICE_SIZE": 512, "OVERLAP": 0.4, "IOU_THRESHOLD": 0.4,
                        "MIN_AREA_RATIO": 0.0005, "MAX_AREA_RATIO": 0.03}.items():
        monkeypatch.setattr(mod, name, value)
    return mod


def run(det, monkeypatch, per_call, size=(600, 600)):
    monkeypatch.setattr(det, "model", FakeYOLO(per_call))
    return det.detect_products(np.zeros((size[1], size[0], 3), np.uint8))


# 600 x 600 image, 512 slice, 0.4 overlap -> step 307, windows in order:
# call 0 at (x=0, y=0), call 1 at (307, 0), call 2 at (0, 307), call 3 at (307, 307).
# Area limits: 0.0005 * 360000 = 180 < area < 0.03 * 360000 = 10800.


# ---------------------------------------------------------------- sliding window
def windows(det, h, w):
    return [(x, y, win.shape[:2]) for x, y, win in det.sliding_window(np.zeros((h, w, 3), np.uint8))]


def test_image_smaller_than_slice_is_one_window(det):
    assert windows(det, 200, 300) == [(0, 0, (200, 300))]


def test_strips_thinner_than_50px_are_skipped(det):
    assert windows(det, 40, 400) == []


def test_step_follows_overlap(det, monkeypatch):
    monkeypatch.setattr(det, "SLICE_SIZE", 100)
    monkeypatch.setattr(det, "OVERLAP", 0.5)
    offsets = sorted({x for x, _, _ in windows(det, 200, 200)})
    assert offsets == [0, 50, 100, 150]
    assert len(windows(det, 200, 200)) == 16


def test_windows_cover_every_pixel(det):
    h, w = 700, 1000
    covered = np.zeros((h, w), bool)
    for x, y, (wh, ww) in windows(det, h, w):
        covered[y:y + wh, x:x + ww] = True
    assert covered.all()


# ---------------------------------------------------------------- detect_products
def test_model_not_loaded_raises(det, monkeypatch):
    monkeypatch.setattr(det, "model", None)
    with pytest.raises(RuntimeError):
        det.detect_products(np.zeros((100, 100, 3), np.uint8))


def test_no_detections(det, monkeypatch):
    assert run(det, monkeypatch, {}) == ([], [])


def test_window_offsets_applied_and_box_padded(det, monkeypatch):
    boxes, _ = run(det, monkeypatch, {1: [[10, 20, 60, 90]]})  # second window, x offset 307
    assert boxes == [[312, 15, 372, 95]]  # [317, 20, 367, 90] plus 5 px padding


def test_area_filter_drops_tiny_and_huge_boxes(det, monkeypatch):
    assert run(det, monkeypatch, {0: [[10, 10, 15, 15], [0, 0, 200, 200]]}) == ([], [])


def test_same_box_from_overlapping_windows_kept_once(det, monkeypatch):
    # Global box [320, 20, 370, 90] seen by window (0, 0) and window (307, 0)
    boxes, _ = run(det, monkeypatch, {0: [[320, 20, 370, 90]], 1: [[13, 20, 63, 90]]})
    assert boxes == [[315, 15, 375, 95]]


def test_nms_keeps_the_smaller_of_overlapping_boxes(det, monkeypatch):
    boxes, _ = run(det, monkeypatch, {0: [[0, 0, 100, 100], [0, 0, 90, 90]]})  # IoU 0.81
    assert boxes == [[0, 0, 95, 95]]


def test_near_duplicate_by_top_left_corner_removed(det, monkeypatch):
    # Low IoU (0.22) so NMS keeps both, but top-left corners are within 10 px:
    # the smaller box (ranked first by NMS) is kept.
    boxes, _ = run(det, monkeypatch, {0: [[100, 100, 140, 200], [105, 105, 200, 145]]})
    assert boxes == [[100, 100, 205, 150]]


def test_wide_boxes_removed_by_aspect_ratio(det, monkeypatch):
    boxes, _ = run(det, monkeypatch, {0: [[0, 0, 150, 40], [200, 200, 319, 240]]})  # 3.75 vs 2.975
    assert boxes == [[195, 195, 324, 245]]


def test_padding_clamped_to_image_border(det, monkeypatch):
    boxes, _ = run(det, monkeypatch, {0: [[2, 3, 52, 73]]})
    assert boxes == [[0, 0, 57, 78]]


def test_crops_match_boxes_and_are_224(det, monkeypatch):
    boxes, crops = run(det, monkeypatch, {0: [[10, 10, 60, 80], [300, 300, 350, 380]]})
    assert len(boxes) == len(crops) == 2
    assert all(c.shape == (224, 224, 3) and c.dtype == np.uint8 for c in crops)
