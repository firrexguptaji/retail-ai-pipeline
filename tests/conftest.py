"""Test setup: stub the model modules so tests run fast without weights.

The real detector loads YOLO and the real grouping module downloads ResNet18 at
import time. Tests replace both modules before the Flask apps import them, and
override `detect_products` / `group_products` per test with monkeypatch.
"""

import base64
import os
import sys
import tempfile
import types
from io import BytesIO

import pytest
from PIL import Image

os.environ.setdefault("OUTPUT_DIR", tempfile.mkdtemp(prefix="retail-test-outputs-"))


def _stub(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module


_stub("detector_service.detector", load_model=lambda: None, detect_products=lambda image: ([], []))
_stub("grouping_service.grouping", load_model=lambda: None,
      group_products=lambda crops, boxes, shape: [0] * len(crops))


def image_bytes(fmt="PNG", size=(64, 48), color=(200, 30, 30)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


def image_b64(fmt="JPEG", size=(64, 48)) -> str:
    return base64.b64encode(image_bytes(fmt, size)).decode()


@pytest.fixture
def png():
    return image_bytes("PNG")


@pytest.fixture
def gateway_client():
    from gateway.app import app
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def detector_client():
    from detector_service.app import app
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def grouping_client():
    from grouping_service.app import app
    app.config["TESTING"] = True
    return app.test_client()
