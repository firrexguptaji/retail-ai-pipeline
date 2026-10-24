"""Gateway /predict: input validation and downstream failure mapping."""

from io import BytesIO

import pytest
import requests

import gateway.client as client


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=None):
        self.status_code = status_code
        self._body = body
        self.text = text if text is not None else str(body)

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def post_image(c, data, filename="shelf.png"):
    return c.post("/predict", data={"image": (BytesIO(data), filename)},
                  content_type="multipart/form-data")


@pytest.fixture
def fake_downstream(monkeypatch):
    """Route requests.post to per-URL fake responses or exceptions."""
    routes = {}

    def fake_post(url, **kwargs):
        result = routes[url]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(client.requests, "post", fake_post)
    return routes


GROUPED = {"request_id": "abc", "results": [{"bbox": [1, 2, 3, 4], "group_id": 0}],
           "output_image": "outputs/result_abc.jpg"}
DETECTED = {"boxes": [[1, 2, 3, 4]], "crops": ["x"], "image": "y"}


def test_happy_path_returns_grouping_response(gateway_client, png, fake_downstream):
    fake_downstream[client.DETECTOR_URL] = FakeResponse(200, DETECTED)
    fake_downstream[client.GROUPING_URL] = FakeResponse(200, GROUPED)
    r = post_image(gateway_client, png)
    assert r.status_code == 200
    assert r.get_json() == GROUPED


def test_missing_image_keeps_original_message(gateway_client):
    r = gateway_client.post("/predict", data={}, content_type="multipart/form-data")
    assert r.status_code == 400
    assert r.get_json() == {"error": "No image provided", "code": "missing_image"}


def test_empty_filename_is_missing_image(gateway_client):
    r = post_image(gateway_client, b"", filename="")
    assert r.status_code == 400
    assert r.get_json()["code"] == "missing_image"


def test_corrupt_upload_rejected_before_detector(gateway_client, fake_downstream):
    r = post_image(gateway_client, b"not an image at all")
    assert r.status_code == 400
    assert r.get_json()["code"] == "invalid_image"
    assert fake_downstream == {}  # detector never called


def test_unsupported_type_is_415(gateway_client):
    from tests.conftest import image_bytes
    r = post_image(gateway_client, image_bytes("GIF"), "anim.gif")
    assert r.status_code == 415
    assert r.get_json()["code"] == "unsupported_image_type"


def test_oversized_upload_is_413(gateway_client, monkeypatch, png):
    monkeypatch.setitem(gateway_client.application.config, "MAX_CONTENT_LENGTH", 100)
    r = post_image(gateway_client, png + b"\0" * 500)
    assert r.status_code == 413
    assert r.get_json()["code"] == "payload_too_large"


def test_detector_timeout_is_504(gateway_client, png, fake_downstream):
    fake_downstream[client.DETECTOR_URL] = requests.Timeout("read timed out")
    r = post_image(gateway_client, png)
    assert r.status_code == 504
    assert r.get_json() == {"error": "Detector service timed out", "code": "upstream_timeout"}


def test_detector_unreachable_is_502(gateway_client, png, fake_downstream):
    fake_downstream[client.DETECTOR_URL] = requests.ConnectionError("refused")
    r = post_image(gateway_client, png)
    assert r.status_code == 502
    assert r.get_json()["code"] == "upstream_error"


def test_detector_client_error_passes_through(gateway_client, png, fake_downstream):
    fake_downstream[client.DETECTOR_URL] = FakeResponse(
        400, {"error": "Uploaded file is not a valid image", "code": "invalid_image"})
    r = post_image(gateway_client, png)
    assert r.status_code == 400
    assert r.get_json()["code"] == "invalid_image"


def test_detector_non_json_error_is_502(gateway_client, png, fake_downstream):
    fake_downstream[client.DETECTOR_URL] = FakeResponse(400, None, text="<html>bad</html>")
    r = post_image(gateway_client, png)
    assert r.status_code == 502


def test_grouping_failure_is_502_and_hides_details(gateway_client, png, fake_downstream):
    fake_downstream[client.DETECTOR_URL] = FakeResponse(200, DETECTED)
    fake_downstream[client.GROUPING_URL] = FakeResponse(
        500, {"error": "Traceback: secret internals", "code": "internal_error"})
    r = post_image(gateway_client, png)
    assert r.status_code == 502
    assert "secret" not in r.get_data(as_text=True)


def test_grouping_4xx_is_gateway_fault_not_passed_through(gateway_client, png, fake_downstream):
    fake_downstream[client.DETECTOR_URL] = FakeResponse(200, DETECTED)
    fake_downstream[client.GROUPING_URL] = FakeResponse(
        400, {"error": "'image' is required", "code": "invalid_payload"})
    r = post_image(gateway_client, png)
    assert r.status_code == 502


def test_unexpected_error_is_500_with_error_id_only(gateway_client, png, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("database password is hunter2")
    monkeypatch.setattr("gateway.app.call_detector", boom)
    r = post_image(gateway_client, png)
    body = r.get_json()
    assert r.status_code == 500
    assert body["code"] == "internal_error" and len(body["error_id"]) == 12
    assert "hunter2" not in r.get_data(as_text=True)


def test_request_timeout_is_configurable(monkeypatch):
    import importlib

    import gateway.config as cfg
    monkeypatch.setenv("REQUEST_TIMEOUT", "7.5")
    assert importlib.reload(cfg).REQUEST_TIMEOUT == 7.5
    monkeypatch.delenv("REQUEST_TIMEOUT")
    importlib.reload(cfg)
