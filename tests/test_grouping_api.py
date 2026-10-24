"""Grouping /group: payload validation and sanitised errors (embedding model is stubbed)."""

import pytest

import grouping_service.app as grp_app
from tests.conftest import image_b64

IMG = image_b64(size=(100, 80))
CROP = image_b64(size=(16, 16))


def post(c, payload):
    return c.post("/group", json=payload)


def test_two_boxes_grouped(grouping_client):
    payload = {"boxes": [[0, 0, 10, 10], [20, 20, 40, 40]], "crops": [CROP, CROP], "image": IMG}
    r = post(grouping_client, payload)
    body = r.get_json()
    assert r.status_code == 200
    assert [x["bbox"] for x in body["results"]] == payload["boxes"]
    assert body["output_image"].startswith("outputs/result_")


def test_zero_boxes_returns_empty_result(grouping_client):
    r = post(grouping_client, {"boxes": [], "crops": [], "image": IMG})
    assert r.status_code == 200
    assert r.get_json() == {"request_id": None, "results": [], "output_image": None}


def test_non_json_body(grouping_client):
    r = grouping_client.post("/group", data="nope", content_type="text/plain")
    assert r.status_code == 400
    assert r.get_json()["code"] == "invalid_payload"


@pytest.mark.parametrize(
    "payload",
    [
        [],                                                         # not an object
        {"boxes": [], "crops": []},                                 # image missing (used to crash)
        {"boxes": "x", "crops": [], "image": IMG},                  # boxes not a list
        {"boxes": [[0, 0, 5, 5]], "crops": [], "image": IMG},       # length mismatch
        {"boxes": [[0, 0, 5]], "crops": [CROP], "image": IMG},      # 3 coordinates
        {"boxes": [[5, 5, 0, 0]], "crops": [CROP], "image": IMG},   # x2 < x1
        {"boxes": [[0.5, 0, 5, 5]], "crops": [CROP], "image": IMG}, # floats
        {"boxes": [[0, 0, 5, 5]], "crops": [1], "image": IMG},      # crop not a string
    ],
)
def test_invalid_payloads_are_400(grouping_client, payload):
    r = post(grouping_client, payload)
    assert r.status_code == 400
    assert r.get_json()["code"] == "invalid_payload"


@pytest.mark.parametrize("bad", ["%%%not-base64%%%", "aGVsbG8="])  # invalid b64 / valid b64 of "hello"
def test_undecodable_image_is_400(grouping_client, bad):
    r = post(grouping_client, {"boxes": [[0, 0, 5, 5]], "crops": [CROP], "image": bad})
    assert r.status_code == 400
    assert r.get_json()["code"] == "invalid_image"


def test_undecodable_crop_names_the_field(grouping_client):
    r = post(grouping_client, {"boxes": [[0, 0, 5, 5]], "crops": ["aGVsbG8="], "image": IMG})
    assert r.status_code == 400
    assert "crops[0]" in r.get_json()["error"]


def test_clustering_failure_is_500_without_exception_text(grouping_client, monkeypatch):
    def boom(crops, boxes, shape):
        raise ValueError("Found array with 1 sample(s) -- internal detail")
    monkeypatch.setattr(grp_app, "group_products", boom)
    r = post(grouping_client, {"boxes": [[0, 0, 5, 5]], "crops": [CROP], "image": IMG})
    assert r.status_code == 500
    assert "internal detail" not in r.get_data(as_text=True)
