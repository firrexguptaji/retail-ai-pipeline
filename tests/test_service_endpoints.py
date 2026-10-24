"""Health, readiness, UI and output-file endpoints of all three services."""

import os

import pytest
import requests

import gateway.app as gw


class FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


@pytest.fixture
def fake_ready(monkeypatch):
    """Route gateway readiness probes to per-URL fake responses or exceptions."""
    routes, calls = {}, []

    def fake_get(url, timeout=None, **kwargs):
        calls.append((url, timeout))
        result = routes.get(url, FakeResponse(200))
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(gw.requests, "get", fake_get)
    return routes, calls


# ---------------------------------------------------------------- gateway
def test_ui_page_renders(gateway_client):
    r = gateway_client.get("/")
    assert r.status_code == 200
    assert "Retail AI" in r.get_data(as_text=True)


def test_gateway_health(gateway_client):
    r = gateway_client.get("/health")
    assert r.status_code == 200
    assert r.get_json() == {"service": "gateway", "status": "alive"}


def test_gateway_ready_when_dependencies_ready(gateway_client, fake_ready):
    _, calls = fake_ready
    r = gateway_client.get("/ready")
    assert r.status_code == 200
    assert r.get_json() == {"service": "gateway", "status": "ready"}
    assert [u for u, _ in calls] == [gw.DETECTOR_READY_URL, gw.GROUPING_READY_URL]
    assert all(t is not None for _, t in calls)  # probes never wait forever


@pytest.mark.parametrize("which", ["detector", "grouping"])
def test_gateway_not_ready_when_dependency_not_ready(gateway_client, fake_ready, which):
    routes, _ = fake_ready
    url = gw.DETECTOR_READY_URL if which == "detector" else gw.GROUPING_READY_URL
    routes[url] = FakeResponse(503)
    r = gateway_client.get("/ready")
    assert r.status_code == 503
    assert r.get_json()["status"] == "not ready"


@pytest.mark.parametrize("exc", [requests.ConnectionError("refused"), requests.Timeout("slow")])
def test_gateway_not_ready_when_dependency_unreachable(gateway_client, fake_ready, exc):
    routes, _ = fake_ready
    routes[gw.GROUPING_READY_URL] = exc
    r = gateway_client.get("/ready")
    assert r.status_code == 503
    assert r.get_json() == {"status": "not ready", "reason": "dependency failure"}


def test_output_image_served(gateway_client):
    name = "result_test-serve.jpg"
    path = os.path.join(gw.OUTPUT_DIR, name)
    with open(path, "wb") as f:
        f.write(b"\xff\xd8fake-jpeg")
    try:
        r = gateway_client.get(f"/outputs/{name}")
        try:
            assert r.status_code == 200
            assert r.data == b"\xff\xd8fake-jpeg"
        finally:
            r.close()  # releases the streamed file handle; Windows cannot delete open files
    finally:
        os.remove(path)


def test_missing_output_is_404_json(gateway_client):
    r = gateway_client.get("/outputs/result_does-not-exist.jpg")
    assert r.status_code == 404
    assert r.get_json()["code"] == "not_found"


@pytest.mark.parametrize("url", ["/outputs/../secret.txt", "/outputs/..%2Fsecret.txt", "/outputs/%2E%2E%2Fsecret.txt"])
def test_output_path_traversal_blocked(gateway_client, url):
    secret = os.path.join(os.path.dirname(os.path.abspath(gw.OUTPUT_DIR)), "secret.txt")
    with open(secret, "w") as f:
        f.write("TOP-SECRET")
    try:
        r = gateway_client.get(url)
        assert r.status_code == 404
        assert "TOP-SECRET" not in r.get_data(as_text=True)
    finally:
        os.remove(secret)


# ---------------------------------------------------------------- detector and grouping
@pytest.mark.parametrize(
    "client_fixture, service",
    [("detector_client", "detector"), ("grouping_client", "grouping")],
)
@pytest.mark.parametrize("endpoint, status", [("/health", "ok"), ("/ready", "ready")])
def test_model_service_health_and_ready(request, client_fixture, service, endpoint, status):
    client = request.getfixturevalue(client_fixture)
    r = client.get(endpoint)
    assert r.status_code == 200
    assert r.get_json() == {"service": service, "status": status}


def test_unknown_route_is_json_404(gateway_client):
    r = gateway_client.get("/nope")
    assert r.status_code == 404
    assert r.get_json() == {"error": "Not Found", "code": "not_found"}


def test_wrong_method_is_json_405(gateway_client):
    r = gateway_client.get("/predict")
    assert r.status_code == 405
    assert r.get_json()["code"] == "method_not_allowed"
