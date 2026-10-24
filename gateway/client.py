import time

import requests

from common.errors import ServiceError
from common.logger import setup_logger

from .config import DETECTOR_URL, GROUPING_URL, REQUEST_TIMEOUT

logger = setup_logger("gateway.client")


class DownstreamError(Exception):
    """A downstream service failed or returned an unusable response (-> 502)."""

    def __init__(self, service: str):
        super().__init__(f"{service} service failed")
        self.service = service


class DownstreamTimeout(DownstreamError):
    """A downstream service did not answer within REQUEST_TIMEOUT (-> 504)."""


def _post(service: str, url: str, passthrough_client_errors: bool = False, **kwargs) -> dict:
    """POST to a downstream service and return its JSON body.

    passthrough_client_errors: re-raise the service's 4xx as a ServiceError so the
    caller's own input problem (e.g. a corrupt image) reaches the user unchanged.
    """
    start = time.time()
    try:
        response = requests.post(url, timeout=REQUEST_TIMEOUT, **kwargs)
    except requests.Timeout:
        logger.error("%s timed out after %.1fs", service, time.time() - start)
        raise DownstreamTimeout(service) from None
    except requests.RequestException as exc:
        logger.error("%s unreachable: %s", service, type(exc).__name__)
        raise DownstreamError(service) from None

    logger.info("%s response: %s (%.2fs)", service, response.status_code, time.time() - start)

    if passthrough_client_errors and 400 <= response.status_code < 500:
        try:
            body = response.json()
            raise ServiceError(body["code"], body["error"], response.status_code)
        except (ValueError, KeyError, TypeError):
            pass  # not our error format: treat as a downstream failure below

    if response.status_code != 200:
        logger.error("%s failed with status %s", service, response.status_code)
        raise DownstreamError(service)

    try:
        return response.json()
    except ValueError:
        logger.error("%s returned a non-JSON body", service)
        raise DownstreamError(service) from None


def call_detector(file_bytes, filename="upload", mimetype="application/octet-stream"):
    logger.info(f"Calling detector at {DETECTOR_URL}")
    files = {"image": (filename, file_bytes, mimetype)}
    return _post("Detector", DETECTOR_URL, passthrough_client_errors=True, files=files)


def call_grouping(detector_output):
    logger.info(f"Calling grouping at {GROUPING_URL}")
    # A 4xx from grouping means the gateway sent bad data: that is our failure, not the user's.
    return _post("Grouping", GROUPING_URL, json=detector_output)