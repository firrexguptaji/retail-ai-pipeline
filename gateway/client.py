import time

import requests

from common.logger import setup_logger

from .config import DETECTOR_URL, GROUPING_URL, REQUEST_TIMEOUT

logger = setup_logger("gateway.client")


def call_detector(file_bytes, filename="image.jpg", mimetype="image/jpeg"):
    logger.info(f"Calling detector at {DETECTOR_URL}")

    try:
        files = {
            "image": (filename, file_bytes, mimetype)
        }

        start = time.time()

        response = requests.post(
            DETECTOR_URL,
            files=files,
            timeout=REQUEST_TIMEOUT
        )

        elapsed = time.time() - start

        logger.info(f"Detector response: {response.status_code} ({elapsed:.2f}s)")

        if response.status_code != 200:
            logger.error(f"Detector failed: {response.status_code} {response.text}")
            raise Exception("Detector service failed")

        return response.json()

    except Exception:
        logger.exception("Error calling detector service")
        raise


def call_grouping(detector_output):
    logger.info(f"Calling grouping at {GROUPING_URL}")

    try:
        start = time.time()

        response = requests.post(
            GROUPING_URL,
            json=detector_output,
            timeout=REQUEST_TIMEOUT
        )

        elapsed = time.time() - start

        logger.info(f"Grouping response: {response.status_code} ({elapsed:.2f}s)")

        if response.status_code != 200:
            logger.error(f"Grouping failed: {response.status_code} {response.text}")
            raise Exception("Grouping service failed")

        return response.json()

    except Exception:
        logger.exception("Error calling grouping service")
        raise