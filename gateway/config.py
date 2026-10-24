import os

# Base URLs (no endpoints)
DETECTOR_BASE_URL = os.getenv(
    "DETECTOR_BASE_URL",
    "http://127.0.0.1:8001"
)

GROUPING_BASE_URL = os.getenv(
    "GROUPING_BASE_URL",
    "http://127.0.0.1:8002"
)

# Endpoints
DETECTOR_URL = f"{DETECTOR_BASE_URL}/detect"
GROUPING_URL = f"{GROUPING_BASE_URL}/group"

DETECTOR_READY_URL = f"{DETECTOR_BASE_URL}/ready"
GROUPING_READY_URL = f"{GROUPING_BASE_URL}/ready"

# Seconds to wait for detector/grouping before returning 504
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "30"))


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.getenv(
    "OUTPUT_DIR",
    os.path.join(BASE_DIR, "outputs")
)