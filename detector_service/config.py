import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "yolov8n.pt"


def resolve_model_path(value: str | None) -> Path:
    """Absolute paths are used as-is; relative paths resolve from the repo root (not the CWD).

    Backslashes are accepted so Windows-style values also work on Linux and in Docker.
    """
    if not value:
        return DEFAULT_MODEL_PATH
    path = Path(value.replace("\\", "/")).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def parse_conf_threshold(value: str) -> float:
    conf = float(value)
    if not 0 < conf <= 1:
        raise ValueError(f"CONF_THRESHOLD must be in (0, 1], got {value}")
    return conf


MODEL_PATH = resolve_model_path(os.getenv("MODEL_PATH"))

# YOLO confidence threshold per sliding window. Kept low on purpose (recall first);
# tuned against the evaluation baseline in M4, not by hand.
CONF_THRESHOLD = parse_conf_threshold(os.getenv("CONF_THRESHOLD", "0.03"))

SLICE_SIZE = int(os.getenv("SLICE_SIZE", 512))
OVERLAP = float(os.getenv("OVERLAP", 0.4))

IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", 0.4))

MIN_AREA_RATIO = float(os.getenv("MIN_AREA_RATIO", 0.0005))
MAX_AREA_RATIO = float(os.getenv("MAX_AREA_RATIO", 0.03))