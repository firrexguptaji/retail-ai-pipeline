"""Evaluation-set format, loading and integrity checks.

`data/eval/annotations.json`::

    {
      "version": 1,
      "images": [
        {
          "file": "sample_images/x.jpg",          # relative to the repo root, forward slashes
          "sha256": "...", "width": 1024, "height": 768,
          "boxes": [{"bbox": [x1, y1, x2, y2], "group": "a"}, ...]
        }
      ]
    }

Boxes are integer pixel coordinates, x2 > x1 and y2 > y1. Group codes are only
meaningful within one image.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = REPO_ROOT / "data" / "eval" / "annotations.json"
FORMAT_VERSION = 1


@dataclass
class EvalImage:
    file: str
    sha256: str
    width: int
    height: int
    boxes: list = field(default_factory=list)   # [[x1, y1, x2, y2], ...]
    groups: list = field(default_factory=list)  # one code per box

    @property
    def path(self) -> Path:
        return REPO_ROOT / self.file


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_box(bbox, width, height, where):
    if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, int) for v in bbox)):
        raise ValueError(f"{where}: bbox must be 4 integers, got {bbox!r}")
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"{where}: bbox {bbox} outside image {width}x{height} or empty")


def validate(data: dict) -> None:
    """Raise ValueError if the structure is not a valid evaluation set."""
    if data.get("version") != FORMAT_VERSION:
        raise ValueError(f"unsupported version {data.get('version')!r}")
    seen = set()
    for i, img in enumerate(data.get("images", [])):
        where = f"images[{i}] ({img.get('file')})"
        if img.get("file") in seen:
            raise ValueError(f"{where}: duplicate image")
        seen.add(img.get("file"))
        for key in ("file", "sha256", "width", "height", "boxes"):
            if key not in img:
                raise ValueError(f"{where}: missing '{key}'")
        for j, box in enumerate(img["boxes"]):
            _check_box(box.get("bbox"), img["width"], img["height"], f"{where} boxes[{j}]")
            group = box.get("group")
            if not isinstance(group, str) or not group.strip():
                raise ValueError(f"{where} boxes[{j}]: missing group code")


def load_eval_set(path=DEFAULT_PATH, verify_files: bool = True) -> list[EvalImage]:
    """Load and validate the evaluation set; by default also verify each image's checksum."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate(data)
    images = []
    for img in data["images"]:
        item = EvalImage(
            file=img["file"], sha256=img["sha256"], width=img["width"], height=img["height"],
            boxes=[b["bbox"] for b in img["boxes"]], groups=[b["group"].strip() for b in img["boxes"]],
        )
        if verify_files:
            if not item.path.is_file():
                raise FileNotFoundError(f"evaluation image missing: {item.file}")
            if sha256_of(item.path) != item.sha256:
                raise ValueError(f"evaluation image changed since labelling (checksum mismatch): {item.file}")
        images.append(item)
    return images


def save_eval_set(images: list[EvalImage], path=DEFAULT_PATH) -> None:
    data = {
        "version": FORMAT_VERSION,
        "images": [
            {"file": im.file, "sha256": im.sha256, "width": im.width, "height": im.height,
             "boxes": [{"bbox": b, "group": g} for b, g in zip(im.boxes, im.groups)]}
            for im in images
        ],
    }
    validate(data)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
