"""Write Label Studio tasks pre-filled with the current detector's boxes.

Usage (repo root, service venv):
    python scripts/eval/prelabel.py                      # all images in sample_images/
    python scripts/eval/prelabel.py --images sample_images/a.jpg sample_images/b.jpg

Output: data/eval/prelabels.json (not committed). Import it into Label Studio.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

from evaluation.labelstudio import boxes_to_task  # noqa: E402

PAD = 5  # detector_service adds 5 px around each box; remove it for tighter pre-labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", help="image paths relative to the repo root")
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "eval" / "prelabels.json"))
    args = ap.parse_args()

    from detector_service.detector import detect_products, load_model

    rels = args.images or sorted(
        p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / "sample_images").glob("*.jpg")
    )
    load_model()
    tasks = []
    for rel in rels:
        image = cv2.imread(str(REPO_ROOT / rel), cv2.IMREAD_COLOR)
        if image is None:
            print(f"skip (cannot read): {rel}")
            continue
        h, w = image.shape[:2]
        boxes, _ = detect_products(image)
        boxes = [[x1 + PAD, y1 + PAD, x2 - PAD, y2 - PAD] for x1, y1, x2, y2 in boxes]
        tasks.append(boxes_to_task(rel, w, h, boxes, model_version="yolov8n-coco-baseline"))
        print(f"{rel}: {w}x{h}, {len(boxes)} pre-filled boxes")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tasks, indent=1), encoding="utf-8")
    print(f"\nWrote {len(tasks)} tasks to {out}")


if __name__ == "__main__":
    main()
