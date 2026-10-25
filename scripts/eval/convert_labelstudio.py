"""Convert a Label Studio JSON export into the evaluation set.

Usage (repo root):
    python scripts/eval/convert_labelstudio.py data/eval/labelstudio_export.json

Writes data/eval/annotations.json with image checksums and prints a summary.
Fails (and writes nothing) if any image is unannotated, any box lacks a group code,
or an image's size differs from the file on disk.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

from evaluation.dataset import DEFAULT_PATH, EvalImage, save_eval_set, sha256_of  # noqa: E402
from evaluation.labelstudio import task_to_boxes  # noqa: E402


def convert(export_path, out_path=DEFAULT_PATH, allow_partial=False):
    tasks = json.loads(Path(export_path).read_text(encoding="utf-8"))
    images, problems = [], []
    for task in tasks:
        try:
            rel, w, h, boxes, groups = task_to_boxes(task)
        except ValueError as e:
            problems.append(str(e))
            continue
        path = REPO_ROOT / rel
        img = cv2.imread(str(path))
        if img is None:
            problems.append(f"{rel}: image file not found")
            continue
        if (img.shape[1], img.shape[0]) != (w, h):
            problems.append(f"{rel}: labelled at {w}x{h} but file is {img.shape[1]}x{img.shape[0]}")
            continue
        images.append(EvalImage(rel, sha256_of(path), w, h, boxes, groups))

    if problems and not allow_partial:
        raise SystemExit("Not written. Fix these in Label Studio and export again:\n  - "
                         + "\n  - ".join(problems))
    images.sort(key=lambda im: im.file)
    save_eval_set(images, out_path)
    return images, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export", help="Label Studio JSON export")
    ap.add_argument("--out", default=str(DEFAULT_PATH))
    ap.add_argument("--allow-partial", action="store_true",
                    help="write the images that are complete and list the rest")
    args = ap.parse_args()

    images, problems = convert(args.export, args.out, args.allow_partial)
    print(f"{'image':<62} {'boxes':>6} {'groups':>7}")
    for im in images:
        print(f"{im.file:<62} {len(im.boxes):>6} {len(Counter(im.groups)):>7}")
    print(f"\n{len(images)} images, {sum(len(im.boxes) for im in images)} boxes -> {args.out}")
    for p in problems:
        print(f"skipped: {p}")


if __name__ == "__main__":
    main()
