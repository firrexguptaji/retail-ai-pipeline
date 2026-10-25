"""Conversion between Label Studio JSON and the evaluation-set format.

Label Studio stores rectangles as percentages of the image size. The labelling
interface (evaluation/labelstudio_config.xml) uses `box` (RectangleLabels) and a
per-region `group` (TextArea) that shares the rectangle's region id.
"""

from urllib.parse import parse_qs, urlparse

FROM_BOX, FROM_GROUP, TO_IMAGE, LABEL = "box", "group", "image", "product"


def image_url(rel_path: str) -> str:
    """Label Studio local-files URL for a path relative to the document root (repo root)."""
    return "/data/local-files/?d=" + rel_path.replace("\\", "/")


def rel_path_from_url(url: str) -> str:
    query = parse_qs(urlparse(url).query)
    if "d" not in query:
        raise ValueError(f"not a local-files URL: {url!r}")
    return query["d"][0].replace("\\", "/")


def boxes_to_task(rel_path, width, height, boxes, model_version="detector"):
    """One Label Studio task for an image, pre-filled with pixel boxes as predictions."""
    result = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        result.append({
            "id": f"p{i}", "from_name": FROM_BOX, "to_name": TO_IMAGE, "type": "rectanglelabels",
            "original_width": width, "original_height": height, "image_rotation": 0,
            "value": {"x": 100 * x1 / width, "y": 100 * y1 / height,
                      "width": 100 * (x2 - x1) / width, "height": 100 * (y2 - y1) / height,
                      "rotation": 0, "rectanglelabels": [LABEL]},
        })
    return {"data": {"image": image_url(rel_path)},
            "predictions": [{"model_version": model_version, "result": result}]}


def _latest_annotation(task):
    annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    if not annotations:
        return None
    return max(annotations, key=lambda a: a.get("updated_at") or a.get("created_at") or "")


def task_to_boxes(task):
    """Return (rel_path, width, height, boxes, groups) from an exported, annotated task.

    Raises ValueError for unannotated tasks, rotated boxes, or boxes without a group code.
    """
    rel_path = rel_path_from_url(task["data"]["image"])
    ann = _latest_annotation(task)
    if ann is None:
        raise ValueError(f"{rel_path}: no submitted annotation")

    rects, groups = {}, {}
    width = height = None
    for r in ann.get("result", []):
        if r.get("from_name") == FROM_BOX and r.get("type") == "rectanglelabels":
            if abs(r["value"].get("rotation", 0)) > 1e-6:
                raise ValueError(f"{rel_path}: rotated box {r['id']} is not supported")
            rects[r["id"]] = r
            width, height = r["original_width"], r["original_height"]
        elif r.get("from_name") == FROM_GROUP and r.get("type") == "textarea":
            texts = r.get("value", {}).get("text") or []
            groups[r["id"]] = (texts[0] if texts else "").strip()

    boxes, codes = [], []
    for rid, r in rects.items():
        v = r["value"]
        x1 = round(v["x"] / 100 * width)
        y1 = round(v["y"] / 100 * height)
        x2 = round((v["x"] + v["width"]) / 100 * width)
        y2 = round((v["y"] + v["height"]) / 100 * height)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue  # degenerate after rounding
        code = groups.get(rid, "")
        if not code:
            raise ValueError(f"{rel_path}: box {rid} at {[x1, y1, x2, y2]} has no group code")
        boxes.append([x1, y1, x2, y2])
        codes.append(code)
    return rel_path, width, height, boxes, codes
