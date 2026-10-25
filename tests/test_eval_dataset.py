"""Evaluation set: Label Studio conversion, format validation and integrity checks."""

import importlib.util
import json

import pytest

from evaluation.dataset import EvalImage, load_eval_set, save_eval_set, sha256_of, validate
from evaluation.labelstudio import boxes_to_task, image_url, rel_path_from_url, task_to_boxes

SAMPLE = "sample_images/2019-12-12T152726.jpg"  # 768 wide x 1024 high
W, H = 768, 1024


def annotated(task, groups, rotate=None, cancelled=False, updated="2026-10-25T10:00:00"):
    """Turn a pre-filled task into an exported, annotated one (as Label Studio would)."""
    result = []
    for r, code in zip(task["predictions"][0]["result"], groups):
        r = json.loads(json.dumps(r))
        if rotate is not None:
            r["value"]["rotation"] = rotate
        result.append(r)
        if code is not None:
            result.append({"id": r["id"], "from_name": "group", "to_name": "image",
                           "type": "textarea", "value": {"text": [code]}})
    task = dict(task)
    task["annotations"] = [{"result": result, "was_cancelled": cancelled, "updated_at": updated}]
    return task


BOXES = [[10, 20, 110, 220], [120, 20, 220, 220], [300, 500, 400, 700]]


# ---------------------------------------------------------------- Label Studio conversion
def test_round_trip_keeps_pixel_boxes_and_groups():
    task = annotated(boxes_to_task(SAMPLE, W, H, BOXES), ["a", "a", "b"])
    rel, w, h, boxes, groups = task_to_boxes(task)
    assert (rel, w, h) == (SAMPLE, W, H)
    assert boxes == BOXES
    assert groups == ["a", "a", "b"]


def test_prelabel_boxes_are_clipped_and_degenerate_dropped():
    task = boxes_to_task(SAMPLE, W, H, [[-5, -5, 50, 50], [700, 1000, 900, 1100], [5, 5, 5, 9]])
    rects = task["predictions"][0]["result"]
    assert len(rects) == 2
    assert rects[0]["value"]["x"] == 0 and rects[0]["value"]["y"] == 0


@pytest.mark.parametrize("url", [image_url(SAMPLE), "/data/local-files/?d=sample_images%2F2019-12-12T152726.jpg",
                                 "/data/local-files/?d=sample_images\\2019-12-12T152726.jpg"])
def test_local_files_url_parsing(url):
    assert rel_path_from_url(url) == SAMPLE


def test_box_without_group_code_rejected():
    task = annotated(boxes_to_task(SAMPLE, W, H, BOXES), ["a", None, "b"])
    with pytest.raises(ValueError, match="no group code"):
        task_to_boxes(task)


def test_unannotated_and_cancelled_tasks_rejected():
    with pytest.raises(ValueError, match="no submitted annotation"):
        task_to_boxes(boxes_to_task(SAMPLE, W, H, BOXES))
    with pytest.raises(ValueError, match="no submitted annotation"):
        task_to_boxes(annotated(boxes_to_task(SAMPLE, W, H, BOXES), ["a"] * 3, cancelled=True))


def test_rotated_box_rejected():
    with pytest.raises(ValueError, match="rotated"):
        task_to_boxes(annotated(boxes_to_task(SAMPLE, W, H, BOXES), ["a"] * 3, rotate=12))


def test_latest_annotation_wins():
    base = boxes_to_task(SAMPLE, W, H, BOXES)
    old = annotated(base, ["x", "x", "x"], updated="2026-10-24T10:00:00")["annotations"][0]
    new = annotated(base, ["a", "a", "b"], updated="2026-10-25T10:00:00")["annotations"][0]
    task = dict(base, annotations=[new, old])
    assert task_to_boxes(task)[4] == ["a", "a", "b"]


# ---------------------------------------------------------------- format validation
def _data(boxes):
    return {"version": 1, "images": [{"file": SAMPLE, "sha256": "x", "width": W, "height": H, "boxes": boxes}]}


@pytest.mark.parametrize("boxes, msg", [
    ([{"bbox": [0, 0, 800, 10], "group": "a"}], "outside image"),
    ([{"bbox": [5, 5, 5, 10], "group": "a"}], "outside image"),
    ([{"bbox": [0.5, 0, 10, 10], "group": "a"}], "4 integers"),
    ([{"bbox": [0, 0, 10, 10], "group": " "}], "group code"),
])
def test_invalid_boxes_rejected(boxes, msg):
    with pytest.raises(ValueError, match=msg):
        validate(_data(boxes))


def test_duplicate_image_rejected():
    data = _data([])
    data["images"].append(dict(data["images"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        validate(data)


# ---------------------------------------------------------------- loading and integrity
def test_save_and_load_round_trip(tmp_path):
    from evaluation.dataset import REPO_ROOT
    img = EvalImage(SAMPLE, sha256_of(REPO_ROOT / SAMPLE), W, H, BOXES, ["a", "a", "b"])
    save_eval_set([img], tmp_path / "ann.json")
    [loaded] = load_eval_set(tmp_path / "ann.json")
    assert (loaded.boxes, loaded.groups, loaded.width, loaded.height) == (BOXES, ["a", "a", "b"], W, H)


def test_changed_image_detected_by_checksum(tmp_path):
    save_eval_set([EvalImage(SAMPLE, "0" * 64, W, H, BOXES, ["a"] * 3)], tmp_path / "ann.json")
    with pytest.raises(ValueError, match="checksum"):
        load_eval_set(tmp_path / "ann.json")
    assert load_eval_set(tmp_path / "ann.json", verify_files=False)  # structure alone is fine


def test_missing_image_detected(tmp_path):
    save_eval_set([EvalImage("sample_images/nope.jpg", "0" * 64, W, H, [], [])], tmp_path / "ann.json")
    with pytest.raises(FileNotFoundError):
        load_eval_set(tmp_path / "ann.json")


# ---------------------------------------------------------------- converter script
@pytest.fixture
def converter():
    from evaluation.dataset import REPO_ROOT
    spec = importlib.util.spec_from_file_location("convert_ls", REPO_ROOT / "scripts/eval/convert_labelstudio.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_converter_writes_eval_set(converter, tmp_path):
    export = tmp_path / "export.json"
    export.write_text(json.dumps([annotated(boxes_to_task(SAMPLE, W, H, BOXES), ["a", "a", "b"])]))
    images, problems = converter.convert(export, tmp_path / "ann.json")
    assert problems == [] and len(images) == 1
    assert load_eval_set(tmp_path / "ann.json")[0].groups == ["a", "a", "b"]


def test_converter_refuses_incomplete_export(converter, tmp_path):
    export = tmp_path / "export.json"
    good = annotated(boxes_to_task(SAMPLE, W, H, BOXES), ["a", "a", "b"])
    wrong_size = annotated(boxes_to_task(SAMPLE, 1000, 1000, BOXES), ["a", "a", "b"])
    unlabelled = boxes_to_task("sample_images/128008.jpg", 1080, 1920, BOXES)
    export.write_text(json.dumps([good, wrong_size, unlabelled]))
    with pytest.raises(SystemExit, match="Not written"):
        converter.convert(export, tmp_path / "ann.json")
    assert not (tmp_path / "ann.json").exists()
    images, problems = converter.convert(export, tmp_path / "ann.json", allow_partial=True)
    assert len(images) == 1 and len(problems) == 2
