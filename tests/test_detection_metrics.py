"""Detection metrics and bootstrap intervals (numpy/scipy only, no models)."""

import pytest

from evaluation.dataset import EvalImage
from evaluation.detection import (
    average_precision,
    count_error,
    evaluate_detections,
    f1,
    image_stats,
    iou_matrix,
    match_boxes,
    precision,
    recall,
)
from evaluation.stats import bootstrap_ci, paired_bootstrap_diff

A = [0, 0, 10, 10]


# ---------------------------------------------------------------- IoU
@pytest.mark.parametrize("b, expected", [
    ([0, 0, 10, 10], 1.0),
    ([20, 20, 30, 30], 0.0),
    ([5, 0, 15, 10], 50 / 150),
    ([0, 0, 5, 10], 0.5),
])
def test_iou(b, expected):
    assert iou_matrix([A], [b])[0, 0] == pytest.approx(expected)


def test_iou_empty_and_zero_area():
    assert iou_matrix([], [A]).shape == (0, 1)
    assert iou_matrix([[5, 5, 5, 5]], [[5, 5, 5, 5]])[0, 0] == 0.0  # no division by zero


# ---------------------------------------------------------------- matching
def test_optimal_matching_beats_greedy():
    gt = [[0, 0, 10, 10], [4, 0, 14, 10]]
    pred = [[3, 0, 13, 10], [6, 0, 16, 10]]
    iou = iou_matrix(pred, gt)
    # Premise: greedy-by-IoU would pair pred0 with gt1 (0.82) and leave pred1 unmatched.
    assert iou[0, 1] > iou[0, 0] >= 0.5 and iou[1, 1] >= 0.5 > iou[1, 0]
    assert {(p, g) for p, g, _ in match_boxes(pred, gt)} == {(0, 0), (1, 1)}


def test_matching_is_one_to_one():
    matches = match_boxes([A, A], [A])  # duplicate prediction
    assert len(matches) == 1


def test_below_threshold_is_not_a_match():
    assert match_boxes([[0, 0, 10, 10]], [[6, 0, 16, 10]]) == []  # IoU 0.25


def test_matching_independent_of_order():
    gt = [[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]]
    pred = [[1, 0, 11, 10], [41, 0, 51, 10], [21, 0, 31, 10]]
    s1 = image_stats("x", pred, gt)
    s2 = image_stats("x", pred[::-1], gt[::-1])
    assert s1.tp == s2.tp == 3 and s1.iou_sum == pytest.approx(s2.iou_sum)


# ---------------------------------------------------------------- precision / recall / F1
def test_perfect_predictions():
    s = [image_stats("x", [A, [20, 0, 30, 10]], [A, [20, 0, 30, 10]])]
    assert precision(s) == recall(s) == f1(s) == 1.0
    assert count_error(s) == 0.0


def test_duplicates_and_misses_counted():
    s = image_stats("x", [A, A], [A, [50, 50, 60, 60]])  # 1 TP, 1 duplicate FP, 1 missed GT
    assert (s.tp, s.fp, s.fn) == (1, 1, 1)
    assert precision([s]) == recall([s]) == 0.5


def test_no_predictions_and_no_labels():
    assert precision([image_stats("x", [], [A])]) == 0.0
    assert recall([image_stats("x", [], [A])]) == 0.0
    assert f1([image_stats("x", [], [])]) == 0.0


def test_micro_average_weights_by_box_count():
    big = image_stats("a", [[i * 20, 0, i * 20 + 10, 10] for i in range(9)],
                      [[i * 20, 0, i * 20 + 10, 10] for i in range(9)])  # 9/9
    small = image_stats("b", [], [A])  # 0/1
    assert recall([big, small]) == pytest.approx(0.9)


# ---------------------------------------------------------------- average precision
def test_average_precision_known_value():
    gts = [[[0, 0, 10, 10], [20, 0, 30, 10]]]
    preds = [[[0, 0, 10, 10], [50, 50, 60, 60], [20, 0, 30, 10]]]
    scores = [[0.9, 0.8, 0.7]]  # TP, FP, TP
    assert average_precision(preds, scores, gts) == pytest.approx(0.5 * 1 + 0.5 * 2 / 3)


def test_average_precision_bounds():
    gts = [[A]]
    assert average_precision([[A]], [[0.5]], gts) == 1.0
    assert average_precision([[[50, 50, 60, 60]]], [[0.5]], gts) == 0.0
    with pytest.raises(ValueError):
        average_precision([[A]], [[]], gts)


# ---------------------------------------------------------------- bootstrap intervals
def _stats(recalls):
    """Per-image stats with 10 labelled boxes and the given number found."""
    return [image_stats(f"im{i}", [[j * 20, 0, j * 20 + 10, 10] for j in range(k)],
                        [[j * 20, 0, j * 20 + 10, 10] for j in range(10)]) for i, k in enumerate(recalls)]


def test_bootstrap_interval_contains_estimate_and_is_deterministic():
    s = _stats([10, 8, 6, 9, 7, 5, 10, 4])
    est, low, high = bootstrap_ci(s, recall, n_resamples=500)
    assert low <= est <= high and low < high
    assert bootstrap_ci(s, recall, n_resamples=500) == (est, low, high)


def test_bootstrap_single_image_has_no_spread():
    est, low, high = bootstrap_ci(_stats([7]), recall)
    assert est == low == high == 0.7


def test_paired_difference_identical_models_is_zero():
    s = _stats([10, 8, 6, 9])
    assert paired_bootstrap_diff(s, s, recall, n_resamples=200) == (0.0, 0.0, 0.0)


def test_paired_difference_detects_consistent_improvement():
    a, b = _stats([5, 6, 4, 5, 6, 5, 4, 6]), _stats([8, 9, 7, 8, 9, 8, 7, 9])
    diff, low, high = paired_bootstrap_diff(a, b, recall, n_resamples=500)
    assert diff == pytest.approx(0.3) and low > 0


def test_paired_difference_requires_same_images():
    with pytest.raises(ValueError):
        paired_bootstrap_diff(_stats([1, 2]), _stats([1]), recall)


# ---------------------------------------------------------------- evaluate_detections
def _eval_images():
    return [EvalImage("a.jpg", "", 100, 100, [A, [20, 0, 30, 10]], ["a", "a"]),
            EvalImage("b.jpg", "", 100, 100, [[50, 50, 60, 60]], ["a"])]


def test_evaluate_detections_report():
    preds = {"a.jpg": [A], "b.jpg": [[50, 50, 60, 60], [80, 80, 90, 90]]}
    r = evaluate_detections(_eval_images(), preds, n_resamples=200)
    assert (r["n_gt"], r["n_pred"]) == (3, 3)
    assert r["metrics"]["precision"]["value"] == pytest.approx(2 / 3)
    assert r["metrics"]["recall"]["value"] == pytest.approx(2 / 3)
    assert set(r["metrics"]) == {"precision", "recall", "f1", "mean_matched_iou", "count_error"}
    assert [row["tp"] for row in r["per_image"]] == [1, 1]


def test_evaluate_detections_with_scores_adds_ap():
    preds = {"a.jpg": [A, [20, 0, 30, 10]], "b.jpg": [[50, 50, 60, 60]]}
    scores = {"a.jpg": [0.9, 0.8], "b.jpg": [0.7]}
    r = evaluate_detections(_eval_images(), preds, scores=scores, n_resamples=50)
    assert r["metrics"]["ap"]["value"] == 1.0


def test_missing_predictions_rejected():
    with pytest.raises(KeyError, match="b.jpg"):
        evaluate_detections(_eval_images(), {"a.jpg": []})
