"""Detection metrics: matching predicted boxes to labelled boxes.

Main metrics (no confidence scores needed): precision, recall and F1 at an IoU
threshold, mean IoU of matched pairs, and relative count error. Matching is a
one-to-one assignment that maximises the number of pairs with IoU >= threshold
(ties broken by higher IoU), so the result does not depend on box order.

Average precision (AP) needs a confidence score per predicted box. The current
detector does not return scores, so AP is only computed when scores are given.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .stats import bootstrap_ci


def iou_matrix(a, b) -> np.ndarray:
    """IoU between every box in a (N, 4) and b (M, 4), boxes as [x1, y1, x2, y2]."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_boxes(pred, gt, iou_threshold=0.5):
    """One-to-one matches as a list of (pred_index, gt_index, iou), all with iou >= threshold."""
    iou = iou_matrix(pred, gt)
    if iou.size == 0:
        return []
    valid = iou >= iou_threshold
    # Large constant per valid pair: maximise the number of matches first, then total IoU.
    weight = np.where(valid, 1000.0 + iou, 0.0)
    rows, cols = linear_sum_assignment(weight, maximize=True)
    return [(int(r), int(c), float(iou[r, c])) for r, c in zip(rows, cols) if valid[r, c]]


@dataclass
class ImageDetectionStats:
    file: str
    n_pred: int
    n_gt: int
    tp: int
    iou_sum: float = 0.0
    matches: list = field(default_factory=list)

    @property
    def fp(self):
        return self.n_pred - self.tp

    @property
    def fn(self):
        return self.n_gt - self.tp


def image_stats(file, pred, gt, iou_threshold=0.5) -> ImageDetectionStats:
    matches = match_boxes(pred, gt, iou_threshold)
    return ImageDetectionStats(file, len(pred), len(gt), len(matches),
                               sum(m[2] for m in matches), matches)


# ---------------------------------------------------------------- aggregate metrics (micro-averaged)
def precision(stats):
    tp, n_pred = sum(s.tp for s in stats), sum(s.n_pred for s in stats)
    return tp / n_pred if n_pred else 0.0


def recall(stats):
    tp, n_gt = sum(s.tp for s in stats), sum(s.n_gt for s in stats)
    return tp / n_gt if n_gt else 0.0


def f1(stats):
    p, r = precision(stats), recall(stats)
    return 2 * p * r / (p + r) if p + r else 0.0


def mean_matched_iou(stats):
    tp = sum(s.tp for s in stats)
    return sum(s.iou_sum for s in stats) / tp if tp else 0.0


def count_error(stats):
    """Mean over images of |predicted count - labelled count| / labelled count."""
    errors = [abs(s.n_pred - s.n_gt) / s.n_gt for s in stats if s.n_gt]
    return float(np.mean(errors)) if errors else 0.0


METRICS = {
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "mean_matched_iou": mean_matched_iou,
    "count_error": count_error,
}


def average_precision(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """All-point interpolated AP over a set of images.

    pred_boxes / pred_scores / gt_boxes: lists with one entry per image.
    Predictions are matched greedily in descending score order (standard for AP).
    """
    records, n_gt = [], 0
    for boxes, scores, gts in zip(pred_boxes, pred_scores, gt_boxes):
        if len(boxes) != len(scores):
            raise ValueError("one score per predicted box is required")
        n_gt += len(gts)
        iou = iou_matrix(boxes, gts)
        taken = np.zeros(len(gts), bool)
        for i in np.argsort(-np.asarray(scores, dtype=float), kind="stable"):
            hit = False
            if len(gts):
                candidates = np.where(~taken, iou[i], -1.0)
                j = int(np.argmax(candidates))
                if candidates[j] >= iou_threshold:
                    taken[j], hit = True, True
            records.append((float(scores[i]), hit))
    if n_gt == 0:
        return 0.0
    records.sort(key=lambda r: -r[0])
    hits = np.array([h for _, h in records], dtype=float)
    tp_cum, fp_cum = np.cumsum(hits), np.cumsum(1 - hits)
    rec = tp_cum / n_gt
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]  # precision envelope
    steps = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[steps + 1] - mrec[steps]) * mpre[steps + 1]))


def evaluate_detections(eval_images, predictions, iou_threshold=0.5, scores=None, **ci_kwargs):
    """Score predicted boxes against an evaluation set.

    eval_images: list of evaluation.dataset.EvalImage
    predictions: {file: [[x1, y1, x2, y2], ...]} with an entry for every image
    scores: optional {file: [score, ...]} enabling average precision
    Returns a dict with each metric as {"value", "ci_low", "ci_high"}, plus per-image rows.
    """
    missing = [im.file for im in eval_images if im.file not in predictions]
    if missing:
        raise KeyError(f"no predictions for: {missing}")

    stats = [image_stats(im.file, predictions[im.file], im.boxes, iou_threshold) for im in eval_images]
    result = {"iou_threshold": iou_threshold, "n_images": len(stats),
              "n_gt": sum(s.n_gt for s in stats), "n_pred": sum(s.n_pred for s in stats),
              "metrics": {}, "per_image": []}
    for name, fn in METRICS.items():
        value, low, high = bootstrap_ci(stats, fn, **ci_kwargs)
        result["metrics"][name] = {"value": value, "ci_low": low, "ci_high": high}
    if scores is not None:
        result["metrics"]["ap"] = {"value": average_precision(
            [predictions[im.file] for im in eval_images], [scores[im.file] for im in eval_images],
            [im.boxes for im in eval_images], iou_threshold)}
    for s in stats:
        result["per_image"].append({"file": s.file, "n_gt": s.n_gt, "n_pred": s.n_pred, "tp": s.tp,
                                    "fp": s.fp, "fn": s.fn, "precision": precision([s]),
                                    "recall": recall([s])})
    return result
