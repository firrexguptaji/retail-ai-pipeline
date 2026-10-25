"""Uncertainty for evaluation metrics: bootstrap confidence intervals over images.

Metrics are computed from per-image statistics, so resampling *images* with
replacement gives an interval that reflects how much the result depends on which
images happen to be in the evaluation set.
"""

import numpy as np

DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 0


def bootstrap_ci(per_image, metric_fn, n_resamples=DEFAULT_RESAMPLES, confidence=0.95, seed=DEFAULT_SEED):
    """Return (estimate, low, high) for metric_fn(list_of_per_image_stats)."""
    per_image = list(per_image)
    estimate = metric_fn(per_image)
    if len(per_image) < 2:
        return estimate, estimate, estimate
    rng = np.random.default_rng(seed)
    n = len(per_image)
    samples = [metric_fn([per_image[i] for i in rng.integers(0, n, n)]) for _ in range(n_resamples)]
    alpha = (1 - confidence) / 2
    low, high = np.quantile(samples, [alpha, 1 - alpha])
    return estimate, float(low), float(high)


def paired_bootstrap_diff(per_image_a, per_image_b, metric_fn, n_resamples=DEFAULT_RESAMPLES,
                          confidence=0.95, seed=DEFAULT_SEED):
    """Difference metric(B) - metric(A) on the same images, with a paired bootstrap interval.

    Returns (diff, low, high). If the interval contains 0, report "no measurable difference".
    """
    a, b = list(per_image_a), list(per_image_b)
    if len(a) != len(b):
        raise ValueError("paired comparison needs the same images in the same order")
    diff = metric_fn(b) - metric_fn(a)
    if len(a) < 2:
        return diff, diff, diff
    rng = np.random.default_rng(seed)
    n = len(a)
    samples = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        samples.append(metric_fn([b[i] for i in idx]) - metric_fn([a[i] for i in idx]))
    alpha = (1 - confidence) / 2
    low, high = np.quantile(samples, [alpha, 1 - alpha])
    return diff, float(low), float(high)
