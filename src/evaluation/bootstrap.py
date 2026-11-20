"""Paired Bootstrap Confidence Intervals (Sections 8.6, 12.2).

Computes 95% confidence intervals for single system means and paired differences
between two systems (`system_a - system_b`) with at least 1,000 resamples.
"""

from __future__ import annotations

import numpy as np


def bootstrap_mean_ci(
    values: list[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Compute bootstrap mean and 95% percentile confidence interval for a list of metric values."""
    if not values:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_resamples, n))
    sample_means = arr[indices].mean(axis=1)

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(sample_means, 100.0 * alpha))
    upper = float(np.percentile(sample_means, 100.0 * (1.0 - alpha)))
    return {
        "mean": round(float(arr.mean()), 4),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
    }


def paired_bootstrap(
    scores_a: list[float],
    scores_b: list[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Compute paired bootstrap difference `A - B` and 95% confidence interval (Section 8.6 & 12.2).

    Requires `len(scores_a) == len(scores_b)` (ordered identically by `question_id`).
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"Paired bootstrap requires equal lengths, got {len(scores_a)} and {len(scores_b)}"
        )
    if not scores_a:
        return {
            "mean_a": 0.0,
            "mean_b": 0.0,
            "diff_mean": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "p_value_two_sided": 1.0,
        }

    arr_a = np.asarray(scores_a, dtype=np.float64)
    arr_b = np.asarray(scores_b, dtype=np.float64)
    diffs = arr_a - arr_b
    n = len(diffs)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_resamples, n))
    boot_diffs = diffs[indices].mean(axis=1)

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_diffs, 100.0 * alpha))
    upper = float(np.percentile(boot_diffs, 100.0 * (1.0 - alpha)))

    prop_le_zero = float(np.mean(boot_diffs <= 0.0))
    prop_ge_zero = float(np.mean(boot_diffs >= 0.0))
    p_val = min(1.0, 2.0 * min(prop_le_zero, prop_ge_zero))

    return {
        "mean_a": round(float(arr_a.mean()), 4),
        "mean_b": round(float(arr_b.mean()), 4),
        "diff_mean": round(float(diffs.mean()), 4),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
        "p_value_two_sided": round(p_val, 4),
    }
