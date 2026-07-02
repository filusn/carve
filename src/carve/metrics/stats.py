"""Bootstrap statistics for CARVE aggregation (docs/METRICS.md §Aggregation & statistics).

Image-level metrics → bootstrap 95% CIs (≥1000 resamples). Method comparisons ("SAE > CAV")
→ paired bootstrap; never claim a win whose CI overlaps 0. NaNs (e.g. skipped-|e_in| images
from causal_recovery) are dropped before resampling.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def _clean(x) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    return x[~np.isnan(x)]


def bootstrap_ci(
    values,
    n: int = 1000,
    ci: float = 0.95,
    rng: np.random.Generator | int | None = None,
    statistic: Callable[[np.ndarray], float] = np.median,
) -> tuple[float, float]:
    """Percentile bootstrap CI for `statistic` (default median) of `values`.

    Returns (lo, hi). Empty input ⇒ (nan, nan). `rng` may be a Generator, an int seed,
    or None.
    """
    vals = _clean(values)
    if vals.size == 0:
        return (float("nan"), float("nan"))
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    idx = rng.integers(0, vals.size, size=(n, vals.size))
    stats = statistic(vals[idx], axis=1)
    lo = np.quantile(stats, (1 - ci) / 2)
    hi = np.quantile(stats, 1 - (1 - ci) / 2)
    return (float(lo), float(hi))


def paired_bootstrap_diff(
    a,
    b,
    n: int = 1000,
    ci: float = 0.95,
    rng: np.random.Generator | int | None = None,
) -> tuple[float, float, float]:
    """Paired bootstrap of the mean difference (a − b) over matched items.

    Returns (mean_diff, lo, hi). Resamples item indices jointly so pairing is preserved.
    A win is only defensible when the CI excludes 0. Requires len(a) == len(b); pairs with
    a NaN in either arm are dropped.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"paired inputs must have equal length; got {a.shape} vs {b.shape}")
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    if a.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    diff = a - b
    idx = rng.integers(0, diff.size, size=(n, diff.size))
    means = diff[idx].mean(axis=1)
    lo = np.quantile(means, (1 - ci) / 2)
    hi = np.quantile(means, 1 - (1 - ci) / 2)
    return (float(diff.mean()), float(lo), float(hi))
