"""Seeded, pairwise-disjoint splits and the spurious-correlation (ρ) knob.

Pure numpy, CPU-only, deterministic per seed. Implements the contracts in
tests/test_splits.py. See docs/DATASETS.md (§Splits) and docs/EXECUTION_PLAN.md Phase 1.

No split used to *select* a feature may overlap the split used to *evaluate* its causal
effect (INTEGRITY.md §4); make_splits produces an exact partition and carve.utils.
assert_disjoint guards it downstream.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np


def _boundaries(n: int, fractions: list[float]) -> list[int]:
    """Split sizes for `n` items given fractions summing to 1, via rounded cumulative
    boundaries so the sizes total exactly n and each is within 1 of frac*n."""
    cum = np.cumsum(fractions)
    edges = np.round(cum * n).astype(int)
    edges[-1] = n  # absorb rounding drift into the last split
    sizes = np.diff(np.concatenate([[0], edges]))
    return sizes.tolist()


def make_splits(
    n_or_labels: int | np.ndarray,
    fractions: Mapping[str, float],
    seed: int = 0,
    stratify: bool = False,
) -> dict[str, np.ndarray]:
    """Partition indices [0, n) into named, pairwise-disjoint splits.

    Args:
        n_or_labels: number of items, or a 1-D label array (needed when stratify=True).
        fractions:   {split_name: fraction}; fractions should sum to ~1.
        seed:        RNG seed — deterministic and seed-sensitive.
        stratify:    if True (requires a label array), preserve each split's class balance.

    Returns: {split_name: np.ndarray[int]} — a partition of [0, n) (no overlaps, no gaps).
    """
    names = list(fractions)
    fracs = [float(fractions[k]) for k in names]
    rng = np.random.default_rng(seed)

    if isinstance(n_or_labels, np.ndarray) or (stratify and not np.isscalar(n_or_labels)):
        labels = np.asarray(n_or_labels)
        n = len(labels)
    else:
        labels = None
        n = int(n_or_labels)

    out: dict[str, list[int]] = {k: [] for k in names}

    if stratify:
        if labels is None:
            raise ValueError("stratify=True requires a label array as the first argument")
        # split each class independently, then merge per split → global balance preserved
        for cls in np.unique(labels):
            cls_idx = np.where(labels == cls)[0]
            cls_idx = rng.permutation(cls_idx)
            sizes = _boundaries(len(cls_idx), fracs)
            pos = 0
            for k, s in zip(names, sizes):
                out[k].extend(cls_idx[pos : pos + s].tolist())
                pos += s
    else:
        perm = rng.permutation(n)
        sizes = _boundaries(n, fracs)
        pos = 0
        for k, s in zip(names, sizes):
            out[k] = perm[pos : pos + s].tolist()
            pos += s

    return {k: np.array(sorted(v) if stratify else v, dtype=int) for k, v in out.items()}


def assign_artifact_presence(
    labels: np.ndarray, rho: float, rng: np.random.Generator
) -> np.ndarray:
    """Assign artifact presence so it correlates with the (binary) label at strength ρ.

    Positives carry the artifact with prob ρ; negatives with prob (1−ρ). Then
    P(present | y=1) = ρ and P(present | y=0) = 1−ρ, so ρ=0.5 is no correlation and
    ρ=1.0 makes the artifact a perfect predictor. See docs/METRICS.md / DATASETS.md.
    """
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    if not set(uniq.tolist()).issubset({0, 1}):
        raise ValueError(f"labels must be binary {{0,1}}; got classes {uniq.tolist()}")
    labels = labels.astype(bool)
    p = np.where(labels, rho, 1.0 - rho)
    return rng.random(len(labels)) < p


def realized_rho(labels: np.ndarray, present: np.ndarray) -> dict[str, float]:
    """Empirical artifact-presence rates per class — audits the realized bias strength."""
    labels = np.asarray(labels).astype(bool)
    present = np.asarray(present).astype(bool)

    def _rate(mask: np.ndarray) -> float:
        return float(present[mask].mean()) if mask.any() else 0.0

    return {
        "p_present_given_pos": _rate(labels),
        "p_present_given_neg": _rate(~labels),
        "p_present": float(present.mean()) if len(present) else 0.0,
    }
