"""Causal / control metrics — exact definitions in docs/METRICS.md.

The scalar decision signal ``f`` is the binary-task logit margin ``z_pos − z_neg``
(configured in ``model.decision_signal``); every effect below is a difference in ``f``.
Elementwise metrics accept torch tensors or array-likes and return torch tensors so the
run harness can pipe model outputs straight through; reductions return plain floats.
"""
from __future__ import annotations

import numpy as np
import torch

Tensorish = "torch.Tensor | np.ndarray | list | float"


def _as_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float64)
    return torch.as_tensor(np.asarray(x, dtype=float), dtype=torch.float64)


def _np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


# ── §1 bias gap ───────────────────────────────────────────────────────────────────────
def bias_gap(acc_aligned: float, acc_conflicting: float) -> float:
    """Shortcut reliance: acc(artifact-aligned) − acc(artifact-conflicting). ≈0 ⇒ no reliance."""
    return float(acc_aligned) - float(acc_conflicting)


# ── §2 input-level (gold) effect ──────────────────────────────────────────────────────
def input_effect(f_art, f_removed) -> torch.Tensor:
    """e_in = f(x_art) − f(remove(x_art)) — the artifact's TRUE causal effect, per image."""
    return _as_tensor(f_art) - _as_tensor(f_removed)


# ── §3 causal recovery (headline) ─────────────────────────────────────────────────────
def causal_recovery(f_art, f_feat, f_removed, eps: float = 1e-6) -> torch.Tensor:
    """Normalized mediation residual, per image:

        R = 1 − |f_feat − f_removed| / |f_art − f_removed|

    R=1 perfect mediation, R=0 no recovery, R<0 wrong-direction/overshoot. Images with a
    negligible gold effect (|e_in| < eps) are returned as NaN (skip in aggregation).
    """
    f_art, f_feat, f_removed = _as_tensor(f_art), _as_tensor(f_feat), _as_tensor(f_removed)
    denom = (f_art - f_removed).abs()
    resid = (f_feat - f_removed).abs()
    R = 1.0 - resid / denom
    return torch.where(denom < eps, torch.full_like(R, float("nan")), R)


# ── §4 steering selectivity (headline) ────────────────────────────────────────────────
def selectivity(f_art, f_feat_art, f_clean, f_feat_clean) -> dict:
    """RAVEL-style cause/isolation split (all values are effect magnitudes in f):

        Cause     = mean over ARTIFACT cases of |f(x_art)   − f_{−S}(x_art)|    (want high)
        Isolation = mean over CLEAN    cases of |f(x_clean) − f_{−S}(x_clean)|  (want ~0)
        Selectivity = Cause / (Cause + Isolation) ∈ [0,1]                       (1 = perfect)
    """
    cause = float(np.abs(_np(f_art) - _np(f_feat_art)).mean())
    isolation = float(np.abs(_np(f_clean) - _np(f_feat_clean)).mean())
    total = cause + isolation
    sel = cause / total if total > 0 else float("nan")
    return {"cause": cause, "isolation": isolation, "selectivity": sel}


# ── §5 off-target damage ──────────────────────────────────────────────────────────────
def off_target(acc_clean_before: float, acc_clean_after: float) -> float:
    """Collateral cost on the clean task: acc_before − acc_after (lower is better)."""
    return float(acc_clean_before) - float(acc_clean_after)


# ── §6 detection AUROC ────────────────────────────────────────────────────────────────
def detection_auroc(feature_scores, artifact_present) -> float:
    """AUROC of the feature/CAV/neuron activation as a score for artifact-present vs absent.
    Returns NaN if only one class is present (AUROC undefined)."""
    from sklearn.metrics import roc_auc_score

    scores = _np(feature_scores).ravel()
    present = _np(artifact_present).ravel().astype(int)
    if len(np.unique(present)) < 2:
        return float("nan")
    return float(roc_auc_score(present, scores))


# ── §7 steering response frontier ─────────────────────────────────────────────────────
def steering_frontier(cause_by_coeff, off_target_by_coeff) -> np.ndarray:
    """Achievable Cause-vs-off_target frontier across steering coefficients.

    Returns the non-dominated (Pareto) points — maximize Cause while minimizing off_target
    — as an ndarray of shape (m, 2) with columns [off_target, cause], sorted by off_target.
    """
    cause = _np(cause_by_coeff).ravel()
    off = _np(off_target_by_coeff).ravel()
    order = np.argsort(off, kind="stable")
    off, cause = off[order], cause[order]
    frontier, best = [], -np.inf
    for o, c in zip(off, cause):
        if c > best:  # lower off_target already; keep only if it also raises Cause
            frontier.append((o, c))
            best = c
    return np.array(frontier, dtype=float).reshape(-1, 2)
