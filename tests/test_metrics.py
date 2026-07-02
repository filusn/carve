"""Phase-5 unit tests for the metrics, on toy tensors with answers known by hand.
CPU-only, no model, no data. Definitions: docs/METRICS.md."""
import numpy as np
import pytest
import torch

from carve.metrics import causal as C
from carve.metrics import stats as S


# ── §1 bias gap ────────────────────────────────────────────────────────────────────────
def test_bias_gap():
    assert C.bias_gap(0.95, 0.60) == pytest.approx(0.35)
    assert C.bias_gap(0.8, 0.8) == pytest.approx(0.0)  # no reliance


# ── §2 input effect ────────────────────────────────────────────────────────────────────
def test_input_effect_elementwise():
    e = C.input_effect(torch.tensor([0.55, 0.20]), torch.tensor([0.20, 0.20]))
    assert torch.allclose(e, torch.tensor([0.35, 0.0], dtype=e.dtype))


def test_input_effect_accepts_numpy_and_lists():
    e = C.input_effect([1.0, 2.0], np.array([0.5, 0.5]))
    assert isinstance(e, torch.Tensor)
    assert torch.allclose(e, torch.tensor([0.5, 1.5], dtype=e.dtype))


# ── §3 causal recovery ─────────────────────────────────────────────────────────────────
def test_causal_recovery_perfect_and_none_and_overshoot():
    f_art = torch.tensor([0.55, 0.55, 0.55])
    f_removed = torch.tensor([0.20, 0.20, 0.20])
    # feat == removed → perfect (R=1); feat == art → none (R=0); feat overshoots → R<0
    f_feat = torch.tensor([0.20, 0.55, -0.50])  # -0.50 is 0.70 *past* removed (2×e_in)
    R = C.causal_recovery(f_art, f_feat, f_removed, eps=1e-6)
    assert R[0].item() == pytest.approx(1.0)
    assert R[1].item() == pytest.approx(0.0)
    assert R[2].item() == pytest.approx(-1.0)  # |(-0.50)-0.20| / 0.35 = 2 → R = -1


def test_causal_recovery_partial():
    # feat halfway between art and removed → R = 0.5
    R = C.causal_recovery([0.6], [0.4], [0.2], eps=1e-6)
    assert R[0].item() == pytest.approx(0.5)


def test_causal_recovery_guards_tiny_effect_with_nan():
    R = C.causal_recovery([0.30001], [0.10], [0.30], eps=1e-3)  # |e_in|=1e-5 < eps
    assert torch.isnan(R[0])


# ── §4 selectivity ─────────────────────────────────────────────────────────────────────
def test_selectivity_perfectly_selective():
    # intervention moves artifact cases a lot, clean cases not at all → selectivity = 1
    out = C.selectivity(f_art=[1.0, 1.0], f_feat_art=[0.0, 0.0],
                        f_clean=[0.5, 0.5], f_feat_clean=[0.5, 0.5])
    assert out["cause"] == pytest.approx(1.0)
    assert out["isolation"] == pytest.approx(0.0)
    assert out["selectivity"] == pytest.approx(1.0)


def test_selectivity_half_when_symmetric():
    out = C.selectivity([1.0], [0.0], [1.0], [0.0])  # equal cause & isolation
    assert out["selectivity"] == pytest.approx(0.5)


def test_selectivity_nan_when_no_effect_anywhere():
    out = C.selectivity([1.0], [1.0], [1.0], [1.0])
    assert np.isnan(out["selectivity"])


# ── §5 off-target ──────────────────────────────────────────────────────────────────────
def test_off_target():
    assert C.off_target(0.90, 0.88) == pytest.approx(0.02)


# ── §6 detection AUROC ─────────────────────────────────────────────────────────────────
def test_detection_auroc_perfect_and_random():
    present = [0, 0, 1, 1]
    assert C.detection_auroc([0.1, 0.2, 0.8, 0.9], present) == pytest.approx(1.0)
    assert C.detection_auroc([0.9, 0.8, 0.2, 0.1], present) == pytest.approx(0.0)


def test_detection_auroc_single_class_is_nan():
    assert np.isnan(C.detection_auroc([0.1, 0.2, 0.3], [1, 1, 1]))


# ── §7 steering frontier ───────────────────────────────────────────────────────────────
def test_steering_frontier_drops_dominated_points():
    # coeffs: (off,cause) = (0.1,0.2),(0.2,0.5),(0.3,0.4[dominated]),(0.4,0.9)
    off = [0.1, 0.2, 0.3, 0.4]
    cause = [0.2, 0.5, 0.4, 0.9]
    fr = C.steering_frontier(cause, off)
    assert fr.shape[1] == 2
    # the dominated (0.3, 0.4) point is removed; the other three remain, sorted by off
    assert np.allclose(fr, [[0.1, 0.2], [0.2, 0.5], [0.4, 0.9]])


# ── stats: bootstrap ───────────────────────────────────────────────────────────────────
def test_bootstrap_ci_brackets_median_and_is_deterministic():
    vals = np.arange(0.0, 100.0)
    lo, hi = S.bootstrap_ci(vals, n=500, rng=0)
    assert lo < np.median(vals) < hi
    lo2, hi2 = S.bootstrap_ci(vals, n=500, rng=0)
    assert (lo, hi) == (lo2, hi2)  # same seed → identical


def test_bootstrap_ci_drops_nans_and_handles_empty():
    lo, hi = S.bootstrap_ci([1.0, np.nan, 1.0, 1.0], n=200, rng=1)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)
    assert all(np.isnan(v) for v in S.bootstrap_ci([np.nan, np.nan], rng=1))


def test_paired_bootstrap_diff_detects_and_denies_wins():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.1, 500)  # a consistently ~1 above b
    b = rng.normal(0.0, 0.1, 500)
    md, lo, hi = S.paired_bootstrap_diff(a, b, n=500, rng=0)
    assert md == pytest.approx(1.0, abs=0.1) and lo > 0  # CI excludes 0 → real win

    z1 = rng.normal(0.0, 1.0, 500)
    z2 = rng.normal(0.0, 1.0, 500)  # no real difference
    _, lo2, hi2 = S.paired_bootstrap_diff(z1, z2, n=500, rng=0)
    assert lo2 < 0 < hi2  # CI overlaps 0 → do NOT claim a win


def test_paired_bootstrap_requires_equal_length():
    with pytest.raises(ValueError):
        S.paired_bootstrap_diff([1, 2, 3], [1, 2])
