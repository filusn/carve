"""Phase-1 unit tests for seeded disjoint splits and the rho knob. CPU-only."""
import numpy as np
import pytest

from carve.data.datasets import (
    assign_artifact_presence,
    make_splits,
    realized_rho,
)

FRACTIONS = {"probe_train": 0.40, "sae_train": 0.25, "select": 0.10, "eval": 0.15, "test": 0.10}


def test_splits_disjoint_and_complete():
    splits = make_splits(1000, FRACTIONS, seed=0)
    all_idx = np.concatenate(list(splits.values()))
    assert len(all_idx) == 1000
    assert len(np.unique(all_idx)) == 1000  # partition: no overlaps, no gaps
    for k, frac in FRACTIONS.items():
        assert abs(len(splits[k]) - frac * 1000) <= 1


def test_splits_deterministic_and_seed_sensitive():
    a = make_splits(500, FRACTIONS, seed=1)
    b = make_splits(500, FRACTIONS, seed=1)
    c = make_splits(500, FRACTIONS, seed=2)
    for k in FRACTIONS:
        assert np.array_equal(a[k], b[k])
    assert any(not np.array_equal(a[k], c[k]) for k in FRACTIONS)


def test_splits_are_disjoint_pairwise():
    splits = make_splits(777, FRACTIONS, seed=3)
    keys = list(splits)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert not (set(splits[keys[i]].tolist()) & set(splits[keys[j]].tolist()))


def test_splits_stratified_preserve_balance():
    rng = np.random.default_rng(0)
    labels = (rng.random(2000) < 0.3).astype(int)  # ~30% positive
    splits = make_splits(labels, FRACTIONS, seed=0, stratify=True)
    for k in FRACTIONS:
        pos_rate = labels[splits[k]].mean()
        assert abs(pos_rate - 0.3) < 0.05  # class balance held within each split


@pytest.mark.parametrize("rho", [0.5, 0.7, 0.9, 1.0])
def test_assign_presence_realizes_target_rho(rho):
    rng = np.random.default_rng(0)
    labels = (rng.random(8000) < 0.5).astype(int)
    present = assign_artifact_presence(labels, rho, rng)
    r = realized_rho(labels, present)
    assert abs(r["p_present_given_pos"] - rho) < 0.03
    assert abs(r["p_present_given_neg"] - (1 - rho)) < 0.03
    if rho == 1.0:  # perfect predictor
        assert present[labels == 1].all() and not present[labels == 0].any()


def test_assign_presence_rejects_non_binary():
    with pytest.raises(ValueError):
        assign_artifact_presence(np.array([0, 1, 2]), 0.7, np.random.default_rng(0))
