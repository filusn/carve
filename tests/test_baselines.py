"""Unit tests for the Phase-6 baselines + the generalized run_cell interface.
Toy tensors + a tiny fake encoder — no MONET/data, runs anywhere."""
import numpy as np
import torch
import torch.nn as nn

from carve.baselines import (
    dermfmzero_select,
    random_raw_neurons,
    random_sae_features,
    raw_neuron_ablate_fn,
    raw_neuron_select,
)
from carve.eval.harness import run_cell


class FakeEncoder:
    """Images are float vectors [d]. The block-ℓ activation is the vector passed through a
    real nn.Identity (so carve.interventions forward-hooks actually fire); pool=None
    broadcasts over T tokens, and the zero-shot margin is the (hooked) vector's sum — so
    ablating a dimension via act_fn changes the decision exactly as on real MONET."""

    def __init__(self, d=16, tokens=3):
        self.d, self.tokens = d, tokens
        self.layers = [nn.Identity() for _ in range(24)]   # real, hookable modules

    def _hooked(self, images, layer):
        X = torch.as_tensor(np.stack(images), dtype=torch.float32)  # [b,d]
        return self.layers[layer](X)                               # forward hook (if any) fires

    def activations(self, images, layer, pool=None, batch_size=32):
        X = self._hooked(images, layer)
        if pool is None:
            return X[:, None, :].repeat(1, self.tokens, 1)          # [b,T,d]
        return X

    def zero_shot_margin(self, images, batch_size=32):
        return self._hooked(images, 12).sum(dim=1).to(torch.float64)


# ── raw-neuron baseline ───────────────────────────────────────────────────────────────
def test_raw_neuron_select_finds_artifact_dim():
    rng = np.random.default_rng(0)
    d, N = 16, 80
    present = rng.random(N) < 0.5
    X = rng.normal(0, 0.1, size=(N, d)).astype(np.float32)
    X[present, 7] += 3.0                                            # the artifact neuron
    items = [{"image": X[i], "present": bool(present[i])} for i in range(N)]
    out = raw_neuron_select(FakeEncoder(d), 12, items, top_k=1)
    assert out["neurons"] == [7]
    assert out["best_auroc"] > 0.9


def test_raw_neuron_ablate_fn_replaces_with_baseline():
    d = 16
    base = np.arange(d, dtype=np.float32)
    fn = raw_neuron_ablate_fn([7], base)
    a = torch.ones(3, 3, d)
    out = fn(a)
    assert torch.allclose(out[..., 7], torch.full((3, 3), 7.0))     # ablated → baseline value
    assert torch.allclose(out[..., 0], torch.ones(3, 3))            # others untouched
    assert torch.allclose(a[..., 7], torch.ones(3, 3))             # input not mutated in place


# ── DermFM-Zero suppression (selection rule) ──────────────────────────────────────────
def test_dermfmzero_selects_top_activated_present_features(monkeypatch):
    import carve.baselines.dermfmzero_suppress as m

    N, width = 40, 20
    present = np.zeros(N, bool)
    present[:20] = True
    scores = np.zeros((N, width), np.float32)
    scores[present, 5] = 2.0          # strongest on present
    scores[present, 9] = 1.0          # second
    scores[:, 0] = 0.4                # fires on everything, but weaker on present
    monkeypatch.setattr(m, "feature_image_scores", lambda *a, **k: scores)
    items = [{"image": i, "present": bool(present[i])} for i in range(N)]
    out = m.dermfmzero_select(sae=None, encoder=None, layer=12, items=items, top_k=2)
    assert out["features"] == [5, 9]


# ── random control ────────────────────────────────────────────────────────────────────
def test_random_features_distinct_and_in_range():
    class _SAE:
        width = 100

    rng = np.random.default_rng(0)
    f = random_sae_features(_SAE, 5, rng)
    assert len(set(f)) == 5 and all(0 <= x < 100 for x in f)
    n = random_raw_neurons(1024, 6, rng)
    assert len(set(n)) == 6 and all(0 <= x < 1024 for x in n)


# ── input-removal oracle via run_cell ─────────────────────────────────────────────────
def test_run_cell_oracle_is_full_recovery_and_selective():
    enc = FakeEncoder(4)
    clean = [np.zeros(4, np.float32) for _ in range(12)]
    art = [np.array([1, 0, 0, 0], np.float32) for _ in range(12)]     # e_in = 1 per image
    meta = {"model": "m", "artifact": "ruler", "rho": 0.9, "opacity": 1.0,
            "selection": "oracle", "method": "input_remove"}
    rec = run_cell(meta, enc, 12, None, [], art, clean, oracle=True, bootstrap=50)
    assert abs(rec["R_median"] - 1.0) < 1e-9        # input removal recovers e_in exactly
    assert abs(rec["selectivity"] - 1.0) < 1e-9     # zero effect on clean → perfectly selective
    assert abs(rec["off_target"]) < 1e-9


def test_run_cell_act_fn_path_matches_manual_hook():
    # A generic activation editor plugs in through act_fn and moves the decision.
    enc = FakeEncoder(4)
    clean = [np.zeros(4, np.float32) for _ in range(8)]
    art = [np.array([2, 0, 0, 0], np.float32) for _ in range(8)]
    # ablate raw neuron 0 to baseline 0 → art decision should collapse toward clean.
    fn = raw_neuron_ablate_fn([0], np.zeros(4, np.float32))
    meta = {"model": "m", "artifact": "ruler", "rho": 0.9, "opacity": 1.0,
            "selection": "raw_neuron", "method": "raw_ablate"}
    rec = run_cell(meta, enc, 12, None, [0], art, clean, act_fn=fn, bootstrap=50)
    assert rec["R_median"] > 0.9    # zeroing the artifact neuron recovers the effect here
