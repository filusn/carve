"""Random-feature / random-neuron control (Phase 6) — the sanity floor.

Ablating a randomly chosen SAE feature (or raw neuron) must recover ≈nothing and stay
selective; a method that beats random is doing real work, and a "recovery" that random also
achieves is an artifact of the metric, not the method. Same count as the real method's S so
the intervention budget matches.
"""
from __future__ import annotations

import numpy as np


def random_sae_features(sae, n: int, rng: np.random.Generator) -> list[int]:
    """n distinct random SAE feature indices."""
    n = min(int(n), int(sae.width))
    return rng.choice(int(sae.width), size=n, replace=False).astype(int).tolist()


def random_raw_neurons(d: int, n: int, rng: np.random.Generator) -> list[int]:
    """n distinct random raw-activation neuron indices in [0, d)."""
    n = min(int(n), int(d))
    return rng.choice(int(d), size=n, replace=False).astype(int).tolist()
