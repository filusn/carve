"""Input-vs-feature mediation: the quantities causal_recovery is computed from (Phase 5).

The gold (input-level) effect e_in = f(x_art) − f(remove(x_art)) is what a feature-level
intervention must reproduce. f_intervened runs f with feature(s) S ablated/steered at ℓ
(input UNCHANGED); causal_recovery(f_art, f_intervened, f_removed) then says how much of e_in
the intervention recovered. f is carve.models.probe.f_decision (probe=None ⇒ zero-shot arm).
"""
from __future__ import annotations

import torch

from ..models.probe import f_decision
from .hooks import ablate, steer


def f_removed(probe, encoder, layer, x_removed, batch_size: int = 32) -> torch.Tensor:
    """f(remove(x_art)) — the clean-source decision (target of the mediation)."""
    return f_decision(probe, encoder, x_removed, layer, batch_size=batch_size)


def input_effect(probe, encoder, layer, x_art, x_removed, batch_size: int = 32) -> torch.Tensor:
    """e_in = f(x_art) − f(remove(x_art)) — the artifact's TRUE causal effect, per image."""
    fa = f_decision(probe, encoder, x_art, layer, batch_size=batch_size)
    fr = f_decision(probe, encoder, x_removed, layer, batch_size=batch_size)
    return fa - fr


def f_intervened(probe, encoder, layer, images, sae, S, op: str = "ablate",
                 coeff: float | None = None, batch_size: int = 32) -> torch.Tensor:
    """f_{−S}(·): decision signal with feature(s) S ablated (or steered) at ℓ, input unchanged."""
    if op == "ablate":
        cm = ablate(encoder, layer, sae, S)
    elif op == "steer":
        if coeff is None:
            raise ValueError("steer requires a coeff")
        cm = steer(encoder, layer, sae, S, coeff)
    else:
        raise ValueError(f"unknown op {op!r} (expected 'ablate' or 'steer')")
    with cm:
        return f_decision(probe, encoder, images, layer, batch_size=batch_size)


def feature_effect(probe, encoder, layer, x_art, sae, S, op: str = "ablate",
                   coeff: float | None = None, batch_size: int = 32) -> torch.Tensor:
    """e_S = f(x_art) − f_{−S}(x_art) — how much intervening on S moves the decision."""
    fa = f_decision(probe, encoder, x_art, layer, batch_size=batch_size)
    ff = f_intervened(probe, encoder, layer, x_art, sae, S, op=op, coeff=coeff, batch_size=batch_size)
    return fa - ff
