"""Activation-level ablation/steering of SAE features via residual-stream hooks (Phase 5).

The SAE lives in MONET's raw layer-ℓ activation space, so we intervene exactly at the
block-ℓ output (the tensor the SAE was trained on and that carve.models.encoders reads):

  ablate(S): remove the feature's reconstructed contribution from the activation,
             a ← a − Σ_{f∈S} z_f(a) · W_dec[f]   (z_f = the SAE's TopK latent for feature f)
             — only where the feature actually fires; the SAE error term is untouched.
  steer(S): push along the (unit) decoder direction with a fixed coefficient,
             a ← a − c · Σ_{f∈S} W_dec[f]

The input image is UNCHANGED; only activations at ℓ are edited, then the forward completes
(so the decision signal f reflects the edit). Hooks compose with the encoder's read hook,
so both the zero-shot and probe arms see the intervention.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch


def _as_idx(features) -> torch.Tensor:
    t = torch.as_tensor(features)
    return t.reshape(1) if t.ndim == 0 else t


def sae_ablate_fn(sae, features):
    """Return a(·)→a′ that subtracts features S' reconstructed contribution."""
    feats = _as_idx(features)

    def fn(a: torch.Tensor) -> torch.Tensor:
        shape = a.shape
        flat = a.reshape(-1, shape[-1])
        idx = feats.to(flat.device)
        z = sae.encode(flat)                       # [*, m] TopK latents
        contrib = z[:, idx] @ sae.W_dec[idx]       # [*, d] sum of the S features' output
        return (flat - contrib).reshape(shape)

    return fn


def sae_steer_fn(sae, features, coeff: float):
    """Return a(·)→a′ that subtracts coeff · Σ unit decoder directions of S."""
    feats = _as_idx(features)

    def fn(a: torch.Tensor) -> torch.Tensor:
        direction = sae.W_dec[feats.to(a.device)].sum(dim=0)   # [d]
        return a - float(coeff) * direction

    return fn


@contextmanager
def intervene(encoder, layer: int, fn):
    """Register a forward hook on block ℓ that rewrites its output activation with fn."""
    def hook(_m, _i, o):
        if isinstance(o, tuple):
            return (fn(o[0]),) + tuple(o[1:])
        return fn(o)

    h = encoder.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


def ablate(encoder, layer: int, sae, features):
    """Context manager: ablate SAE feature(s) S at block ℓ for the enclosed forward passes."""
    return intervene(encoder, layer, sae_ablate_fn(sae, features))


def steer(encoder, layer: int, sae, features, coeff: float):
    """Context manager: steer along S' decoder direction with coefficient c at block ℓ."""
    return intervene(encoder, layer, sae_steer_fn(sae, features, coeff))
