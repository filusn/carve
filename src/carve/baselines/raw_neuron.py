"""Raw-neuron ablation baseline (Phase 6).

The key reviewer objection to an SAE result is *"what does the sparse dictionary add over
the raw activation neurons?"* This baseline answers it on identical ground truth: pick the
raw block-ℓ neuron(s) whose activation best detects the injected artifact on `select`, then
mean-ablate them (replace with their expected value) and measure the same causal-recovery /
selectivity metrics on the disjoint `eval` split.

Selection mirrors the SAE oracle (AUROC on the injected present/absent labels) but ranks by
|AUROC−0.5| because a raw neuron may encode the artifact by firing *up* or *down*.
Feature activation per image = MAX over tokens (artifacts are spatially localized), matching
carve.sae.discovery.feature_image_scores.
"""
from __future__ import annotations

import numpy as np
import torch

from ..sae.discovery import _auroc_columns


@torch.no_grad()
def raw_image_scores(encoder, layer: int, images, batch_size: int = 16, pool: str = "max"):
    """Per-image raw block-ℓ activations, pooled over tokens → np.ndarray [N, d]."""
    outs = []
    for i in range(0, len(images), batch_size):
        acts = encoder.activations(images[i : i + batch_size], layer, pool=None)  # [b,T,d]
        outs.append(acts.amax(dim=1) if pool == "max" else acts.mean(dim=1))
    return torch.cat(outs, dim=0).numpy()


def raw_neuron_select(encoder, layer: int, items, top_k: int = 1) -> dict:
    """Most artifact-correlated raw neuron(s) on `select`, by |AUROC−0.5| (up- or down-firing).

    items: biased set of {"image","present"}. Returns {neurons, auroc, best_auroc}.
    """
    imgs = [it["image"] for it in items]
    present = np.array([int(it["present"]) for it in items])
    scores = raw_image_scores(encoder, layer, imgs)
    aurocs = _auroc_columns(scores, present)
    aurocs = np.where(np.isnan(aurocs), 0.5, aurocs)
    order = np.argsort(-np.abs(aurocs - 0.5))[:top_k]
    return {
        "neurons": order.tolist(),
        "auroc": aurocs[order].tolist(),
        "best_auroc": float(aurocs[order[0]]),
        "selection": "raw_neuron",
    }


def raw_neuron_ablate_fn(neurons, baseline):
    """Return a(·)→a′ that mean-ablates neuron(s): replace coord n with its expected value
    baseline[n] (removing the neuron's per-image information, the raw analogue of SAE ablate).

    baseline: [d] mean activation vector over a reference (e.g. sae_train) set.
    """
    idx = torch.as_tensor(neurons).reshape(-1)
    base = torch.as_tensor(np.asarray(baseline), dtype=torch.float32)

    def fn(a: torch.Tensor) -> torch.Tensor:
        a = a.clone()
        a[..., idx.to(a.device)] = base[idx].to(a.device, a.dtype)
        return a

    return fn


@torch.no_grad()
def reference_mean(encoder, layer: int, images, batch_size: int = 16) -> np.ndarray:
    """Mean block-ℓ activation over tokens & images → [d] baseline for mean-ablation."""
    tot, n = None, 0
    for i in range(0, len(images), batch_size):
        acts = encoder.activations(images[i : i + batch_size], layer, pool=None)  # [b,T,d]
        s = acts.reshape(-1, acts.shape[-1]).sum(dim=0)
        tot = s if tot is None else tot + s
        n += acts.shape[0] * acts.shape[1]
    return (tot / max(1, n)).cpu().numpy()
