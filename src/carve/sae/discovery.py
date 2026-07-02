"""Candidate artifact-feature discovery (Phase 4), kept in two strictly separate settings:

  * select_oracle       — upper bound: pick feature(s) by highest artifact detection AUROC
                          on the `select` split (uses the injected present/absent labels).
  * discover_unsupervised — realistic: rank features WITHOUT injected labels (activation
                          variance across a mixed set); check later if the oracle feature is found.

INTEGRITY (docs/INTEGRITY.md §4): selection happens on `select`; the causal effect is later
evaluated on the disjoint `eval` split. Keep S_oracle and S_discovered labelled separately.
Feature activation per image = MAX over tokens (artifacts are spatially localized).
"""
from __future__ import annotations

import numpy as np
import torch

from .model import TopKSAE


@torch.no_grad()
def feature_image_scores(sae: TopKSAE, encoder, layer: int, images, batch_size: int = 16):
    """Per-image SAE feature activations, max-pooled over tokens → np.ndarray [N, width]."""
    dev = next(sae.parameters()).device
    outs = []
    for i in range(0, len(images), batch_size):
        acts = encoder.activations(images[i : i + batch_size], layer, pool=None)  # [b,T,d] cpu
        b, T, d = acts.shape
        z = sae.encode(acts.reshape(b * T, d).to(dev))          # [b*T, width]
        outs.append(z.reshape(b, T, sae.width).amax(dim=1).cpu())  # max over tokens
    return torch.cat(outs, dim=0).numpy()


def _auroc_columns(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized column-wise AUROC (tie-averaged ranks) of each feature vs binary y."""
    from scipy.stats import rankdata

    y = np.asarray(y).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return np.full(scores.shape[1], np.nan)
    ranks = rankdata(scores, axis=0)                 # average ties (many zeros tie)
    sum_pos = ranks[y].sum(axis=0)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def select_oracle(sae: TopKSAE, encoder, layer: int, select_items, top_m: int = 1) -> dict:
    """Choose artifact feature(s) by highest detection AUROC on `select` (uses injected
    present/absent). Returns {features, auroc, best_auroc}."""
    imgs = [it["image"] for it in select_items]
    present = np.array([int(it["present"]) for it in select_items])
    S = feature_image_scores(sae, encoder, layer, imgs)
    aurocs = _auroc_columns(S, present)
    aurocs = np.where(np.isnan(aurocs), 0.5, aurocs)
    order = np.argsort(-aurocs)[:top_m]
    return {
        "features": order.tolist(),
        "auroc": aurocs[order].tolist(),
        "best_auroc": float(aurocs[order[0]]),
        "selection": "oracle",
    }


def discover_unsupervised(sae: TopKSAE, encoder, layer: int, items, top_m: int = 5) -> dict:
    """Rank features WITHOUT injected labels: by activation variance across the (mixed) set
    — high-variance features are candidate concepts. Returns {features, variance}."""
    imgs = [it["image"] for it in items]
    S = feature_image_scores(sae, encoder, layer, imgs)
    var = S.var(axis=0)
    order = np.argsort(-var)[:top_m]
    return {
        "features": order.tolist(),
        "variance": var[order].tolist(),
        "selection": "discovered",
    }


def discovery_precision_at_k(discovered: dict, oracle: dict, k: int | None = None) -> float:
    """Fraction of the top-k unsupervised features that are also oracle artifact features."""
    k = k if k is not None else len(discovered["features"])
    disc = set(discovered["features"][:k])
    return len(disc & set(oracle["features"])) / max(1, len(disc))
