"""DermFM-Zero-style top-k SAE-feature suppression baseline (Phase 6).

The incumbent whose finding we validate. DermFM-Zero (arXiv 2602.10624, Fig. 6i) "identified
the top five neurons most strongly activated by each artifact type and suppressed their
activations at inference time." We reimplement that recipe from the paper description (their
code is CC-BY-NC-ND / weights private — not forked): rank SAE features by MEAN activation on
artifact-present images, take the top-k, and suppress them (= carve.interventions SAE ablate).

This differs from our S_oracle only in the SELECTION RULE — activation magnitude vs detection
AUROC — so benchmarking it on controlled ground truth isolates what selection buys, and
disarms the "you didn't compare to the obvious method" reviewer.
"""
from __future__ import annotations

import numpy as np

from ..sae.discovery import feature_image_scores


def dermfmzero_select(sae, encoder, layer: int, items, top_k: int = 5) -> dict:
    """Top-k SAE features by mean activation on artifact-PRESENT images (their criterion).

    items: biased set of {"image","present"}. Returns {features, activation, best_activation}.
    Suppress the returned features via SAE ablate (harness op="ablate", S=features).
    """
    imgs = [it["image"] for it in items]
    present = np.array([bool(it["present"]) for it in items])
    scores = feature_image_scores(sae, encoder, layer, imgs)  # [N, width], max over tokens
    if present.any():
        strength = scores[present].mean(axis=0)
    else:  # degenerate select set → fall back to overall mean
        strength = scores.mean(axis=0)
    order = np.argsort(-strength)[:top_k]
    return {
        "features": order.tolist(),
        "activation": strength[order].tolist(),
        "best_activation": float(strength[order[0]]),
        "selection": "dermfmzero",
    }
