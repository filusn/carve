"""Linear probe on frozen layer-ℓ features + the unified decision signal f (Phase 2).

The induced bias arm trains a light, class-weighted linear probe on layer-ℓ CLS features
of a ρ-biased set (the 6:1 mel:nevus imbalance is handled by class weighting). The decision
signal ``f`` used by every metric is the **logit margin** ``z_pos − z_neg``:
  * induced  arm → probe decision_function (w·feat + b, the binary logit margin);
  * zero-shot arm → encoder.zero_shot_margin (pass probe=None).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .encoders import MonetEncoder


@dataclass
class LinearProbe:
    clf: object          # fitted sklearn LogisticRegression
    layer: int
    pool: str = "cls"

    def margin(self, feats: np.ndarray) -> np.ndarray:
        """Binary logit margin w·feat + b for each row."""
        return self.clf.decision_function(np.asarray(feats))


def train_probe(encoder: MonetEncoder, layer: int, probe_train_items, cfg=None) -> LinearProbe:
    """Fit a class-weighted logistic-regression probe on layer-ℓ CLS features.

    `probe_train_items`: iterable of dicts with 'image' and 'label' (0/1). Features are
    extracted from the frozen encoder, so only the linear head is trained.
    """
    from sklearn.linear_model import LogisticRegression

    items = list(probe_train_items)
    images = [it["image"] for it in items]
    y = np.array([int(it["label"]) for it in items])
    X = encoder.activations(images, layer, pool="cls").numpy()
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X, y)
    return LinearProbe(clf=clf, layer=layer, pool="cls")


def f_decision(
    probe: LinearProbe | None,
    encoder: MonetEncoder,
    images,
    layer: int | None = None,
    batch_size: int = 32,
) -> torch.Tensor:
    """The scalar decision signal f per image (logit margin z_pos − z_neg).

    probe=None → zero-shot arm (MONET text prompts). Otherwise the induced-arm probe on
    layer-ℓ features. Runs a forward pass, so any ablate/steer hooks on the encoder are
    reflected in f (that is how feature-level interventions move the decision).
    """
    if probe is None:
        return encoder.zero_shot_margin(images, batch_size=batch_size)
    ell = layer if layer is not None else probe.layer
    feats = encoder.activations(images, ell, pool=probe.pool, batch_size=batch_size).numpy()
    return torch.as_tensor(probe.margin(feats), dtype=torch.float64)


def probe_accuracy(probe: LinearProbe, encoder: MonetEncoder, items) -> float:
    """Clean-task accuracy of the probe on labelled items (for bias-gap / off-target)."""
    items = list(items)
    images = [it["image"] for it in items]
    y = np.array([int(it["label"]) for it in items])
    pred = (f_decision(probe, encoder, images).numpy() > 0).astype(int)
    return float((pred == y).mean())
