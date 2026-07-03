"""CAV / Reveal2Revise concept-suppression baseline (Phase 6, 2nd pass).

A Concept Activation Vector (Kim et al., TCAV; used for *correction* in Reveal2Revise /
ClArC, Anders et al.) is the normal of a linear classifier that separates artifact-PRESENT
from artifact-ABSENT activations at block ℓ. Mitigation edits activations at inference to
clamp the component along that direction back to its clean baseline — the concept-removal
("ClArC"-style) operation.

Same footing as the other methods: selection uses the injected present/absent labels on
`select`; the intervention plugs into carve.eval.harness.run_cell as an act_fn and is scored
on the disjoint `eval` split. Unlike the SAE feature / raw neuron (a single coordinate), the
CAV is an arbitrary learned linear direction — the fair "linear concept removal" comparator.
Activation per image = MAX over tokens (localized artifacts), matching the other selectors.
"""
from __future__ import annotations

import numpy as np
import torch

from ..metrics.causal import detection_auroc
from .raw_neuron import raw_image_scores


def fit_cav(encoder, layer: int, items, pool: str = "max") -> dict:
    """Fit the artifact CAV on `select`: a class-weighted logistic separator of present vs
    absent pooled block-ℓ activations. Returns {direction (unit [d]), best_auroc, selection}.
    """
    from sklearn.linear_model import LogisticRegression

    imgs = [it["image"] for it in items]
    present = np.array([int(it["present"]) for it in items])
    X = raw_image_scores(encoder, layer, imgs, pool=pool)          # [N, d]
    if len(np.unique(present)) < 2:                                # degenerate select set
        w = np.zeros(X.shape[1], np.float32)
        return {"direction": w, "best_auroc": 0.5, "selection": "cav"}
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, present)
    w = clf.coef_.ravel().astype(np.float32)
    w /= float(np.linalg.norm(w)) + 1e-8                          # unit CAV direction
    auroc = detection_auroc(X @ w, present)                       # projection detects the artifact
    return {"direction": w, "best_auroc": float(auroc), "selection": "cav"}


def cav_suppress_fn(direction, baseline_proj: float):
    """Return a(·)→a′ that clamps the CAV coordinate to its clean baseline:

        a′ = a − (a·û − b)·û        (û = unit CAV, b = mean projection on clean acts)

    i.e. remove the artifact-concept component beyond its clean level. Applied per token over
    the [*, d] residual; symmetric with raw-neuron mean-ablation but along a learned direction.
    """
    u = torch.as_tensor(np.asarray(direction), dtype=torch.float32)
    b = float(baseline_proj)

    def fn(a: torch.Tensor) -> torch.Tensor:
        uu = u.to(a.device, a.dtype)
        proj = a @ uu                                            # [...] component along û
        return a - (proj - b).unsqueeze(-1) * uu

    return fn


@torch.no_grad()
def cav_baseline_proj(acts, direction) -> float:
    """Mean projection ⟨a, û⟩ over reference (clean) activations → the clamp target b."""
    u = torch.as_tensor(np.asarray(direction), dtype=torch.float32)
    a = acts if isinstance(acts, torch.Tensor) else torch.as_tensor(np.asarray(acts))
    return float((a.to(torch.float32) @ u).mean())
