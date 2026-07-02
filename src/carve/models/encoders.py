"""Frozen, hookable MONET encoder + the analysis-layer picker (EXECUTION_PLAN Phase 2).

MONET = HF ``chanwkim/monet`` (CLIP ViT-L/14; 24 vision blocks, hidden 1024) — the load
path and read/write hookability were confirmed in scripts/02_monet_hook_probe.py.

The "activation at layer ℓ" is the **output of vision block ℓ** (residual stream,
[B, 257, 1024]) — the exact site where carve.interventions ablate/steer. Reading uses a
forward hook on ``encoder.layers[ℓ]`` so read and write refer to the same tensor.

Two decision signals ``f`` (both reported, docs/DATASETS.md §two arms):
  * zero-shot arm  — MONET's own mel-vs-nevus logit margin from text prompts (no head);
                     the purest causal ground truth. -> ``zero_shot_margin``.
  * induced arm    — a linear probe on layer-ℓ features (carve.models.probe).
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch
from PIL import Image

from ..utils import device as _device

MONET_HF_ID = "chanwkim/monet"


def _to_pil(img) -> Image.Image:
    """Coerce PIL / float[0,1] / uint8 array (HxWx3 or HxW) to an RGB PIL image."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    return Image.fromarray(a).convert("RGB")


class MonetEncoder:
    """Frozen MONET wrapped for layer-ℓ activation read + zero-shot scoring.

    Attributes: ``layers`` (ModuleList of vision blocks — the intervention sites),
    ``n_layers``, ``hidden``, ``device``.
    """

    def __init__(self, cfg=None, hf_id: str = MONET_HF_ID, dev: str | None = None):
        from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

        self.device = dev or _device((cfg.get("device", "auto") if cfg else "auto"))
        self.processor = AutoProcessor.from_pretrained(hf_id)
        self.model = AutoModelForZeroShotImageClassification.from_pretrained(hf_id)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.layers = self.model.vision_model.encoder.layers
        self.n_layers = len(self.layers)
        self.hidden = int(self.model.config.vision_config.hidden_size)

        pos = cfg.model.zero_shot_prompts.pos if cfg else "a dermoscopic image of melanoma"
        neg = cfg.model.zero_shot_prompts.neg if cfg else "a dermoscopic image of a benign nevus"
        self.prompts = [pos, neg]
        self._text = self.processor(text=self.prompts, return_tensors="pt", padding=True)
        self._text = {k: v.to(self.device) for k, v in self._text.items()}

    # -- preprocessing --------------------------------------------------------------------
    def _pixels(self, images) -> torch.Tensor:
        pil = [_to_pil(im) for im in images]
        return self.processor(images=pil, return_tensors="pt")["pixel_values"].to(self.device)

    def _batches(self, images, batch_size):
        for i in range(0, len(images), batch_size):
            yield images[i : i + batch_size]

    # -- reading activations at layer ℓ ---------------------------------------------------
    @torch.no_grad()
    def activations(self, images, layer: int, pool: str = "cls", batch_size: int = 32):
        """Return block-ℓ residual activations. pool='cls' → [N, hidden] (CLS token),
        'mean' → [N, hidden] (token mean), None → [N, tokens, hidden]."""
        outs = []
        cap = {}

        def hook(_m, _i, o):
            cap["a"] = (o[0] if isinstance(o, tuple) else o).detach()

        h = self.layers[layer].register_forward_hook(hook)
        try:
            for batch in self._batches(images, batch_size):
                self.model.vision_model(self._pixels(batch))
                a = cap["a"]
                if pool == "cls":
                    a = a[:, 0, :]
                elif pool == "mean":
                    a = a.mean(dim=1)
                outs.append(a.float().cpu())
        finally:
            h.remove()
        return torch.cat(outs, dim=0)

    # -- zero-shot decision signal --------------------------------------------------------
    @torch.no_grad()
    def zero_shot_margin(self, images, batch_size: int = 32) -> torch.Tensor:
        """MONET zero-shot logit margin z_pos − z_neg per image. Any hooks registered on
        ``self.layers`` (ablate/steer) are active during this forward pass."""
        outs = []
        for batch in self._batches(images, batch_size):
            out = self.model(pixel_values=self._pixels(batch), **self._text)
            lpi = out.logits_per_image  # [B, 2]
            outs.append((lpi[:, 0] - lpi[:, 1]).float().cpu())
        return torch.cat(outs, dim=0).to(torch.float64)

    @contextmanager
    def hooks(self, handles):
        """Register forward hooks (list of (layer_idx, fn)) for the duration of the block."""
        registered = [self.layers[i].register_forward_hook(fn) for i, fn in handles]
        try:
            yield
        finally:
            for h in registered:
                h.remove()


def load_encoder(cfg=None, which: str = "primary") -> MonetEncoder:
    """Load the frozen, hookable encoder. which='primary' → MONET (generality CLIP TBD)."""
    if which != "primary":
        raise NotImplementedError("generality encoder (CLIP ViT-B/16) not wired yet")
    return MonetEncoder(cfg)


def _items_to(images_labels):
    """Accept either a list of dicts ({'image','present'/'label'}) or (images, targets)."""
    if isinstance(images_labels, tuple):
        return list(images_labels[0]), np.asarray(images_labels[1])
    items = list(images_labels)
    return [it["image"] for it in items], items


def pick_layer(encoder: MonetEncoder, select_items, cfg=None, layers=None, target_key="present"):
    """Pre-committed rule (cfg.model.layer_rule = max artifact linear-decodability on
    `select`): pick the layer whose CLS features best linearly separate artifact
    present/absent (3-fold CV AUROC). Returns (best_layer, {layer: auroc}) — full sweep logged.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    layers = layers if layers is not None else list(
        cfg.model.layer_sweep if cfg else range(encoder.n_layers)
    )
    images = [it["image"] for it in select_items]
    y = np.array([int(it[target_key]) for it in select_items])

    sweep = {}
    for ell in layers:
        X = encoder.activations(images, ell, pool="cls").numpy()
        if len(np.unique(y)) < 2:
            sweep[int(ell)] = float("nan")
            continue
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        sweep[int(ell)] = float(cross_val_score(clf, X, y, cv=3, scoring="roc_auc").mean())
    best = max(sweep, key=lambda k: (sweep[k] if sweep[k] == sweep[k] else -1))
    return best, sweep
