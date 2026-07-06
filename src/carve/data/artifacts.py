"""Controlled artifact injection + gold counterfactual removal.

Pure numpy, CPU-only, deterministic per seed. Implements the contracts in
tests/test_injection.py. See docs/EXECUTION_PLAN.md Phase 1 and docs/METRICS.md §Notation.

The premise of CARVE: because we *paste in* the artifact, we own the ground-truth clean
image, so ``remove(x_art, mask, source=clean)`` is an EXACT counterfactual — the truth
meter every feature-level intervention is scored against.

Each ``inject`` returns ``(image, mask)`` with:
    image  float32 HxWx3 in [0,1]      — the artifact composited over the input
    mask   float32 HxW   in [0,1]      — per-pixel artifact coverage (α-weighted footprint)

Blend is a convex combination controlled by opacity α:
    out = img·(1 − α·mask) + color·(α·mask)
so α=0 is an exact no-op and larger α means a strictly larger deviation from the input.
"""
from __future__ import annotations

import numpy as np

# ≥3 required for the MVP (docs/EXECUTION_PLAN.md Phase 1). text_overlay is optional/later.
ARTIFACT_KINDS = ["ruler", "marker_ink", "dark_corner"]


# --------------------------------------------------------------------------------------
# input normalization
# --------------------------------------------------------------------------------------
def _to_float_rgb(img: np.ndarray) -> np.ndarray:
    """Coerce uint8/float, grayscale/RGB input to float32 HxWx3 in [0,1]."""
    a = np.asarray(img)
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    a = a.astype(np.float32)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    if a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError(f"expected HxW or HxWx3 image, got shape {np.asarray(img).shape}")
    return np.clip(a, 0.0, 1.0)


def _blend(img: np.ndarray, color: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """Convex composite: exact no-op at alpha=0, deviation grows monotonically with alpha."""
    m3 = (alpha * mask)[..., None]
    out = img * (1.0 - m3) + color * m3
    return np.clip(out.astype(np.float32), 0.0, 1.0)


# --------------------------------------------------------------------------------------
# per-artifact footprint builders  → (mask float32 HxW in [0,1], color float32 [.]x3 or 3)
# --------------------------------------------------------------------------------------
def _ruler(h: int, w: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A dermoscopy ruler: a horizontal rule near one edge with evenly spaced tick marks."""
    mask = np.zeros((h, w), np.float32)
    thickness = max(1, h // 40)
    # place the rule in the top or bottom margin
    top = bool(rng.integers(0, 2))
    row = thickness if top else h - 2 * thickness
    row = int(np.clip(row, 0, h - thickness - 1))
    mask[row : row + thickness, :] = 1.0
    # ticks: short marks dropping from the rule, evenly spaced across the width
    n_ticks = max(2, w // max(4, w // 12))
    tick_len = max(2, h // 12)
    xs = np.linspace(0, w - 1, n_ticks).astype(int)
    for i, x in enumerate(xs):
        long = (i % 2 == 0)
        tl = tick_len if long else tick_len // 2
        y0 = row + thickness if top else row - tl
        y0 = int(np.clip(y0, 0, h - 1))
        y1 = int(np.clip(y0 + tl, 0, h))
        x0 = int(np.clip(x, 0, w - 1))
        mask[y0:y1, x0 : x0 + thickness] = 1.0
    color = np.full(3, 0.05, np.float32)  # near-black ruler markings
    return mask, color


def _marker_ink(h: int, w: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A small pen/marker ink dot (maps to the Bissoto `ink` category).

    Base radius is 0.10–0.22 of the image, scaled DOWN by 0.15–0.35 so the blob is a small
    ink mark (≈3–17% of the image diameter) rather than a large patch, and its size varies.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy = float(rng.uniform(0.2, 0.8)) * h
    cx = float(rng.uniform(0.2, 0.8)) * w
    scale = float(rng.uniform(0.15, 0.35))  # 15–35% of the previous size (varying)
    ry = float(rng.uniform(0.10, 0.22)) * h * scale
    rx = float(rng.uniform(0.10, 0.22)) * w * scale
    d = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    mask = (d <= 1.0).astype(np.float32)
    if mask.sum() == 0:  # guarantee a footprint on tiny canvases
        mask[int(np.clip(cy, 0, h - 1)), int(np.clip(cx, 0, w - 1))] = 1.0
    color = np.array([0.35, 0.10, 0.55], np.float32)  # purple surgical ink
    return mask, color


def _dark_corner(h: int, w: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A dark-corner vignette: coverage rises smoothly toward the image corners."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt(((yy - cy) / (cy + 1e-6)) ** 2 + ((xx - cx) / (cx + 1e-6)) ** 2)
    r = r / (r.max() + 1e-6)
    inner = 0.6  # no darkening within the central disk, ramp to full at the corners
    mask = np.clip((r - inner) / (1.0 - inner), 0.0, 1.0).astype(np.float32)
    if mask.sum() == 0:
        mask[0, 0] = 1.0
    color = np.zeros(3, np.float32)  # black corners
    return mask, color


_BUILDERS = {
    "ruler": _ruler,
    "marker_ink": _marker_ink,
    "dark_corner": _dark_corner,
}


# --------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------
def inject(
    img: np.ndarray, kind: str, alpha: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Composite artifact `kind` onto `img` at opacity `alpha`, deterministically per `rng`.

    Returns (image float32 HxWx3 in [0,1], mask float32 HxW in [0,1]).
    Same rng state ⇒ identical pixels and mask.
    """
    if kind not in _BUILDERS:
        raise ValueError(f"unknown artifact kind {kind!r}; expected one of {ARTIFACT_KINDS}")
    base = _to_float_rgb(img)
    h, w = base.shape[:2]
    mask, color = _BUILDERS[kind](h, w, rng)
    color = np.asarray(color, np.float32)
    out = _blend(base, color, mask, float(alpha))
    return out, mask.astype(np.float32)


def remove(
    art: np.ndarray,
    mask: np.ndarray,
    source: np.ndarray | None = None,
    method: str = "source",
) -> np.ndarray:
    """Undo an injection to obtain the counterfactual clean image.

    source given (default path, GOLD): return the known clean canvas → an *exact*
        counterfactual (the whole point of injecting: we own the truth).
    method='inpaint' (no source): numpy fallback that fills the masked region with the
        mean color of the unmasked pixels, leaving everything outside the mask untouched.
    """
    art = np.asarray(art).astype(np.float32)
    if source is not None:
        return _to_float_rgb(source).copy()

    if method != "inpaint":
        raise ValueError("remove() needs source=<clean image>, or method='inpaint'")
    m = np.asarray(mask) > 0.05
    out = art.copy()
    if (~m).any():
        fill = art[~m].reshape(-1, art.shape[-1]).mean(axis=0)
    else:
        fill = np.full(art.shape[-1], 0.5, np.float32)
    out[m] = fill
    return out


def make_biased_set(
    imgs: list[np.ndarray],
    labels: np.ndarray,
    kind: str,
    rho: float,
    alpha: float,
    seed: int,
) -> tuple[list[dict], dict]:
    """Materialize a ρ-biased set: inject `kind` so its presence correlates with the label
    at strength ρ. Each item keeps the metadata the pipeline needs (label, present, mask,
    clean canvas, α, kind); the summary records the realized correlation.

    When the artifact is absent the mask is all-zeros (mask.sum()==0 ⇔ not present).
    """
    from .datasets import assign_artifact_presence, realized_rho

    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    present = assign_artifact_presence(labels, rho, rng)

    items: list[dict] = []
    for i, (img, lab, pres) in enumerate(zip(imgs, labels, present)):
        clean = _to_float_rgb(img)
        if pres:
            image, mask = inject(clean, kind, alpha, np.random.default_rng(seed + i + 1))
        else:
            image, mask = clean.copy(), np.zeros(clean.shape[:2], np.float32)
        items.append(
            {
                "image": image,
                "clean": clean,
                "mask": mask,
                "label": int(lab),
                "present": bool(pres),
                "artifact_type": kind,
                "alpha": float(alpha),
            }
        )

    summary = {
        "n": len(items),
        "artifact_type": kind,
        "rho": float(rho),
        "alpha": float(alpha),
        "realized": realized_rho(labels, present),
    }
    return items, summary
