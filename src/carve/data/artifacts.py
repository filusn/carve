"""Controlled artifact injection + gold counterfactual removal.

CPU-only, deterministic per seed. Implements the contracts in tests/test_injection.py.
See docs/EXECUTION_PLAN.md Phase 1 and docs/METRICS.md §Notation. Numpy for the compositing
math; Pillow only to rasterize the real photographic overlays (rulers/arrows) — the placement
is still driven entirely by the passed numpy RNG, so injection stays deterministic per seed.

The premise of CARVE: because we *paste in* the artifact, we own the ground-truth clean
image, so ``remove(x_art, mask, source=clean)`` is an EXACT counterfactual — the truth
meter every feature-level intervention is scored against.

Each ``inject`` returns ``(image, mask)`` with:
    image  float32 HxWx3 in [0,1]      — the artifact composited over the input
    mask   float32 HxW   in [0,1]      — per-pixel artifact coverage (α-weighted footprint)

Blend is a convex combination controlled by opacity α:
    out = img·(1 − α·mask) + color·(α·mask)
so α=0 is an exact no-op and larger α means a strictly larger deviation from the input.
``color`` may be a single RGB (synthetic marks) or a full HxWx3 image (real overlays); the
blend broadcasts either way.

Two families of artifacts:
  • real overlays (DEFAULT): ``ruler`` (dermoscopy rulers, m*.png) and ``arrow`` (annotation
    arrows, a*.png), real PNG templates bundled under ``overlays/`` and composited with their
    own alpha. ``overlay_both`` layers a ruler + an arrow. These are the pipeline default.
  • synthetic marks (LEGACY, opt-in): ``ruler_synthetic`` (procedural rule+ticks),
    ``marker_ink`` (pen dot), ``dark_corner`` (original smooth vignette), ``black_corner``
    (hard circular cutoff mimicking a real dermoscope's circular field-of-view) — the
    original numpy-drawn set.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

# DEFAULT set = real photographic overlays (rulers + arrows). See _BUILDERS below.
ARTIFACT_KINDS = ["ruler", "arrow"]
# Original synthetic marks, kept as an opt-in option (set configs artifacts.types to use them).
LEGACY_ARTIFACT_KINDS = ["ruler_synthetic", "marker_ink", "dark_corner", "black_corner"]


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


def _black_corner(h: int, w: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A dermoscope's circular field-of-view: a HARD circular cutoff.

    Inside a centred circle the skin is fully visible (mask=0); outside it is solid black
    (mask=1) with a sharp edge — no gradient. The circle is the inscribed circle,
    ``min(h, w) / 2`` (so it touches the shorter-side edges and the four corners are black),
    with a small deterministic radius/centre jitter from ``rng`` so it varies per image while
    staying a hard circle. mask is binary in {0.0, 1.0}; corners are always covered so
    ``mask.sum() > 0`` for any radius ≤ the inscribed circle.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # subtle centre jitter: a few percent of the image, deterministic in rng
    cy = (h - 1) / 2.0 + float(rng.uniform(-0.03, 0.03)) * h
    cx = (w - 1) / 2.0 + float(rng.uniform(-0.03, 0.03)) * w
    radius = (min(h, w) / 2.0) * float(rng.uniform(0.92, 1.0))  # small radius jitter
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask = (dist > radius).astype(np.float32)  # sharp edge: black strictly outside the circle
    color = np.zeros(3, np.float32)  # solid black outside the field-of-view
    return mask, color


# --------------------------------------------------------------------------------------
# real photographic overlays (rulers / arrows) — bundled PNG templates, RNG-driven placement
# --------------------------------------------------------------------------------------
_OVERLAY_ROOT = Path(__file__).resolve().parent / "overlays"
_TEMPLATE_SPEC = {"ruler": ("rulers", "m*.png"), "arrow": ("arrows", "a*.png")}
_TEMPLATE_CACHE: dict[str, list[Image.Image]] = {}


def _templates(family: str) -> list[Image.Image]:
    """Load & cache the RGBA overlay templates for a family, sorted for determinism."""
    if family not in _TEMPLATE_CACHE:
        subdir, pattern = _TEMPLATE_SPEC[family]
        d = _OVERLAY_ROOT / subdir
        paths = sorted(d.glob(pattern))
        if not paths:
            raise FileNotFoundError(
                f"no {family!r} overlay templates under {d} (pattern {pattern!r}); bundle "
                "the PNGs in src/carve/data/overlays/ — see docs/EXECUTION_PLAN.md Phase 1"
            )
        _TEMPLATE_CACHE[family] = [Image.open(p).convert("RGBA") for p in paths]
    return _TEMPLATE_CACHE[family]


def _placed_layer(overlay: Image.Image, h: int, w: int, px: int, py: int) -> Image.Image:
    """Paste an RGBA overlay (its own alpha preserved) at (px,py) onto a transparent HxW canvas.
    ``paste`` (no mask) copies all four bands and clips to the canvas, so partial-off-canvas
    placement is fine."""
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    layer.paste(overlay, (int(px), int(py)))
    return layer


def _ruler_layer(h: int, w: int, rng: np.random.Generator) -> Image.Image:
    """A dermoscopy ruler overlay: sized 60–100% of the image's longer side, placed on an
    orbit around the centre and oriented ~tangentially so it *revolves around* (and never
    crosses through) the central lesion. It may hang partly off-canvas, as real rulers do.
    Adapted from projects/masks-rulers MedicalOverlayTransform, RNG-driven for determinism."""
    tpl = _templates("ruler")
    ruler = tpl[int(rng.integers(len(tpl)))]
    longer = max(h, w)
    # ruler spans 60–100% of the image's longer side (was 0.4–0.8; too small/subtle per image)
    target = max(1, int(longer * float(rng.uniform(0.6, 1.0))))
    rw, rh = ruler.size
    if rw >= rh:
        new_w, new_h = target, max(1, round(rh * target / rw))
    else:
        new_h, new_w = target, max(1, round(rw * target / rh))
    # Position the ruler on an orbit at angle theta around the image centre, and rotate it so
    # its long axis is ~tangent to that radius (perpendicular to the spoke). A tangential ruler
    # sitting at radius >= the central disk keeps its whole span out of the middle — so it wraps
    # around the lesion instead of covering it. Templates are horizontal, so tangent = theta+90°.
    theta = float(rng.uniform(0.0, 2.0 * math.pi))
    # Orient the (horizontal) ruler tangent to the radius at angle theta. Image arrays are
    # y-DOWN, so PIL's counter-clockwise rotate is clockwise in array space: the tangent angle
    # is -(theta+90), NOT +(theta+90). (The + form is only tangential at theta=0/90 and becomes
    # radial — pointing straight at the centre — at theta=45/135, which pulled diagonal rulers
    # into the lesion.) Small ±10° jitter keeps it near-tangent without aiming inward.
    tangent_deg = -(math.degrees(theta) + 90.0) + float(rng.uniform(-10.0, 10.0))
    over = ruler.resize((new_w, new_h), Image.LANCZOS).rotate(
        tangent_deg, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0)
    )
    ow, oh = over.size
    # Orbit radius of the ruler's midpoint. Now that placement is truly tangential the ruler
    # stays ~radius·cos(jitter) from the centre for every theta, so 0.40–0.48·min(H,W) keeps it
    # clear of even a large lesion while its midpoint stays near the edge (hangs ~1/3 off).
    radius = float(rng.uniform(0.40, 0.48)) * min(h, w)
    cx = w / 2.0 + radius * math.cos(theta)
    cy = h / 2.0 + radius * math.sin(theta)
    return _placed_layer(over, h, w, round(cx - ow / 2.0), round(cy - oh / 2.0))


def _arrow_layer(h: int, w: int, rng: np.random.Generator) -> Image.Image:
    """An annotation arrow overlay placed off-centre and rotated to point at the lesion centre.
    Adapted from projects/masks-rulers MedicalOverlayTransform, RNG-driven for determinism."""
    tpl = _templates("arrow")
    arrow = tpl[int(rng.integers(len(tpl)))]
    longer = max(h, w)
    target = max(1, int(longer * float(rng.uniform(0.15, 0.3))))
    arrow = arrow.resize((target, target), Image.LANCZOS)
    cx, cy = w / 2.0, h / 2.0
    radius = float(rng.uniform(0.4 * longer / 2.0, 0.6 * longer / 2.0))
    ang = float(rng.uniform(0, 2 * math.pi))
    ax, ay = cx + radius * math.cos(ang), cy + radius * math.sin(ang)
    dx, dy = cx - ax, cy - ay
    rot = -math.degrees(math.atan2(dx, -dy))  # arrow's tip turns toward the centre
    arrow = arrow.rotate(rot, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0))
    aw, ah = arrow.size
    return _placed_layer(arrow, h, w, round(ax - aw / 2.0), round(ay - ah / 2.0))


_LAYER_FN = {"ruler": _ruler_layer, "arrow": _arrow_layer}


def _overlay_builder(*families: str):
    """Build a (mask, color) builder that layers one or more real overlays onto a canvas.

    color is a full HxWx3 image (the overlay's RGB); mask is its per-pixel alpha coverage.
    Guarantees a non-empty footprint (retries centred) so mask.sum() > 0 as the tests require.
    """

    def build(h: int, w: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        layer = _LAYER_FN[families[0]](h, w, rng)
        for fam in families[1:]:
            layer = Image.alpha_composite(layer, _LAYER_FN[fam](h, w, rng))
        arr = np.asarray(layer, dtype=np.float32) / 255.0  # HxWx4
        mask = np.ascontiguousarray(arr[..., 3])
        if mask.sum() == 0:  # overlay fell fully off-canvas (tiny canvases): drop it dead-centre
            fam = families[0]
            over = _templates(fam)[int(rng.integers(len(_templates(fam))))]
            side = max(1, min(h, w) // 2)
            over = over.resize((side, side), Image.LANCZOS)
            layer = _placed_layer(over, h, w, (w - side) // 2, (h - side) // 2)
            arr = np.asarray(layer, dtype=np.float32) / 255.0
            mask = np.ascontiguousarray(arr[..., 3])
        color = np.ascontiguousarray(arr[..., :3])
        return mask, color

    return build


_BUILDERS = {
    # DEFAULT: real photographic overlays
    "ruler": _overlay_builder("ruler"),
    "arrow": _overlay_builder("arrow"),
    "overlay_both": _overlay_builder("ruler", "arrow"),
    # LEGACY: original synthetic marks (opt-in)
    "ruler_synthetic": _ruler,
    "marker_ink": _marker_ink,
    "dark_corner": _dark_corner,
    "black_corner": _black_corner,
}
# every registered kind (real defaults + composite + synthetic legacy)
ALL_ARTIFACT_KINDS = list(_BUILDERS)


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
        raise ValueError(f"unknown artifact kind {kind!r}; expected one of {ALL_ARTIFACT_KINDS}")
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
