#!/usr/bin/env python
"""Eyeball the REAL photographic overlays (rulers + arrows) on real ISIC images.

Companion to scripts/20_make_artifact_examples.py, but for the new default artifact family:
the real ruler (m*.png) and arrow (a*.png) PNG overlays bundled under
src/carve/data/overlays/, composited by carve.data.artifacts.inject exactly as the pipeline
uses them (same 224 canvas, same rng=default_rng(index) seed convention).

For each sampled image it writes, under experiments/overlay_examples/<isic_id>/:
    original.png                         the clean image (model input, 224x224)
    ruler.png / arrow.png / both.png     α=1.0 (opaque)
    <kind>_a0.6.png                      α=0.6 (semi-transparent example)
    _mask_<kind>.png                     the overlay footprint (α coverage)
Plus contact_sheet.png — a grid across all sampled images for a quick look.

experiments/overlay_examples/ is git-ignored (derived ISIC/HAM10000 patient-photo pixels —
regenerate locally, do not redistribute; see docs/INTEGRITY.md and .gitignore).

    docker exec <container> python3 scripts/21_overlay_examples.py
    # or:  ./docker-run.sh python3 scripts/21_overlay_examples.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from carve.data.artifacts import inject  # noqa: E402
from carve.data.isic import load_isic_binary  # noqa: E402
from carve.utils import load_config  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "experiments" / "overlay_examples"
SIZE = 224
N_IMAGES = 6
# columns: the real-overlay default set + the layered combination
KINDS = ["ruler", "arrow", "overlay_both"]
TITLES = {"ruler": "ruler", "arrow": "arrow", "overlay_both": "both"}


def _save(arr: np.ndarray, path: Path) -> None:
    Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype("uint8")).save(path)


def main() -> None:
    cfg = load_config(str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    ds = load_isic_binary(cfg, image_size=SIZE)

    # deterministic mix of melanoma + nevus
    rng = np.random.default_rng(0)
    mel = rng.choice(np.where(ds.labels == 1)[0], size=N_IMAGES // 2, replace=False)
    nev = rng.choice(np.where(ds.labels == 0)[0], size=N_IMAGES - N_IMAGES // 2, replace=False)
    idx = np.concatenate([mel, nev])

    OUT.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = 1 + len(KINDS)
    fig, axes = plt.subplots(len(idx), ncols, figsize=(2.2 * ncols, 2.2 * len(idx)))
    axes = np.atleast_2d(axes)

    for r, i in enumerate(idx):
        iid = ds.ids[int(i)]
        label = "melanoma" if ds.labels[int(i)] == 1 else "nevus"
        base = ds.load_image(int(i), size=SIZE)  # float32 HxWx3 in [0,1]
        d = OUT / iid
        d.mkdir(exist_ok=True)
        _save(base, d / "original.png")

        axes[r, 0].imshow(base)
        axes[r, 0].set_ylabel(f"{iid}\n({label})", fontsize=8)
        axes[r, 0].set_title("original" if r == 0 else "")

        for c, kind in enumerate(KINDS, start=1):
            # same seed convention as the injector's gate arm: rng = default_rng(image index)
            art, mask = inject(base, kind, 1.0, np.random.default_rng(int(i)))
            soft, _ = inject(base, kind, 0.6, np.random.default_rng(int(i)))
            stem = TITLES[kind]
            _save(art, d / f"{stem}.png")
            _save(soft, d / f"{stem}_a0.6.png")
            _save(np.stack([mask] * 3, -1), d / f"_mask_{stem}.png")
            axes[r, c].imshow(art)
            axes[r, c].set_title(TITLES[kind] if r == 0 else "")

        for c in range(ncols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    fig.suptitle("CARVE — real ruler / arrow overlays on ISIC images (α=1.0)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "contact_sheet.png", dpi=120, bbox_inches="tight")
    print(f"[ok] wrote {len(idx)} example folders + contact_sheet.png under:\n     {OUT}")
    for i in idx:
        print(f"       - {ds.ids[int(i)]}  ({'melanoma' if ds.labels[int(i)] else 'nevus'})")


if __name__ == "__main__":
    main()
