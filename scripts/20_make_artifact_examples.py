#!/usr/bin/env python
"""Materialize real before/after artifact examples for eyeballing.

Uses the exact injector the Phase-0 gate uses (carve.data.artifacts.inject) on real
ISIC-2018 images, at the same size (224) and seed convention the gate used. For each
sampled image it writes, under experiments/artifact_examples/<isic_id>/:
    original.png              the clean image (model input, 224x224)
    ruler.png / marker_ink.png / dark_corner.png     α=1.0 (opaque; what the gate used)
    ruler_a0.6.png ...        α=0.6 (a graded, semi-transparent example)
    _mask_<kind>.png          the artifact footprint mask
Plus a contact_sheet.png overview across all sampled images.

    docker exec carve-dev python3 scripts/20_make_artifact_examples.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from carve.data.artifacts import ARTIFACT_KINDS, inject  # noqa: E402
from carve.data.isic import load_isic_binary  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "experiments" / "artifact_examples"
SIZE = 224
N_IMAGES = 6


def _save(arr: np.ndarray, path: Path):
    Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype("uint8")).save(path)


def main() -> None:
    ds = load_isic_binary(image_size=SIZE)
    # pick a mix of melanoma and nevus, deterministically
    rng = np.random.default_rng(0)
    mel = rng.choice(np.where(ds.labels == 1)[0], size=N_IMAGES // 2, replace=False)
    nev = rng.choice(np.where(ds.labels == 0)[0], size=N_IMAGES - N_IMAGES // 2, replace=False)
    idx = np.concatenate([mel, nev])

    OUT.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = 1 + len(ARTIFACT_KINDS)
    fig, axes = plt.subplots(len(idx), ncols, figsize=(2.2 * ncols, 2.2 * len(idx)))
    axes = np.atleast_2d(axes)

    for r, i in enumerate(idx):
        iid = ds.ids[int(i)]
        label = "melanoma" if ds.labels[int(i)] == 1 else "nevus"
        base = ds.load_image(int(i), size=SIZE)  # float32 HxWx3 [0,1]
        d = OUT / iid
        d.mkdir(exist_ok=True)
        _save(base, d / "original.png")

        axes[r, 0].imshow(base)
        axes[r, 0].set_ylabel(f"{iid}\n({label})", fontsize=8)
        axes[r, 0].set_title("original" if r == 0 else "")

        for c, kind in enumerate(ARTIFACT_KINDS, start=1):
            # same seed convention as the gate's zero-shot arm: rng = default_rng(index)
            art, mask = inject(base, kind, 1.0, np.random.default_rng(int(i)))
            soft, _ = inject(base, kind, 0.6, np.random.default_rng(int(i)))
            _save(art, d / f"{kind}.png")
            _save(soft, d / f"{kind}_a0.6.png")
            _save(np.stack([mask] * 3, -1), d / f"_mask_{kind}.png")
            axes[r, c].imshow(art)
            axes[r, c].set_title(kind if r == 0 else "")

        for c in range(ncols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    fig.suptitle("CARVE — real ISIC images, artifacts injected (α=1.0)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "contact_sheet.png", dpi=120, bbox_inches="tight")
    print(f"[ok] wrote {len(idx)} example folders + contact_sheet.png under:\n     {OUT}")
    for i in idx:
        print(f"       - {ds.ids[int(i)]}  ({'melanoma' if ds.labels[int(i)] else 'nevus'})")


if __name__ == "__main__":
    main()
