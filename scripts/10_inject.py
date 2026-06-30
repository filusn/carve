#!/usr/bin/env python
"""Phase-1 visual sanity check for the artifact injector.

Runs with ZERO data: it generates synthetic "skin" images (skin-toned background + a darker
elliptical lesion), injects every artifact kind across a sweep of opacities, and writes a
contact-sheet PNG plus the gold counterfactual (``remove``). Use it to eyeball that the
injected artifacts look plausible before wiring in real dermatology data.

    python scripts/10_inject.py            # writes a PNG under experiments/runs/<ts>_inject_demo/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# allow running from the repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carve.data.artifacts import ARTIFACT_KINDS, inject, remove  # noqa: E402
from carve.utils import load_config, new_run_dir  # noqa: E402

ALPHAS = (0.4, 0.7, 1.0)


def synthetic_skin(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    base = np.array([0.85, 0.66, 0.56], np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = 0.06 * np.sin(xx / w * np.pi) + 0.04 * (yy / h)
    img = base[None, None, :] + grad[..., None] * 0.12
    img = img + rng.normal(0, 0.012, (h, w, 3)).astype(np.float32)
    cy, cx, ry, rx = h * 0.5, w * 0.5, h * 0.24, w * 0.24
    ell = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
    lesion = np.array([0.32, 0.19, 0.17], np.float32)
    img[ell] = lesion + rng.normal(0, 0.02, (int(ell.sum()), 3)).astype(np.float32)
    return np.clip(img, 0, 1).astype(np.float32)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    cfg = load_config(str(cfg_path))
    seed = int(cfg.get("seed", 0))
    run = new_run_dir(cfg.paths.runs_dir, "inject_demo", cfg, seed)

    rng = np.random.default_rng(seed)
    base = synthetic_skin(160, 160, rng)

    ncols = 1 + len(ALPHAS) + 1  # clean | alphas... | removed(@1.0)
    nrows = len(ARTIFACT_KINDS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.1 * ncols, 2.1 * nrows))
    axes = np.atleast_2d(axes)

    for r, kind in enumerate(ARTIFACT_KINDS):
        axes[r, 0].imshow(base)
        axes[r, 0].set_title("clean" if r == 0 else "")
        axes[r, 0].set_ylabel(kind, fontsize=11)
        for c, a in enumerate(ALPHAS, start=1):
            art, _ = inject(base, kind, a, np.random.default_rng(seed + 99))
            axes[r, c].imshow(art)
            axes[r, c].set_title(f"α={a}" if r == 0 else "")
        art, mask = inject(base, kind, 1.0, np.random.default_rng(seed + 99))
        rec = remove(art, mask, source=base)
        axes[r, -1].imshow(rec)
        axes[r, -1].set_title("remove()" if r == 0 else "")
        for c in range(ncols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    fig.suptitle("CARVE — synthetic artifact injection (Phase 1)", fontsize=13)
    fig.tight_layout()
    out = run / "inject_demo.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"[ok] wrote {out}")
    print(f"[ok] run dir: {run}")


if __name__ == "__main__":
    main()
