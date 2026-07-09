#!/usr/bin/env python
"""Phase-7 FINAL grid: the pre-registered ρ×α sweep (PREREGISTRATION §1: "final experiment
grid (Phase 5–7 ρ×α sweep)"). Shows the detection≠control dissociation holds across the WHOLE
(spurious-correlation strength ρ) × (opacity α) grid, not just the single ρ=0.9/α=1.0 point.

Runs at the PREREG §4 rule-compliant SAE width: the widest dictionary with ≤15% dead features,
which is 4096 (~9.6% dead). 16384 exceeds that bar (~22% dead) so it is EXPLORATORY only (the
§5 single-point grids); this confirmatory sweep uses the frozen-compliant 4096.

The SAE trains ONCE per seed on CLEAN activations (independent of ρ,α), then for every
(artifact, ρ, α) cell we record: detection AUROC (oracle feature on `select`), SAE-ablate
recovery R + selectivity (on the disjoint `eval`), and the input-removal oracle ceiling (R≡1).
Emits per-cell RunRecords + a per-artifact (ρ×α) heatmap of detection and recovery.

    docker exec carve-dev python3 scripts/41_rho_alpha_sweep.py           # full grid
    docker exec carve-dev python3 scripts/41_rho_alpha_sweep.py --quick   # 1 seed, tiny grid
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from carve.data.artifacts import inject, make_biased_set  # noqa: E402
from carve.data.isic import load_isic_binary  # noqa: E402
from carve.eval.grid import finish_grid, iter_seed_contexts, resolve_seeds, setup_run  # noqa: E402
from carve.utils import load_config  # noqa: E402

# Frozen, health-rule-compliant width (≤15% dead ⇒ 4096). NOT the exploratory 16384.
FULL = dict(sae_train=1200, select=300, eval=250, width=4096, k=32, steps=3000)
QUICK = dict(sae_train=400, select=200, eval=120, width=4096, k=32, steps=800)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    N = QUICK if args.quick else FULL

    print("=" * 72)
    print("CARVE — Phase-7 FINAL grid: ρ×α sweep (detection≠control across the whole grid)")
    print("=" * 72)

    cfg = load_config(str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    ARTIFACTS = list(cfg.artifacts.types)
    # ρ×α grids are config-driven (PREREG-frozen); quick uses a 2×1 corner for a fast harness check
    RHOS = [1.0, 0.9] if args.quick else [float(r) for r in cfg.artifacts.rho]      # [0.5,0.7,0.9,1.0]
    ALPHAS = [1.0] if args.quick else [float(a) for a in cfg.artifacts.opacity]      # [0.4,0.7,1.0]
    run, layer, size = setup_run(cfg, "rho_alpha_sweep", N)
    eps = float(cfg.interventions.recovery_eps)
    seeds = resolve_seeds(cfg, args.quick)

    from carve.eval.harness import run_cell
    from carve.models.encoders import load_encoder
    from carve.sae.discovery import select_oracle

    enc = load_encoder(cfg)
    ds = load_isic_binary(cfg, image_size=size)
    print(f"[sweep] seeds={seeds} artifacts={ARTIFACTS} ρ={RHOS} α={ALPHAS}  layer {layer}  "
          f"width {N['width']} (frozen-compliant)")

    all_records, health_by_seed = [], {}
    for ctx in iter_seed_contexts(cfg, ds, enc, N, seeds, layer, size):
        seed, sae = ctx.seed, ctx.sae
        health_by_seed[seed] = ctx.health
        for artifact in ARTIFACTS:
            for rho, alpha in itertools.product(RHOS, ALPHAS):
                items, _ = make_biased_set(ctx.sel_imgs, ctx.sel_labels, artifact,
                                           rho=rho, alpha=alpha, seed=seed + 1)
                oracle = select_oracle(sae, enc, layer, items, top_m=1)
                S, det = oracle["features"], oracle["best_auroc"]
                x_art = [inject(im, artifact, alpha, np.random.default_rng(int(i)))[0]
                         for im, i in zip(ctx.clean, ctx.ev_idx)]
                base = dict(model="monet", artifact=artifact, rho=rho, opacity=alpha)
                common = dict(labels=ctx.labels, recovery_eps=eps,
                              bootstrap=int(cfg.eval.bootstrap_resamples),
                              ci=float(cfg.eval.ci), seed=seed, run_dir=run)
                all_records.append(run_cell({**base, "selection": "oracle", "method": "sae_ablate"},
                                   enc, layer, sae, S, x_art, ctx.clean, op="ablate",
                                   detection_auroc=det, **common))
                all_records.append(run_cell({**base, "selection": "oracle", "method": "input_oracle"},
                                   enc, layer, sae, S, x_art, ctx.clean, oracle=True, **common))
            print(f"  [{artifact:12s}] seed {seed}: swept {len(RHOS)}×{len(ALPHAS)} (ρ,α) cells")

    finish_grid(run, all_records, health_by_seed, title="ρ×α sweep (mean±std over seeds)")
    _heatmaps(run, ARTIFACTS, RHOS, ALPHAS)


def _heatmaps(run, artifacts, rhos, alphas) -> None:
    """Per-artifact (ρ×α) heatmaps: detection AUROC (should stay high) beside SAE recovery R
    (should stay ≈0). The visual proof that the dissociation is grid-wide, not point-wise."""
    import glob
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame([json.loads(open(p).read()) for p in glob.glob(str(run) + "/cell_*.json")])
    sae = df[df.method == "sae_ablate"]
    ry, ax_a = sorted(rhos, reverse=True), sorted(alphas)
    panels = [("detection_auroc", "detection AUROC", 0.5, 1.0, "viridis"),
              ("R_median", "SAE recovery R", -1.0, 1.0, "RdBu")]
    fig, axes = plt.subplots(2, len(artifacts), figsize=(3.3 * len(artifacts), 6.4), squeeze=False)
    for j, art in enumerate(artifacts):
        d = sae[sae.artifact == art]
        for row, (col, ttl, vmin, vmax, cmap) in enumerate(panels):
            M = d.pivot_table(index="rho", columns="opacity", values=col, aggfunc="mean")
            M = M.reindex(index=ry, columns=ax_a)
            axp = axes[row][j]
            im = axp.imshow(M.values, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
            axp.set_xticks(range(len(ax_a))); axp.set_xticklabels(ax_a)
            axp.set_yticks(range(len(ry))); axp.set_yticklabels(ry)
            axp.set_xlabel("α (opacity)"); axp.set_ylabel("ρ (correlation)")
            for (yy, xx), v in np.ndenumerate(M.values):
                if not np.isnan(v):
                    axp.text(xx, yy, f"{v:.2f}", ha="center", va="center", fontsize=7,
                             color="black")
            axp.set_title(f"{art}\n{ttl}" if row == 0 else ttl, fontsize=9)
            fig.colorbar(im, ax=axp, fraction=0.046, pad=0.04)
    fig.suptitle("Detection ≠ control across the ρ×α grid — SAE oracle-ablate "
                 "(detection stays high, recovery stays ≈0)", fontsize=11)
    fig.tight_layout()
    out = Path(run) / "figures"
    out.mkdir(exist_ok=True)
    p = out / "rho_alpha_dissociation.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
