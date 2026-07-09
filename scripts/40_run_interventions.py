#!/usr/bin/env python
"""Phase-5: the core CARVE result — does intervening on the SAE artifact feature RECOVER
the input-level effect? For each seed and each artifact, trains a MONET SAE (once per seed),
selects the oracle artifact feature on `select`, then on the DISJOINT `eval` split computes
causal recovery / selectivity / off-target for ablate(oracle), a steer(oracle,c) sweep, and
an ablate(random) control. All via carve.eval.harness.run_cell (per-image parquet + cell
json). Ends with the mean±std-over-seeds benchmark table.

    docker exec carve-dev python3 scripts/40_run_interventions.py           # 3-seed grid
    docker exec carve-dev python3 scripts/40_run_interventions.py --quick   # 1 seed, small
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from carve.data.artifacts import inject, make_biased_set  # noqa: E402
from carve.data.isic import load_isic_binary  # noqa: E402
from carve.eval.grid import finish_grid, iter_seed_contexts, resolve_seeds, setup_run  # noqa: E402
from carve.utils import load_config  # noqa: E402

FULL = dict(sae_train=1200, select=300, eval=250, width=16384, k=32, steps=3000)
QUICK = dict(sae_train=400, select=200, eval=120, width=4096, k=32, steps=800)
# ARTIFACTS is config-driven (cfg.artifacts.types); resolved in main() — real overlays by default
RHO, ALPHA = 0.9, 1.0
# #1 feature-set-size sweep + #2 best-case steering grid are config-driven (interventions.*)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    N = QUICK if args.quick else FULL

    print("=" * 72)
    print("CARVE — Phase-5 grid: SAE-feature interventions & causal recovery")
    print("=" * 72)

    cfg = load_config(str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    ARTIFACTS = list(cfg.artifacts.types)   # config-driven; default = real ruler + arrow overlays
    M_SWEEP = [int(m) for m in cfg.interventions.feature_set_sizes]   # #1 ablation m-sweep
    STEER_GRID = [float(c) for c in cfg.interventions.steer_grid]     # #2 best-case steer search
    run, layer, size = setup_run(cfg, "interventions_grid", N)
    eps = float(cfg.interventions.recovery_eps)   # config-driven (was hardcoded); PREREG-frozen
    seeds = resolve_seeds(cfg, args.quick)

    from carve.eval.harness import run_cell, run_steer_bestcase
    from carve.models.encoders import load_encoder
    from carve.sae.discovery import select_oracle

    enc = load_encoder(cfg)
    ds = load_isic_binary(cfg, image_size=size)
    print(f"[grid] seeds={seeds}  artifacts={ARTIFACTS}  ρ={RHO} α={ALPHA}  "
          f"layer {layer}  width {N['width']}")

    all_records, health_by_seed = [], {}
    for ctx in iter_seed_contexts(cfg, ds, enc, N, seeds, layer, size):
        seed, sae = ctx.seed, ctx.sae
        health_by_seed[seed] = ctx.health
        sel_imgs, clean, labels, ev_idx = ctx.sel_imgs, ctx.clean, ctx.labels, ctx.ev_idx

        for artifact in ARTIFACTS:
            items, _ = make_biased_set(sel_imgs, ctx.sel_labels, artifact, rho=RHO, alpha=ALPHA, seed=seed + 1)
            # top-m oracle features (nested: S_all[:m] is the top-m set); det = best single-feature AUROC
            oracle = select_oracle(sae, enc, layer, items, top_m=max(M_SWEEP))
            S_all, det = oracle["features"], oracle["best_auroc"]
            x_art = [inject(im, artifact, ALPHA, np.random.default_rng(int(i)))[0]
                     for im, i in zip(clean, ev_idx)]
            base = dict(model="monet", artifact=artifact, rho=RHO, opacity=ALPHA)
            common = dict(labels=labels, recovery_eps=eps, bootstrap=int(cfg.eval.bootstrap_resamples),
                          ci=float(cfg.eval.ci), seed=seed, run_dir=run)

            cells = []
            # #1 feature-set-size sweep: ablate the top-m oracle features together
            for m in M_SWEEP:
                cells.append(run_cell({**base, "selection": "oracle", "method": f"sae_ablate_m{m}"},
                                      enc, layer, sae, S_all[:m], x_art, clean, op="ablate",
                                      detection_auroc=det, **common))
            # #2 best-case steering of the single oracle feature: per-coeff curve + per-image ceiling
            cells += run_steer_bestcase({**base, "selection": "oracle"}, enc, layer, sae, S_all[:1],
                                        x_art, clean, STEER_GRID, detection_auroc=det, **common)
            # random-feature ablation control (single feature, matches m=1)
            rng = np.random.default_rng(seed)
            S_rand = [int(rng.integers(0, sae.width))]
            cells.append(run_cell({**base, "selection": "random", "method": "sae_ablate"},
                                  enc, layer, sae, S_rand, x_art, clean, op="ablate", **common))
            all_records.extend(cells)
            r = {c["method"]: c["R_median"] for c in cells}
            print(f"  [{artifact:11s}] det AUROC {det:.3f} | ablate m1 R={r.get('sae_ablate_m1'):+.3f} "
                  f"m{max(M_SWEEP)} R={r.get(f'sae_ablate_m{max(M_SWEEP)}'):+.3f} | "
                  f"steer bestcase R={r.get('sae_steer_bestcase'):+.3f}")

    finish_grid(run, all_records, health_by_seed, title="benchmark table (mean±std over seeds)")


if __name__ == "__main__":
    main()
