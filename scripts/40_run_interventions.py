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
from carve.data.datasets import make_splits  # noqa: E402
from carve.data.isic import load_isic_binary  # noqa: E402
from carve.utils import assert_disjoint, load_config, new_run_dir, save_json, set_seed  # noqa: E402

FULL = dict(sae_train=1200, select=300, eval=250, width=16384, k=32, steps=3000)
QUICK = dict(sae_train=400, select=200, eval=120, width=4096, k=32, steps=800)
ARTIFACTS = ["ruler", "marker_ink", "dark_corner"]
RHO, ALPHA = 0.9, 1.0
STEER_COEFFS = [2.0, 4.0, 8.0]   # subtract c·decoder dir (recovery direction)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    N = QUICK if args.quick else FULL

    print("=" * 72)
    print("CARVE — Phase-5 grid: SAE-feature interventions & causal recovery")
    print("=" * 72)

    cfg = load_config(str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    run = new_run_dir(cfg.paths.runs_dir, "interventions_grid", cfg, int(cfg.get("seed", 0)))
    size = int(cfg.dataset.image_size)
    layer = int(cfg.sae.get("layer", 12))
    cfg.sae.width, cfg.sae.k, cfg.sae.train.steps = N["width"], N["k"], N["steps"]
    eps = 1e-3
    seeds = [0] if args.quick else list(cfg.get("seeds", [0, 1, 2]))

    from carve.eval.aggregate import benchmark_table
    from carve.eval.harness import run_cell
    from carve.models.encoders import load_encoder
    from carve.sae.discovery import select_oracle
    from carve.sae.train_sae import sae_health, train_sae

    enc = load_encoder(cfg)
    ds = load_isic_binary(cfg, image_size=size)
    print(f"[grid] seeds={seeds}  artifacts={ARTIFACTS}  ρ={RHO} α={ALPHA}  "
          f"layer {layer}  width {N['width']}")

    all_records, health_by_seed = [], {}
    for seed in seeds:
        set_seed(seed)
        splits = make_splits(ds.labels, dict(cfg.dataset.splits), seed=seed, stratify=True)
        assert_disjoint(**{k: set(v.tolist()) for k, v in splits.items()})
        sae_idx = splits["sae_train"][: N["sae_train"]]
        sel_idx = splits["select"][: N["select"]]
        ev_idx = splits["eval"][: N["eval"]]

        # SAE trained ONCE per seed (artifact-agnostic; injected artifacts appear only in
        # select/eval, never in sae_train) → reused across all artifacts.
        print(f"\n[seed {seed}] training SAE ...")
        acts = enc.activations([ds.load_image(int(i), size=size) for i in sae_idx], layer, pool=None)
        acts = acts.reshape(-1, acts.shape[-1])
        sae = train_sae(acts, cfg, seed=seed)
        health = sae_health(sae, acts[int(0.8 * len(acts)):])
        health_by_seed[seed] = health
        print(f"[seed {seed}] SAE R²={health['r2']:.3f}  dead={health['dead_feature_frac']*100:.1f}%")

        sel_imgs = [ds.load_image(int(i), size=size) for i in sel_idx]
        clean = [ds.load_image(int(i), size=size) for i in ev_idx]
        labels = ds.labels[ev_idx]

        for artifact in ARTIFACTS:
            items, _ = make_biased_set(sel_imgs, ds.labels[sel_idx], artifact, rho=RHO, alpha=ALPHA, seed=seed + 1)
            oracle = select_oracle(sae, enc, layer, items, top_m=int(cfg.sae.feature_set_size_m))
            S, det = oracle["features"], oracle["best_auroc"]
            x_art = [inject(im, artifact, ALPHA, np.random.default_rng(int(i)))[0]
                     for im, i in zip(clean, ev_idx)]
            base = dict(model="monet", artifact=artifact, rho=RHO, opacity=ALPHA)
            common = dict(labels=labels, recovery_eps=eps, bootstrap=int(cfg.eval.bootstrap_resamples),
                          ci=float(cfg.eval.ci), seed=seed, run_dir=run)

            cells = [run_cell({**base, "selection": "oracle", "method": "sae_ablate"},
                              enc, layer, sae, S, x_art, clean, op="ablate", detection_auroc=det, **common)]
            for c in STEER_COEFFS:
                cells.append(run_cell({**base, "selection": "oracle", "method": f"sae_steer_c{c}"},
                                      enc, layer, sae, S, x_art, clean, op="steer", coeff=c,
                                      detection_auroc=det, **common))
            rng = np.random.default_rng(seed)
            S_rand = [int(rng.integers(0, sae.width)) for _ in S]
            cells.append(run_cell({**base, "selection": "random", "method": "sae_ablate"},
                                  enc, layer, sae, S_rand, x_art, clean, op="ablate", **common))
            all_records.extend(cells)
            print(f"  [{artifact:11s}] det AUROC {det:.3f} | ablate R={cells[0]['R_median']:+.3f} "
                  f"sel {cells[0]['selectivity']:.2f} | best steer R="
                  f"{max(c['R_median'] for c in cells[1:-1]):+.3f}")

    save_json(run / "metrics.json", {"health_by_seed": health_by_seed, "records": all_records})

    # ---- mean±std over seeds -------------------------------------------------------------
    tbl = benchmark_table(run)
    cols = [c for c in ["artifact", "selection", "method", "detection_auroc_mean",
                        "R_median_mean", "R_median_std", "selectivity_mean", "off_target_mean"]
            if c in tbl.columns]
    print("\n=== benchmark table (mean±std over seeds) ===")
    print(tbl[cols].to_string(index=False))
    print(f"\n[run] {run}")


if __name__ == "__main__":
    main()
