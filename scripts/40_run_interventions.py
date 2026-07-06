#!/usr/bin/env python
"""Phase-5: the core CARVE result — does intervening on the SAE artifact feature RECOVER
the input-level effect? Trains a MONET SAE, selects the oracle artifact feature, then on a
DISJOINT eval split computes causal recovery / selectivity / off-target for:
  - ablate(oracle)          (zero the feature's reconstructed contribution)
  - steer(oracle, c)        (subtract c · decoder direction) over a coefficient sweep
  - ablate(random)          (control — should do ~nothing)
All via carve.eval.harness.run_cell (writes per-image parquet + cell json to the run dir).

    docker exec carve-dev python3 scripts/40_run_interventions.py --quick
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

FULL = dict(sae_train=1500, select=400, eval=300, width=16384, k=32, steps=3000)
QUICK = dict(sae_train=400, select=200, eval=150, width=4096, k=32, steps=800)
ARTIFACT, RHO, ALPHA = "ruler", 0.9, 1.0
STEER_COEFFS = [1.0, 2.0, 4.0, 8.0]   # subtract c·decoder dir (recovery direction)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    N = QUICK if args.quick else FULL

    print("=" * 72)
    print("CARVE — Phase-5: SAE-feature interventions & causal recovery")
    print("=" * 72)

    cfg = load_config(str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    run = new_run_dir(cfg.paths.runs_dir, "interventions", cfg, seed)
    size = int(cfg.dataset.image_size)
    layer = int(cfg.sae.get("layer", 12))
    cfg.sae.width, cfg.sae.k, cfg.sae.train.steps = N["width"], N["k"], N["steps"]
    eps = float(cfg.interventions.recovery_eps)

    from carve.eval.harness import run_cell
    from carve.models.encoders import load_encoder
    from carve.sae.discovery import select_oracle
    from carve.sae.train_sae import sae_health, train_sae

    enc = load_encoder(cfg)
    ds = load_isic_binary(cfg, image_size=size)
    splits = make_splits(ds.labels, dict(cfg.dataset.splits), seed=seed, stratify=True)
    assert_disjoint(**{k: set(v.tolist()) for k, v in splits.items()})
    sae_idx = splits["sae_train"][: N["sae_train"]]
    sel_idx = splits["select"][: N["select"]]
    ev_idx = splits["eval"][: N["eval"]]

    # ---- train SAE ---------------------------------------------------------------------
    print(f"[sae] layer {layer}, width {N['width']}, k {N['k']} — extracting + training ...")
    acts = enc.activations([ds.load_image(int(i), size=size) for i in sae_idx], layer, pool=None)
    acts = acts.reshape(-1, acts.shape[-1])
    sae = train_sae(acts, cfg, seed=seed)
    health = sae_health(sae, acts[int(0.8 * len(acts)):])
    print(f"[sae] R²={health['r2']:.3f}  dead={health['dead_feature_frac']*100:.1f}%")

    # ---- select oracle feature on `select` (disjoint from eval) ------------------------
    sel_imgs = [ds.load_image(int(i), size=size) for i in sel_idx]
    items, _ = make_biased_set(sel_imgs, ds.labels[sel_idx], ARTIFACT, rho=RHO, alpha=ALPHA, seed=seed + 1)
    oracle = select_oracle(sae, enc, layer, items, top_m=int(cfg.sae.feature_set_size_m))
    S, det_auroc = oracle["features"], oracle["best_auroc"]
    print(f"[oracle] feature {S}  detection AUROC {det_auroc:.3f}")

    # ---- eval set (disjoint): inject the artifact on all eval images; clean = source ---
    x_clean = [ds.load_image(int(i), size=size) for i in ev_idx]
    x_art = [inject(im, ARTIFACT, ALPHA, np.random.default_rng(int(i)))[0] for im, i in zip(x_clean, ev_idx)]
    labels = ds.labels[ev_idx]
    base = dict(model="monet", artifact=ARTIFACT, rho=RHO, opacity=ALPHA)
    common = dict(labels=labels, detection_auroc=det_auroc, recovery_eps=eps,
                  bootstrap=int(cfg.eval.bootstrap_resamples), ci=float(cfg.eval.ci),
                  seed=seed, run_dir=run)

    records = []
    # ablate(oracle)
    records.append(run_cell({**base, "selection": "oracle", "method": "sae_ablate"},
                            enc, layer, sae, S, x_art, x_clean, op="ablate", **common))
    # steer(oracle, c) sweep
    for c in STEER_COEFFS:
        records.append(run_cell({**base, "selection": "oracle", "method": f"sae_steer_c{c}"},
                                enc, layer, sae, S, x_art, x_clean, op="steer", coeff=c, **common))
    # ablate(random) control
    rng = np.random.default_rng(seed)
    S_rand = [int(rng.integers(0, sae.width)) for _ in S]
    records.append(run_cell({**base, "selection": "random", "method": "sae_ablate"},
                            enc, layer, sae, S_rand, x_art, x_clean, op="ablate",
                            **{**common, "detection_auroc": None}))

    save_json(run / "metrics.json", {"sae_health": health, "oracle": oracle, "cells": records})

    print(f"\n  {'method':16s} {'sel':7s} {'R_med':>7s} {'R_CI':>16s} {'cause':>7s} "
          f"{'isol':>7s} {'select':>7s} {'offtgt':>7s}")
    for r in records:
        print(f"  {r['method']:16s} {r['selection']:7s} {r['R_median']:+7.3f} "
              f"[{r['R_ci_lo']:+.2f},{r['R_ci_hi']:+.2f}] {r['cause']:7.3f} {r['isolation']:7.3f} "
              f"{r['selectivity']:7.3f} {r['off_target']:+7.3f}")
    print(f"\n[finding] detection AUROC {det_auroc:.3f} vs ablation recovery "
          f"{records[0]['R_median']:+.3f} — detection≠control if recovery is low.")
    print(f"[run] {run}")


if __name__ == "__main__":
    main()
