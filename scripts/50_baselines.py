#!/usr/bin/env python
"""Phase-6: baselines on IDENTICAL ground truth. For each seed × artifact, on the SAME cells
as scripts/40 (same splits, same per-seed SAE, same eval x_art/clean), run every method
through the one carve.eval.harness.run_cell interface so the numbers are directly comparable:

  * sae_oracle_ablate  — our method: ablate the top-AUROC SAE feature(s).
  * raw_neuron         — ablate the most artifact-correlated RAW neuron(s), budget-matched to
                         the SAE (answers "what does the SAE add over raw neurons?").
  * dermfmzero         — suppress top-5 SAE features most activated by the artifact
                         (DermFM-Zero, arXiv 2602.10624 — the incumbent, its own recipe).
  * random_raw         — random raw neuron(s); sanity floor, must do ≈nothing.
  * input_remove       — remove the artifact at the input = achievable ceiling (R≡1).

Writes per-cell JSON + per-image parquet, then the mean±std-over-seeds benchmark table.

    docker exec carve-dev python3 scripts/50_baselines.py           # 3-seed grid
    docker exec carve-dev python3 scripts/50_baselines.py --quick   # 1 seed, small
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from carve.baselines import (  # noqa: E402
    cav_baseline_proj,
    cav_suppress_fn,
    dermfmzero_select,
    fit_cav,
    random_raw_neurons,
    raw_neuron_ablate_fn,
    raw_neuron_select,
)
from carve.data.artifacts import inject, make_biased_set  # noqa: E402
from carve.data.isic import load_isic_binary  # noqa: E402
from carve.eval.grid import finish_grid, iter_seed_contexts, resolve_seeds, setup_run  # noqa: E402
from carve.utils import load_config  # noqa: E402

FULL = dict(sae_train=1200, select=300, eval=250, width=16384, k=32, steps=3000)
QUICK = dict(sae_train=400, select=200, eval=120, width=4096, k=32, steps=800)
ARTIFACTS = ["ruler", "marker_ink", "dark_corner"]
RHO, ALPHA = 0.9, 1.0
DERM_K = 5   # DermFM-Zero's published recipe: top-5 artifact-activated neurons


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    N = QUICK if args.quick else FULL

    print("=" * 72)
    print("CARVE — Phase-6 baselines on identical ground truth")
    print("=" * 72)

    cfg = load_config(str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    run, layer, size = setup_run(cfg, "baselines_grid", N)
    m = int(cfg.sae.feature_set_size_m)          # SAE feature-set size == matched raw budget
    eps = 1e-3
    seeds = resolve_seeds(cfg, args.quick)

    from carve.eval.harness import run_cell
    from carve.models.encoders import load_encoder
    from carve.sae.discovery import select_oracle

    enc = load_encoder(cfg)
    ds = load_isic_binary(cfg, image_size=size)
    print(f"[grid] seeds={seeds}  artifacts={ARTIFACTS}  ρ={RHO} α={ALPHA}  "
          f"layer {layer}  width {N['width']}  sae_m={m}  derm_k={DERM_K}")

    all_records, health_by_seed = [], {}
    for ctx in iter_seed_contexts(cfg, ds, enc, N, seeds, layer, size):
        seed, sae = ctx.seed, ctx.sae
        health_by_seed[seed] = ctx.health
        sel_imgs, clean, labels, ev_idx = ctx.sel_imgs, ctx.clean, ctx.labels, ctx.ev_idx
        baseline_vec, d = ctx.baseline_vec, ctx.d
        acts_mean = baseline_vec                               # clean reference for the CAV clamp
        rng = np.random.default_rng(seed)

        for artifact in ARTIFACTS:
            items, _ = make_biased_set(sel_imgs, ctx.sel_labels, artifact, rho=RHO, alpha=ALPHA, seed=seed + 1)
            x_art = [inject(im, artifact, ALPHA, np.random.default_rng(int(i)))[0]
                     for im, i in zip(clean, ev_idx)]
            base = dict(model="monet", artifact=artifact, rho=RHO, opacity=ALPHA)
            common = dict(labels=labels, recovery_eps=eps, bootstrap=int(cfg.eval.bootstrap_resamples),
                          ci=float(cfg.eval.ci), seed=seed, run_dir=run)

            # -- our method: SAE oracle ablate (top-AUROC feature[s]) ---------------------
            sae_or = select_oracle(sae, enc, layer, items, top_m=m)
            cells = [run_cell({**base, "selection": "oracle", "method": "sae_ablate"},
                              enc, layer, sae, sae_or["features"], x_art, clean, op="ablate",
                              detection_auroc=sae_or["best_auroc"], **common)]

            # -- raw-neuron ablation, budget-matched (m neurons) -------------------------
            raw = raw_neuron_select(enc, layer, items, top_k=m)
            raw_fn = raw_neuron_ablate_fn(raw["neurons"], baseline_vec)
            cells.append(run_cell({**base, "selection": "raw_neuron", "method": "raw_ablate"},
                                  enc, layer, sae, raw["neurons"], x_art, clean, act_fn=raw_fn,
                                  detection_auroc=raw["best_auroc"], **common))

            # -- DermFM-Zero: suppress top-DERM_K activation features (their recipe) ------
            derm = dermfmzero_select(sae, enc, layer, items, top_k=DERM_K)
            cells.append(run_cell({**base, "selection": "dermfmzero", "method": "sae_ablate_topk"},
                                  enc, layer, sae, derm["features"], x_art, clean, op="ablate", **common))

            # -- CAV / Reveal2Revise: clamp the artifact concept direction to clean ------
            cav = fit_cav(enc, layer, items)
            cav_fn = cav_suppress_fn(cav["direction"], cav_baseline_proj(acts_mean, cav["direction"]))
            cells.append(run_cell({**base, "selection": "cav", "method": "cav_suppress"},
                                  enc, layer, sae, [], x_art, clean, act_fn=cav_fn,
                                  detection_auroc=cav["best_auroc"], **common))

            # -- random raw-neuron control (matched budget) ------------------------------
            rnd_neurons = random_raw_neurons(d, m, rng)
            rnd_fn = raw_neuron_ablate_fn(rnd_neurons, baseline_vec)
            cells.append(run_cell({**base, "selection": "random", "method": "raw_ablate"},
                                  enc, layer, sae, rnd_neurons, x_art, clean, act_fn=rnd_fn, **common))

            # -- input-removal oracle = ceiling (R≡1) ------------------------------------
            cells.append(run_cell({**base, "selection": "oracle", "method": "input_remove"},
                                  enc, layer, sae, [], x_art, clean, oracle=True, **common))

            all_records.extend(cells)
            r = {c["selection"] + "/" + c["method"]: c["R_median"] for c in cells}
            print(f"  [{artifact:11s}] R: sae={r['oracle/sae_ablate']:+.3f} "
                  f"raw={r['raw_neuron/raw_ablate']:+.3f} derm={r['dermfmzero/sae_ablate_topk']:+.3f} "
                  f"cav={r['cav/cav_suppress']:+.3f} rand={r['random/raw_ablate']:+.3f} "
                  f"oracle={r['oracle/input_remove']:+.2f}")

    finish_grid(run, all_records, health_by_seed, title="baseline benchmark table (mean±std over seeds)")


if __name__ == "__main__":
    main()
