#!/usr/bin/env python
"""Phase-3/4: train a TopK SAE on MONET layer-ℓ activations, report health, run discovery.

Steps (docs/EXECUTION_PLAN.md Phases 3-4):
  1. load MONET + ISIC; seeded disjoint splits.
  2. extract layer-ℓ residual activations (patch+CLS tokens) over `sae_train` → train SAE.
  3. sae_health on held-out activations (FVU/R², dead-feature fraction).
  4. discovery on a ρ-biased `select` set for the headline artifact:
       - select_oracle (detection AUROC, uses injected labels)
       - discover_unsupervised (variance, no labels) + precision@k vs oracle.
  5. save SAE checkpoint + health/discovery json to the run dir.

    docker exec carve-dev python3 scripts/30_train_sae.py            # full
    docker exec carve-dev python3 scripts/30_train_sae.py --quick    # small, fast smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from carve.data.artifacts import make_biased_set  # noqa: E402
from carve.data.datasets import make_splits  # noqa: E402
from carve.data.isic import load_isic_binary  # noqa: E402
from carve.utils import assert_disjoint, load_config, new_run_dir, save_json, set_seed  # noqa: E402

FULL = dict(sae_train=1500, select=400, width=16384, k=32, steps=3000)
QUICK = dict(sae_train=400, select=200, width=4096, k=32, steps=800)
HEADLINE_ARTIFACT = "ruler"


@torch.no_grad()
def extract_token_acts(enc, ds, idx, layer, size):
    imgs = [ds.load_image(int(i), size=size) for i in idx]
    acts = enc.activations(imgs, layer, pool=None)      # [N, T, d] on cpu
    return acts.reshape(-1, acts.shape[-1])             # [N*T, d]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    N = QUICK if args.quick else FULL

    print("=" * 72)
    print("CARVE — Phase-3/4: train MONET SAE + feature discovery")
    print("=" * 72)

    cfg = load_config(str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    run = new_run_dir(cfg.paths.runs_dir, "train_sae", cfg, seed)
    size = int(cfg.dataset.image_size)
    layer = int(cfg.sae.get("layer", 12))
    # override width/k/steps for --quick via a shallow copy of cfg.sae
    cfg.sae.width, cfg.sae.k = N["width"], N["k"]
    cfg.sae.train.steps = N["steps"]

    from carve.models.encoders import load_encoder
    from carve.sae.discovery import (discover_unsupervised, discovery_precision_at_k,
                                      select_oracle)
    from carve.sae.train_sae import sae_health, train_sae

    print(f"\n[load] MONET + ISIC (layer ℓ={layer}, width={N['width']}, k={N['k']}) ...")
    enc = load_encoder(cfg)
    ds = load_isic_binary(cfg, image_size=size)
    splits = make_splits(ds.labels, dict(cfg.dataset.splits), seed=seed, stratify=True)
    assert_disjoint(**{k: set(v.tolist()) for k, v in splits.items()})
    sae_idx = splits["sae_train"][: N["sae_train"]]
    sel_idx = splits["select"][: N["select"]]

    # ---- extract activations + train ----------------------------------------------------
    print(f"[acts] extracting layer-{layer} tokens over {len(sae_idx)} sae_train images ...")
    acts = extract_token_acts(enc, ds, sae_idx, layer, size)
    print(f"[acts] {tuple(acts.shape)} activation vectors (tokens × images)")
    sae = train_sae(acts, cfg, seed=seed)

    # health on a held-out chunk (last 20% of the extracted tokens, kept out of sampling bias)
    health = sae_health(sae, acts[int(0.8 * len(acts)):])
    print(f"[health] R²={health['r2']:.3f}  FVU={health['fvu']:.3f}  "
          f"dead={health['dead_feature_frac']*100:.1f}%  active={health['n_active_features']}/{sae.width}")

    torch.save(sae.state_dict(), run / "sae.pt")

    # ---- discovery on a rho-biased select set (headline artifact) -----------------------
    sel_imgs = [ds.load_image(int(i), size=size) for i in sel_idx]
    items, summ = make_biased_set(sel_imgs, ds.labels[sel_idx], HEADLINE_ARTIFACT,
                                  rho=0.9, alpha=1.0, seed=seed + 1)
    oracle = select_oracle(sae, enc, layer, items, top_m=int(cfg.sae.feature_set_size_m))
    disc = discover_unsupervised(sae, enc, layer, items, top_m=5)
    prec = discovery_precision_at_k(disc, oracle)
    print(f"[discovery/{HEADLINE_ARTIFACT}] oracle feature {oracle['features']} "
          f"detection AUROC {oracle['best_auroc']:.3f}")
    print(f"[discovery/{HEADLINE_ARTIFACT}] unsupervised top-5 {disc['features']} "
          f"| precision@5 vs oracle {prec:.2f}")

    save_json(run / "metrics.json", {
        "layer": layer, "sae": {"width": sae.width, "k": sae.k},
        "n_sae_train_images": len(sae_idx), "n_activation_vectors": int(acts.shape[0]),
        "health": health,
        "discovery": {"artifact": HEADLINE_ARTIFACT, "biased_set": summ["realized"],
                      "oracle": oracle, "discovered": disc, "precision_at_k": prec},
    })
    print(f"\n[run] {run}")
    good = health["r2"] > 0.5 and oracle["best_auroc"] > 0.9
    print("[detect] SAE reconstructs well AND a feature detects the artifact ✓"
          if good else "[detect] check health/detection before Stage 5.")


if __name__ == "__main__":
    main()
