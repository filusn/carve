#!/usr/bin/env python
"""SAE-health sweep (Phase-4 hardening, task #13): pick the SAE width/config for the FINAL
grid on evidence, not vibes. The width-16384 dictionary is ~55% dead at k=32/3000 steps; is
that an over-wide dictionary (fix = smaller width) or an under-trained one (fix = AuxK
dead-feature revival)? Train each candidate on the SAME per-seed activations and report
dead-feature fraction, R², artifact detection AUROC (ruler `select`), and cross-seed decoder
cosine stability. The winner + rationale is frozen in PREREGISTRATION.md.

    docker exec carve-dev python3 scripts/31_sae_health_sweep.py            # 2 seeds
    docker exec carve-dev python3 scripts/31_sae_health_sweep.py --quick    # 1 seed, tiny
"""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from carve.data.artifacts import make_biased_set  # noqa: E402
from carve.data.datasets import make_splits  # noqa: E402
from carve.data.isic import load_isic_binary  # noqa: E402
from carve.utils import assert_disjoint, load_config, new_run_dir, save_json, set_seed  # noqa: E402

# candidate SAE configs — name, width, k, aux_k (0 = plain TopK)
CONFIGS = [
    ("w4096", 4096, 32, 0),
    ("w16384", 16384, 32, 0),
    ("w16384_auxk", 16384, 32, 256),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_sae, n_sel, steps = (400, 200, 800) if args.quick else (1200, 300, 3000)
    seeds = [0] if args.quick else [0, 1]

    print("=" * 72)
    print("CARVE — SAE health sweep (width / AuxK) for the final-grid decision")
    print("=" * 72)

    cfg = load_config(str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    run = new_run_dir(cfg.paths.runs_dir, "sae_health_sweep", cfg, int(cfg.get("seed", 0)))
    size = int(cfg.dataset.image_size)
    layer = int(cfg.sae.get("layer", 12))

    from carve.models.encoders import load_encoder
    from carve.sae.discovery import select_oracle
    from carve.sae.train_sae import decoder_cosine_stability, sae_health, train_sae

    enc = load_encoder(cfg)
    ds = load_isic_binary(cfg, image_size=size)
    print(f"[sweep] seeds={seeds}  configs={[c[0] for c in CONFIGS]}  layer {layer}  steps {steps}")

    rows, saes = [], {}
    for seed in seeds:
        set_seed(seed)
        splits = make_splits(ds.labels, dict(cfg.dataset.splits), seed=seed, stratify=True)
        assert_disjoint(**{k: set(v.tolist()) for k, v in splits.items()})
        sae_idx = splits["sae_train"][:n_sae]
        sel_idx = splits["select"][:n_sel]
        acts = enc.activations([ds.load_image(int(i), size=size) for i in sae_idx], layer, pool=None)
        acts = acts.reshape(-1, acts.shape[-1])
        held = acts[int(0.8 * len(acts)):]
        sel_imgs = [ds.load_image(int(i), size=size) for i in sel_idx]
        items, _ = make_biased_set(sel_imgs, ds.labels[sel_idx], "ruler", rho=0.9, alpha=1.0, seed=seed + 1)

        for name, width, k, aux_k in CONFIGS:
            scfg = deepcopy(cfg)
            scfg.sae.width, scfg.sae.k = width, k
            scfg.sae.train.steps = steps
            scfg.sae.train.aux_k = aux_k
            sae = train_sae(acts, scfg, seed=seed, log=lambda *a: None)
            h = sae_health(sae, held)
            det = select_oracle(sae, enc, layer, items, top_m=1)["best_auroc"]
            saes[(name, seed)] = sae
            rows.append(dict(config=name, seed=seed, width=width, aux_k=aux_k,
                             r2=round(h["r2"], 4), dead=round(h["dead_feature_frac"], 4),
                             n_active=h["n_active_features"], det_auroc=round(det, 4)))
            print(f"[seed {seed}] {name:14s} R²={h['r2']:.3f} dead={h['dead_feature_frac']*100:4.1f}% "
                  f"n_active={h['n_active_features']:5d} det_auroc={det:.3f}")

    stability = {}
    if len(seeds) >= 2:
        for name, *_ in CONFIGS:
            s = decoder_cosine_stability(saes[(name, seeds[0])], saes[(name, seeds[1])])
            stability[name] = round(s["mean_best_cosine"], 4)
        print("\n[cross-seed decoder stability (mean best-match cosine, ~1 = stable)]")
        for name, v in stability.items():
            print(f"   {name:14s} {v:.3f}")

    save_json(run / "sweep.json", {"rows": rows, "decoder_stability": stability,
                                   "steps": steps, "n_sae_imgs": n_sae})
    print(f"\n[run] {run}")


if __name__ == "__main__":
    main()
