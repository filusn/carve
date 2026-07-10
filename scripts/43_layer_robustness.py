#!/usr/bin/env python
"""Phase-7 ROBUSTNESS: does detection ≠ control (and the §7 direction-mismatch mechanism) hold
across LAYERS, not just block 12?

The obvious reviewer attack on a single-layer result is "you cherry-picked the layer." For each
candidate block ℓ we train an SAE on that layer's CLEAN activations and record, per artifact:
detection AUROC (best present/absent feature on `select`), SAE-ablate recovery R and the
input-removal oracle ceiling (on the disjoint `eval`), and — the §7 mechanism — the effective
rank of the artifact's activation shift Δa plus |cos| between its causal direction (top singular
vector of Δa) and the directions the tools ablate (SAE detection feature, CAV). Frozen-compliant
width 4096. If detection stays high and recovery stays ≈0 at every layer (and the direction
mismatch persists), the dissociation is not a layer artifact.

    docker exec ... python3 scripts/43_layer_robustness.py           # layers {6,8,10,12}, 2 seeds
    docker exec ... python3 scripts/43_layer_robustness.py --quick   # layers {8,12}, 1 seed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from carve.data.artifacts import inject, make_biased_set  # noqa: E402
from carve.data.datasets import make_splits  # noqa: E402
from carve.data.isic import load_isic_binary  # noqa: E402
from carve.utils import assert_disjoint, load_config, new_run_dir, save_json, set_seed  # noqa: E402

ALPHA, RHO = 1.0, 0.9
WIDTH, K, AUXK = 4096, 32, 256                       # frozen-compliant SAE config
N_FULL = dict(sae_train=1200, select=300, eval=250, steps=3000)
N_QUICK = dict(sae_train=400, select=200, eval=120, steps=800)
LAYERS_FULL, LAYERS_QUICK = [6, 8, 10, 12], [8, 12]
SEEDS_FULL = [0, 1]                                   # supplementary robustness; main grid is 3-seed@12


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    N = N_QUICK if args.quick else N_FULL
    LAYERS = LAYERS_QUICK if args.quick else LAYERS_FULL
    seeds = [0] if args.quick else SEEDS_FULL

    print("=" * 72)
    print("CARVE — Phase-7 robustness: detection≠control + mechanism across layers")
    print("=" * 72)

    cfg = load_config(str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    ARTIFACTS = list(cfg.artifacts.types)
    size = int(cfg.dataset.image_size)
    cfg.sae.width, cfg.sae.k = WIDTH, K
    cfg.sae.train.steps, cfg.sae.train.aux_k = N["steps"], AUXK
    run = new_run_dir(cfg.paths.runs_dir, "layer_robustness", cfg, int(cfg.get("seed", 0)))
    eps = float(cfg.interventions.recovery_eps)

    from carve.baselines import fit_cav
    from carve.eval.harness import run_cell
    from carve.models.encoders import load_encoder
    from carve.sae.discovery import select_oracle
    from carve.sae.train_sae import sae_health, train_sae

    enc = load_encoder(cfg)
    ds = load_isic_binary(cfg, image_size=size)
    print(f"[layers] seeds={seeds} layers={LAYERS} artifacts={ARTIFACTS} width {WIDTH}")

    records = []
    for seed in seeds:
        set_seed(seed)
        splits = make_splits(ds.labels, dict(cfg.dataset.splits), seed=seed, stratify=True)
        assert_disjoint(**{k: set(v.tolist()) for k, v in splits.items()})
        sae_idx = splits["sae_train"][:N["sae_train"]]
        sel_idx = splits["select"][:N["select"]]
        ev_idx = splits["eval"][:N["eval"]]
        sae_imgs = [ds.load_image(int(i), size=size) for i in sae_idx]
        sel_imgs = [ds.load_image(int(i), size=size) for i in sel_idx]
        clean = [ds.load_image(int(i), size=size) for i in ev_idx]
        labels = ds.labels[ev_idx]
        sel_labels = ds.labels[sel_idx]

        for layer in LAYERS:
            acts = enc.activations(sae_imgs, layer, pool=None)
            acts = acts.reshape(-1, acts.shape[-1])
            sae = train_sae(acts, cfg, seed=seed)
            health = sae_health(sae, acts[int(0.8 * len(acts)):])
            print(f"[seed {seed} L{layer}] SAE R²={health['r2']:.3f} dead={health['dead_feature_frac']*100:.1f}%")

            for artifact in ARTIFACTS:
                items, _ = make_biased_set(sel_imgs, sel_labels, artifact, rho=RHO, alpha=ALPHA, seed=seed + 1)
                oracle = select_oracle(sae, enc, layer, items, top_m=1)
                fstar, det = oracle["features"][0], oracle["best_auroc"]
                w_sae = _unit(sae.W_dec[fstar].detach().float().cpu())
                cav = fit_cav(enc, layer, items)
                w_cav = _unit(torch.as_tensor(np.asarray(cav["direction"]), dtype=torch.float32))

                x_art = [inject(im, artifact, ALPHA, np.random.default_rng(int(i)))[0]
                         for im, i in zip(clean, ev_idx)]
                base = dict(model="monet", artifact=artifact, rho=RHO, opacity=ALPHA)
                common = dict(labels=labels, recovery_eps=eps, bootstrap=int(cfg.eval.bootstrap_resamples),
                              ci=float(cfg.eval.ci), seed=seed, run_dir=run)
                r_ab = run_cell({**base, "selection": "oracle", "method": f"sae_ablate_L{layer}"},
                                enc, layer, sae, [fstar], x_art, clean, op="ablate",
                                detection_auroc=det, **common)
                r_or = run_cell({**base, "selection": "oracle", "method": f"input_oracle_L{layer}"},
                                enc, layer, sae, [fstar], x_art, clean, oracle=True, **common)

                # mechanism at this layer: causal direction (top SVD of Δa) vs the detection dirs
                a_clean = enc.activations(clean, layer, pool=None).reshape(-1, sae.W_dec.shape[1]).float()
                a_art = enc.activations(x_art, layer, pool=None).reshape(-1, a_clean.shape[1]).float()
                dA = a_art - a_clean
                if dA.shape[0] > 30000:
                    idx = torch.as_tensor(np.random.default_rng(seed).choice(dA.shape[0], 30000, replace=False))
                    dA = dA[idx]
                _, sv, Vh = torch.linalg.svd(dA, full_matrices=False)
                lam = sv ** 2
                pr = float((lam.sum() ** 2) / (lam ** 2).sum())
                u = _unit(Vh[0].cpu())
                cos_sae = float(abs(torch.dot(u, w_sae)))
                cos_cav = float(abs(torch.dot(u, w_cav)))

                records.append(dict(seed=seed, layer=layer, artifact=artifact, detection_auroc=det,
                                    R_sae_ablate=r_ab["R_median"], R_oracle=r_or["R_median"],
                                    selectivity=r_ab["selectivity"], participation_ratio=pr,
                                    cos_causal_sae=cos_sae, cos_causal_cav=cos_cav))
                print(f"  L{layer} [{artifact:12s}] det {det:.3f} | R_ablate {r_ab['R_median']:+.3f} | "
                      f"eff.rank {pr:.2f} | cos(causal,SAE) {cos_sae:.3f}")

    save_json(run / "metrics.json", {"records": records})
    _figure(run, records, LAYERS, ARTIFACTS)
    print(f"\n[run] {run}")


def _figure(run, records, layers, artifacts) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(records)
    print("\n=== across layers (mean over seeds) ===")
    g = df.groupby(["artifact", "layer"]).agg(det=("detection_auroc", "mean"),
                                              R=("R_sae_ablate", "mean"),
                                              cos=("cos_causal_sae", "mean")).reset_index()
    print(g.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    fig, axes = plt.subplots(1, len(artifacts), figsize=(4.3 * len(artifacts), 4.0), squeeze=False)
    for j, art in enumerate(artifacts):
        d = g[g.artifact == art].set_index("layer").reindex(layers)
        ax = axes[0][j]
        ax.plot(layers, d["det"], "o-", color="#0f6e7d", label="detection AUROC")
        ax.plot(layers, d["R"], "s-", color="#1f77b4", label="SAE recovery R")
        ax.plot(layers, d["cos"], "^--", color="#b23a3f", label="|cos(causal, SAE feat)|")
        ax.axhline(0, ls=":", c="gray", lw=1); ax.axhline(1, ls=":", c="gray", lw=1)
        ax.set_xticks(layers); ax.set_xlabel("MONET block ℓ"); ax.set_ylim(-0.15, 1.05)
        ax.set_title(art, fontsize=10)
        if j == 0:
            ax.set_ylabel("value"); ax.legend(fontsize=8, loc="center right")
    fig.suptitle("Detection ≠ control is not a layer artifact: across blocks, detection stays high,\n"
                 "SAE recovery stays ≈0, and the detection feature stays misaligned with the causal direction",
                 fontsize=10)
    fig.tight_layout()
    out = Path(run) / "figures"; out.mkdir(exist_ok=True)
    p = out / "layer_robustness.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
