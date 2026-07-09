#!/usr/bin/env python
"""Phase-7 MECHANISM: why does linear control fail even though the artifact is ~rank-1?

Two measurements, per seed × artifact, at MONET block ℓ:

1. **Effective rank of the causal shift.** The artifact moves the activation by Δa =
   a(x_art) − a(x_clean). Its participation ratio (Σλ)²/Σλ² over the singular values of Δa is
   the *effective number of directions* the artifact uses. It is ≈1 — the causal effect is
   essentially a SINGLE activation direction (u_causal, the top singular vector of Δa).

2. **Detection direction ≠ causal direction.** The SAE feature that best DETECTS the artifact
   (highest present/absent AUROC on `select`) has a decoder direction W_dec[f*]; the CAV points
   along its own learned direction. We measure |cos| between each of these DETECTION directions
   and the artifact's CAUSAL direction u_causal. If the alignment is low, ablating the detection
   feature / CAV removes the WRONG direction — so a method can detect the artifact yet not
   control it. This is "detection ≠ control" at the level of vectors, and it explains the R≈0
   of §5–§6 despite the effect being low-rank.

    docker exec ... python3 scripts/42_effect_dimensionality.py           # 3 seeds
    docker exec ... python3 scripts/42_effect_dimensionality.py --quick   # 1 seed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from carve.data.artifacts import inject, make_biased_set  # noqa: E402
from carve.data.isic import load_isic_binary  # noqa: E402
from carve.eval.grid import iter_seed_contexts, resolve_seeds, setup_run  # noqa: E402
from carve.utils import load_config, save_json  # noqa: E402

ALPHA, RHO = 1.0, 0.9
FULL = dict(sae_train=1200, select=300, eval=250, width=4096, k=32, steps=3000)   # frozen-compliant
QUICK = dict(sae_train=400, select=200, eval=120, width=4096, k=32, steps=800)


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-8)


def _participation_ratio(svals: torch.Tensor) -> float:
    lam = (svals ** 2)
    return float((lam.sum() ** 2) / (lam ** 2).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    N = QUICK if args.quick else FULL

    print("=" * 72)
    print("CARVE — Phase-7 mechanism: rank-1 causal direction vs. the detection directions")
    print("=" * 72)

    cfg = load_config(str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    ARTIFACTS = list(cfg.artifacts.types)
    run, layer, size = setup_run(cfg, "effect_dimensionality", N)
    seeds = resolve_seeds(cfg, args.quick)

    from carve.baselines import fit_cav
    from carve.models.encoders import load_encoder
    from carve.sae.discovery import select_oracle

    enc = load_encoder(cfg)
    ds = load_isic_binary(cfg, image_size=size)
    print(f"[mech] seeds={seeds} artifacts={ARTIFACTS} layer {layer} width {N['width']}")

    records = []
    for ctx in iter_seed_contexts(cfg, ds, enc, N, seeds, layer, size):
        seed, sae = ctx.seed, ctx.sae
        for artifact in ARTIFACTS:
            # detection direction: the SAE feature that best separates present/absent on `select`
            items, _ = make_biased_set(ctx.sel_imgs, ctx.sel_labels, artifact, rho=RHO, alpha=ALPHA, seed=seed + 1)
            oracle = select_oracle(sae, enc, layer, items, top_m=1)
            fstar, det = oracle["features"][0], oracle["best_auroc"]
            w_sae = _unit(sae.W_dec[fstar].detach().float().cpu())          # SAE detection dir
            cav = fit_cav(enc, layer, items)
            w_cav = _unit(torch.as_tensor(np.asarray(cav["direction"]), dtype=torch.float32))

            # causal direction: top singular vector of the per-image/token activation shift Δa
            x_art = [inject(im, artifact, ALPHA, np.random.default_rng(int(i)))[0]
                     for im, i in zip(ctx.clean, ctx.ev_idx)]
            a_clean = enc.activations(ctx.clean, layer, pool=None).reshape(-1, sae.W_dec.shape[1]).float()
            a_art = enc.activations(x_art, layer, pool=None).reshape(-1, a_clean.shape[1]).float()
            dA = a_art - a_clean
            rows = dA.shape[0]
            if rows > 30000:
                idx = torch.as_tensor(np.random.default_rng(seed).choice(rows, 30000, replace=False))
                dA_s = dA[idx]
            else:
                dA_s = dA
            _, svals, Vh = torch.linalg.svd(dA_s, full_matrices=False)
            pr = _participation_ratio(svals)
            u_causal = _unit(Vh[0].cpu())                                   # rank-1 causal dir
            var_top1 = float((svals[0] ** 2) / (svals ** 2).sum())          # variance frac in top dir

            cos_sae = float(abs(torch.dot(u_causal, w_sae)))
            cos_cav = float(abs(torch.dot(u_causal, w_cav)))
            # how well the SAE dictionary represents the causal direction at all (best |cos| over all features)
            Wd = torch.nn.functional.normalize(sae.W_dec.detach().float().cpu(), dim=1)   # [width,d]
            best_cos_any = float(Wd.mv(u_causal).abs().max())

            rec = dict(seed=seed, artifact=artifact, detection_auroc=det,
                       participation_ratio=pr, var_frac_top1=var_top1,
                       cos_causal_vs_sae_feature=cos_sae, cos_causal_vs_cav=cos_cav,
                       best_cos_causal_vs_any_sae_feature=best_cos_any)
            records.append(rec)
            print(f"  [{artifact:12s}] seed {seed}: eff.rank={pr:.2f} (top dir {var_top1*100:.0f}% var) | "
                  f"det AUROC {det:.3f} | cos(causal, SAE feat)={cos_sae:.3f}  cos(causal, CAV)={cos_cav:.3f}  "
                  f"best cos(any SAE feat)={best_cos_any:.3f}")

    save_json(run / "metrics.json", {"records": records})
    _summary_and_figure(run, records, ARTIFACTS)
    print(f"\n[run] {run}")


def _summary_and_figure(run, records, artifacts) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(records)
    g = df.groupby("artifact").agg(
        eff_rank=("participation_ratio", "mean"), det=("detection_auroc", "mean"),
        cos_sae=("cos_causal_vs_sae_feature", "mean"), cos_cav=("cos_causal_vs_cav", "mean"),
        best_cos=("best_cos_causal_vs_any_sae_feature", "mean")).reindex(artifacts)
    print("\n=== mechanism summary (3-seed mean) ===")
    print(g.to_string(float_format=lambda x: f"{x:.3f}"))

    x = np.arange(len(artifacts)); w = 0.35
    fig, ax = plt.subplots(figsize=(1.7 * len(artifacts) + 3, 4.2))
    ax.bar(x - w / 2, g["det"], w, label="detection AUROC (can it SEE the artifact?)", color="#0f6e7d")
    ax.bar(x + w / 2, g["cos_sae"], w, label="|cos(causal dir, SAE detection feature)|", color="#b23a3f")
    ax.plot(x + w / 2, g["cos_cav"], "D", color="#b8752b", label="|cos(causal dir, CAV)|", ms=8)
    for i, a in enumerate(artifacts):
        ax.text(i, 0.02, f"eff. rank\n{g.loc[a,'eff_rank']:.1f}", ha="center", va="bottom", fontsize=8, color="#333")
    ax.set_xticks(x); ax.set_xticklabels(artifacts)
    ax.set_ylim(0, 1.05); ax.set_ylabel("value")
    ax.set_title("Detection ≠ control at the vector level: the artifact effect is ~rank-1, but the\n"
                 "best-detecting SAE feature / CAV point in a DIFFERENT direction than the causal one",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper center")
    fig.tight_layout()
    out = Path(run) / "figures"; out.mkdir(exist_ok=True)
    p = out / "mechanism_detection_vs_causal_direction.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
