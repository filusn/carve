#!/usr/bin/env python
"""CARVE Phase-0 day-1 GO/NO-GO smoke test — bias-confirmation half (real MONET + ISIC).

Purpose (docs/EXECUTION_PLAN.md Phase 0): in the easiest setting — opaque artifacts,
ρ=1.0 — confirm there is a shortcut worth recovering BEFORE building the full grid.
This runs the FM/probe half of the gate:

  ZERO-SHOT arm (no training, purest causal GT):
      e_in = f(x_art) − f(x_clean)  on held-out images, per artifact.
      f = MONET zero-shot mel−nevus logit margin. Report magnitude, direction, and a
      bootstrap CI of the median; the artifact *matters* iff the CI excludes 0.
  INDUCED arm (the ρ knob):
      train a class-weighted probe on a ρ=1.0-biased set (artifact ⇔ melanoma), then
      BIAS GAP = acc(artifact-ALIGNED eval) − acc(artifact-CONFLICTING eval).
      bias_gap ≈ 0 ⇒ the probe ignores the artifact ⇒ injection invalid (fix before scaling).

Splits are seeded and disjoint (probe_train ⟂ eval ⟂ test asserted). The remaining gate
condition — an SAE feature that DETECTS the artifact and whose ablation RECOVERS the effect
— is Stage 4; this script decides whether it is worth building.

    docker exec carve-dev python3 scripts/01_smoke_test.py            # full gate
    docker exec carve-dev python3 scripts/01_smoke_test.py --quick    # fewer images
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from carve.data.artifacts import ARTIFACT_KINDS, inject  # noqa: E402
from carve.data.datasets import make_splits  # noqa: E402
from carve.data.isic import load_isic_binary  # noqa: E402
from carve.metrics.causal import bias_gap, input_effect  # noqa: E402
from carve.metrics.stats import bootstrap_ci  # noqa: E402
from carve.utils import assert_disjoint, load_config, new_run_dir, save_json, set_seed  # noqa: E402

# gate sample sizes (a smoke test, not the full grid) — override with --quick
FULL = dict(probe_train=500, eval=200, zeroshot=120)
QUICK = dict(probe_train=160, eval=96, zeroshot=64)
ALPHA = 1.0   # opaque — easiest setting
RHO = 1.0     # artifact perfectly predicts the label in the induced train set


def _check_env():
    print("[env] optional dependency check:")
    for mod in ["torch", "transformers", "sklearn", "omegaconf"]:
        try:
            importlib.import_module(mod)
            print(f"   - {mod:12s} ok")
        except Exception as e:  # noqa: BLE001
            print(f"   - {mod:12s} MISSING ({type(e).__name__})")


def _load_images(ds, idx, size):
    return [ds.load_image(int(i), size=size) for i in idx]


def _bias_gap_eval(images, labels, kind, alpha, seed):
    """Build ALIGNED and CONFLICTING eval sets under the learned 'artifact ⇔ melanoma' rule.

    aligned:     inject on positives (mel) only  → artifact points to the TRUE label.
    conflicting: inject on negatives (nev) only  → artifact points to the WRONG label.
    A shortcut-reliant probe scores higher on aligned than conflicting.
    """
    rng = np.random.default_rng(seed)
    aligned, conflicting = [], []
    for img, lab in zip(images, labels):
        seed_i = int(rng.integers(0, 2**31))
        pos_art, _ = inject(img, kind, alpha, np.random.default_rng(seed_i))
        aligned.append(pos_art if lab == 1 else img)
        conflicting.append(pos_art if lab == 0 else img)
    return aligned, conflicting


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    N = QUICK if args.quick else FULL

    print("=" * 72)
    print("CARVE — Phase-0 GO/NO-GO (bias confirmation)  —  MONET × ISIC-2018")
    print("=" * 72)
    _check_env()

    cfg = load_config(str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    run = new_run_dir(cfg.paths.runs_dir, "phase0_gate", cfg, seed)
    size = int(cfg.dataset.image_size)

    from carve.models.encoders import load_encoder
    from carve.models.probe import f_decision, probe_accuracy, train_probe

    print("\n[load] MONET + ISIC-2018 Task3 (mel-vs-nevus) ...")
    enc = load_encoder(cfg)
    ds = load_isic_binary(cfg, image_size=size)
    print(f"[data] {ds.summary()}")

    # seeded, disjoint splits (assert no leakage) — cap to gate sizes
    splits = make_splits(ds.labels, dict(cfg.dataset.splits), seed=seed, stratify=True)
    assert_disjoint(**{k: set(v.tolist()) for k, v in splits.items()})
    tr_idx = splits["probe_train"][: N["probe_train"]]
    ev_idx = splits["eval"][: max(N["eval"], N["zeroshot"])]
    layer = int(cfg.model.get("probe_layer", enc.n_layers // 2))
    print(f"[splits] probe_train={len(tr_idx)}  eval={len(ev_idx)}  layer ℓ={layer}  "
          f"α={ALPHA} ρ={RHO}")

    report = {"layer": layer, "alpha": ALPHA, "rho": RHO, "artifacts": {}}

    for kind in ARTIFACT_KINDS:
        entry = {}

        # ---- ZERO-SHOT arm: input-level effect e_in on held-out eval images -------------
        zs_idx = ev_idx[: N["zeroshot"]]
        clean = _load_images(ds, zs_idx, size)
        arted = [inject(im, kind, ALPHA, np.random.default_rng(int(i)))[0]
                 for im, i in zip(clean, zs_idx)]
        f_clean = f_decision(None, enc, clean)          # zero-shot margins
        f_art = f_decision(None, enc, arted)
        e_in = input_effect(f_art, f_clean).numpy()
        lo, hi = bootstrap_ci(e_in, n=cfg.eval.bootstrap_resamples, ci=cfg.eval.ci, rng=seed)
        entry["zero_shot"] = {
            "e_in_mean": float(e_in.mean()),
            "e_in_median": float(np.median(e_in)),
            "e_in_ci95": [lo, hi],
            "abs_median": float(np.median(np.abs(e_in))),
            "frac_toward_melanoma": float((e_in > 0).mean()),
            "n": len(e_in),
            "ci_excludes_0": bool(lo > 0 or hi < 0),
        }

        # ---- INDUCED arm: train ρ=1.0-biased probe, measure bias gap --------------------
        tr_imgs = _load_images(ds, tr_idx, size)
        tr_labels = ds.labels[tr_idx]
        # ρ=1.0: artifact on positives only, never on negatives
        biased_train = []
        for j, (im, lab) in enumerate(zip(tr_imgs, tr_labels)):
            if lab == 1:
                im = inject(im, kind, ALPHA, np.random.default_rng(seed * 7919 + j))[0]
            biased_train.append({"image": im, "label": int(lab)})
        probe = train_probe(enc, layer, biased_train)

        bg_idx = ev_idx[: N["eval"]]
        bg_imgs = _load_images(ds, bg_idx, size)
        bg_labels = ds.labels[bg_idx]
        aligned, conflicting = _bias_gap_eval(bg_imgs, bg_labels, kind, ALPHA, seed + 1)
        acc_aligned = float(((f_decision(probe, enc, aligned, layer).numpy() > 0).astype(int)
                             == bg_labels).mean())
        acc_conflicting = float(((f_decision(probe, enc, conflicting, layer).numpy() > 0).astype(int)
                                 == bg_labels).mean())
        acc_clean = probe_accuracy(
            probe, enc, [{"image": im, "label": int(l)} for im, l in zip(bg_imgs, bg_labels)]
        )
        entry["induced"] = {
            "bias_gap": bias_gap(acc_aligned, acc_conflicting),
            "acc_aligned": acc_aligned,
            "acc_conflicting": acc_conflicting,
            "acc_clean": acc_clean,
            "n_train": len(biased_train),
            "n_eval": len(bg_labels),
        }
        report["artifacts"][kind] = entry

        zs, ind = entry["zero_shot"], entry["induced"]
        print(f"\n[{kind}]")
        print(f"  zero-shot  e_in median {zs['e_in_median']:+.3f}  CI95 "
              f"[{zs['e_in_ci95'][0]:+.3f},{zs['e_in_ci95'][1]:+.3f}]  "
              f"|median| {zs['abs_median']:.3f}  →mel {zs['frac_toward_melanoma']*100:.0f}%  "
              f"{'(CI≠0 ✓)' if zs['ci_excludes_0'] else '(CI overlaps 0)'}")
        print(f"  induced    bias_gap {ind['bias_gap']:+.3f}  "
              f"(aligned {ind['acc_aligned']:.3f} vs conflicting {ind['acc_conflicting']:.3f}; "
              f"clean {ind['acc_clean']:.3f})")

    # ---- decision -----------------------------------------------------------------------
    zs_hits = [k for k, e in report["artifacts"].items() if e["zero_shot"]["ci_excludes_0"]]
    ind_hits = [k for k, e in report["artifacts"].items() if e["induced"]["bias_gap"] >= 0.10]
    bias_confirmed = bool(zs_hits) and bool(ind_hits)
    report["decision"] = {
        "bias_confirmed": bias_confirmed,
        "zero_shot_significant": zs_hits,
        "induced_bias_gap_ge_0.10": ind_hits,
        "note": "SAE detection+recovery (2nd gate condition) is Stage 4.",
    }
    save_json(run / "metrics.json", report)

    print("\n" + "-" * 72)
    if bias_confirmed:
        print(f"[GO — bias confirmed] zero-shot effect significant for {zs_hits}; "
              f"induced bias gap ≥0.10 for {ind_hits}.")
        print("      → There IS a shortcut to recover. Proceed to Stage 4 (SAE) to test the")
        print("        detection+recovery half of the gate.")
    else:
        print("[HOLD / NO-GO] bias not clearly confirmed in the easiest setting.")
        print("      → Inspect injection/prompts before scaling. If robust across settings,")
        print("        this itself is the negative-result benchmark framing (tell PI).")
    print(f"[run] {run}")


if __name__ == "__main__":
    main()
