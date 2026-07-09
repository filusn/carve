"""Run harness: evaluate ONE method on ONE cell and emit a RunRecord (Phase 5).

A "cell" = (model, artifact, ρ, opacity, selection∈{oracle,discovered,random}, method,
seed). run_cell computes the headline metrics (causal recovery, selectivity, off-target)
from the gold input effect and the feature-level intervention, writes per-image outputs to
a run dir, and returns the RunRecord dict. All methods (SAE ablate/steer + baselines) plug
in through the same op/S interface so the numbers stay comparable.

run_steer_bestcase adds the "best-case steering" ceiling (issue #2): sweep a grid of steer
coefficients, emit the recovery-vs-coefficient curve AND a per-image ORACLE coefficient
(each image gets the coefficient that best drives f(x_art) back to clean). If even that
per-image best coefficient can't recover, no single steering magnitude can — the strongest
form of "detection ≠ control".
"""
from __future__ import annotations

import numpy as np

from ..interventions.hooks import sae_ablate_fn, sae_steer_fn
from ..interventions.mediation import hooked_decision
from ..metrics.causal import causal_recovery, off_target, selectivity
from ..metrics.stats import bootstrap_ci
from ..models.probe import f_decision
from ..utils import git_commit, save_json

RUNRECORD_COLUMNS = [
    "model", "artifact", "rho", "opacity", "selection", "method", "seed", "layer",
    "R_median", "R_ci_lo", "R_ci_hi", "cause", "isolation", "selectivity", "off_target",
    "detection_auroc", "n_images", "run_dir", "git_commit",
]


def _finalize_record(
    meta: dict, layer: int, f_art, f_removed, f_feat_art, f_feat_clean, *,
    labels=None, detection_auroc=None, coeff=None, S=None,
    recovery_eps: float = 1e-3, bootstrap: int = 1000, ci: float = 0.95,
    seed: int = 0, run_dir=None,
) -> dict:
    """Compute the headline metrics from the four per-image decision vectors, write the
    per-image parquet + cell json, and return the RunRecord. Shared by run_cell and
    run_steer_bestcase; the four f_* are per-image numpy arrays over the SAME images.
    """
    f_art = np.asarray(f_art, dtype=float)
    f_removed = np.asarray(f_removed, dtype=float)
    f_feat_art = np.asarray(f_feat_art, dtype=float)
    f_feat_clean = np.asarray(f_feat_clean, dtype=float)

    e_in = f_art - f_removed
    R = causal_recovery(f_art, f_feat_art, f_removed, eps=recovery_eps).numpy()
    sel = selectivity(f_art, f_feat_art, f_removed, f_feat_clean)

    kept = R[~np.isnan(R)]
    R_med = float(np.median(kept)) if kept.size else float("nan")
    lo, hi = bootstrap_ci(R, n=bootstrap, ci=ci, rng=seed)

    if labels is not None:
        labels = np.asarray(labels)
        acc_before = float(((f_removed > 0).astype(int) == labels).mean())
        acc_after = float(((f_feat_clean > 0).astype(int) == labels).mean())
        ot = off_target(acc_before, acc_after)
    else:  # no labels → report the clean-side decision shift magnitude
        ot = float(np.abs(f_removed - f_feat_clean).mean())

    rec = {
        **{k: meta.get(k) for k in ("model", "artifact", "rho", "opacity", "selection", "method")},
        "seed": seed, "layer": layer,
        "R_median": R_med, "R_ci_lo": lo, "R_ci_hi": hi,
        "cause": sel["cause"], "isolation": sel["isolation"], "selectivity": sel["selectivity"],
        "off_target": ot, "detection_auroc": detection_auroc,
        "e_in_median": float(np.median(e_in)), "e_in_abs_median": float(np.median(np.abs(e_in))),
        "coeff": coeff, "features": list(np.asarray(S).ravel().tolist()) if S is not None else None,
        "n_images": int(len(f_art)), "run_dir": str(run_dir) if run_dir else None,
        "git_commit": git_commit(),
    }

    if run_dir is not None:
        import pandas as pd
        tag = f"s{seed}_{meta.get('artifact')}_{meta.get('selection')}_{meta.get('method')}"
        pd.DataFrame({
            "f_art": f_art, "f_removed": f_removed,
            "f_feat_art": f_feat_art, "f_feat_clean": f_feat_clean,
            "e_in": e_in, "R": R, "label": labels if labels is not None else np.nan,
        }).to_parquet(f"{run_dir}/perimage_{tag}.parquet")
        save_json(f"{run_dir}/cell_{tag}.json", rec)
    return rec


def run_cell(
    meta: dict, enc, layer: int, sae, S, x_art, x_clean, *, labels=None, probe=None,
    op: str = "ablate", coeff: float | None = None, act_fn=None, oracle: bool = False,
    detection_auroc: float | None = None,
    recovery_eps: float = 1e-3, bootstrap: int = 1000, ci: float = 0.95,
    seed: int = 0, run_dir=None,
) -> dict:
    """meta must carry: model, artifact, rho, opacity, selection, method.

    f = zero-shot margin (probe=None) or the induced probe margin. Every method is applied
    identically to artifact and clean images so selectivity/off-target are on the same
    footing. Three ways to specify the intervention, in priority order:
      * ``oracle=True``    — input-removal ceiling: the "feat" decisions are f(clean) itself
                             (R≡1 by construction; the achievable reference row).
      * ``act_fn`` given   — any activation editor a→a′ (baselines: raw-neuron, dermfmzero,
                             cav, …); hooked at block ℓ.
      * else (default)     — SAE ablate/steer of feature set S (Phase-5 back-compat).
    """
    f_art = f_decision(probe, enc, x_art, layer)
    f_removed = f_decision(probe, enc, x_clean, layer)
    if oracle:
        # Removing the artifact at the input maps x_art→clean and leaves x_clean unchanged;
        # both "feat" decisions are therefore the clean-source decision f_removed.
        f_feat_art = f_removed
        f_feat_clean = f_removed
    else:
        if act_fn is None:
            act_fn = sae_ablate_fn(sae, S) if op == "ablate" else sae_steer_fn(sae, S, coeff)
        f_feat_art = hooked_decision(probe, enc, layer, act_fn, x_art)
        f_feat_clean = hooked_decision(probe, enc, layer, act_fn, x_clean)

    return _finalize_record(
        meta, layer, f_art.numpy(), f_removed.numpy(), f_feat_art.numpy(), f_feat_clean.numpy(),
        labels=labels, detection_auroc=detection_auroc, coeff=coeff, S=S,
        recovery_eps=recovery_eps, bootstrap=bootstrap, ci=ci, seed=seed, run_dir=run_dir,
    )


def run_steer_bestcase(
    meta: dict, enc, layer: int, sae, S, x_art, x_clean, coeffs, *, labels=None, probe=None,
    detection_auroc: float | None = None, recovery_eps: float = 1e-3,
    bootstrap: int = 1000, ci: float = 0.95, seed: int = 0, run_dir=None,
) -> list[dict]:
    """Best-case steering ceiling for feature set S (issue #2).

    Steers ``a ← a − c·Σ W_dec[S]`` for every c in ``coeffs`` and records each as method
    ``sae_steer_c{c}`` (the recovery-vs-coefficient curve — this is where fixed-grid steering
    overshoots). Then, per image, picks the coefficient that minimizes ``|f_feat_art −
    f_clean|`` and records that as ``sae_steer_bestcase`` — the largest recovery ANY single
    steering magnitude could give each image. meta carries model/artifact/rho/opacity/selection;
    method is set per record here. Returns the list of RunRecords.
    """
    f_art = f_decision(probe, enc, x_art, layer).numpy()
    f_removed = f_decision(probe, enc, x_clean, layer).numpy()
    n = len(f_art)
    coeffs = list(coeffs)
    FA = np.empty((len(coeffs), n))   # f_feat on artifact images, per coefficient
    FC = np.empty((len(coeffs), n))   # f_feat on clean images, per coefficient

    recs: list[dict] = []
    for j, c in enumerate(coeffs):
        fn = sae_steer_fn(sae, S, float(c))
        FA[j] = hooked_decision(probe, enc, layer, fn, x_art).numpy()
        FC[j] = hooked_decision(probe, enc, layer, fn, x_clean).numpy()
        recs.append(_finalize_record(
            {**meta, "method": f"sae_steer_c{c}"}, layer, f_art, f_removed, FA[j], FC[j],
            labels=labels, detection_auroc=detection_auroc, coeff=float(c), S=S,
            recovery_eps=recovery_eps, bootstrap=bootstrap, ci=ci, seed=seed, run_dir=run_dir,
        ))

    # per-image ORACLE coefficient: the c that drives each image's f(x_art) closest to clean
    jstar = np.abs(FA - f_removed[None, :]).argmin(axis=0)
    rows = np.arange(n)
    fa_star, fc_star = FA[jstar, rows], FC[jstar, rows]
    recs.append(_finalize_record(
        {**meta, "method": "sae_steer_bestcase"}, layer, f_art, f_removed, fa_star, fc_star,
        labels=labels, detection_auroc=detection_auroc, coeff=-1.0, S=S,
        recovery_eps=recovery_eps, bootstrap=bootstrap, ci=ci, seed=seed, run_dir=run_dir,
    ))
    return recs
