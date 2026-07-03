"""Run harness: evaluate ONE method on ONE cell and emit a RunRecord (Phase 5).

A "cell" = (model, artifact, ρ, opacity, selection∈{oracle,discovered,random}, method,
seed). run_cell computes the headline metrics (causal recovery, selectivity, off-target)
from the gold input effect and the feature-level intervention, writes per-image outputs to
a run dir, and returns the RunRecord dict. All methods (SAE ablate/steer + baselines) plug
in through the same op/S interface so the numbers stay comparable.
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

    e_in = (f_art - f_removed).numpy()
    R = causal_recovery(f_art, f_feat_art, f_removed, eps=recovery_eps).numpy()
    sel = selectivity(f_art, f_feat_art, f_removed, f_feat_clean)

    kept = R[~np.isnan(R)]
    R_med = float(np.median(kept)) if kept.size else float("nan")
    lo, hi = bootstrap_ci(R, n=bootstrap, ci=ci, rng=seed)

    if labels is not None:
        labels = np.asarray(labels)
        acc_before = float(((f_removed.numpy() > 0).astype(int) == labels).mean())
        acc_after = float(((f_feat_clean.numpy() > 0).astype(int) == labels).mean())
        ot = off_target(acc_before, acc_after)
    else:  # no labels → report the clean-side decision shift magnitude
        ot = float(np.abs(f_removed.numpy() - f_feat_clean.numpy()).mean())

    rec = {
        **{k: meta.get(k) for k in ("model", "artifact", "rho", "opacity", "selection", "method")},
        "seed": seed, "layer": layer,
        "R_median": R_med, "R_ci_lo": lo, "R_ci_hi": hi,
        "cause": sel["cause"], "isolation": sel["isolation"], "selectivity": sel["selectivity"],
        "off_target": ot, "detection_auroc": detection_auroc,
        "e_in_median": float(np.median(e_in)), "e_in_abs_median": float(np.median(np.abs(e_in))),
        "coeff": coeff, "features": list(np.asarray(S).ravel().tolist()),
        "n_images": int(len(f_art)), "run_dir": str(run_dir) if run_dir else None,
        "git_commit": git_commit(),
    }

    if run_dir is not None:
        import pandas as pd
        tag = f"s{seed}_{meta.get('artifact')}_{meta.get('selection')}_{meta.get('method')}"
        pd.DataFrame({
            "f_art": f_art.numpy(), "f_removed": f_removed.numpy(),
            "f_feat_art": f_feat_art.numpy(), "f_feat_clean": f_feat_clean.numpy(),
            "e_in": e_in, "R": R, "label": labels if labels is not None else np.nan,
        }).to_parquet(f"{run_dir}/perimage_{tag}.parquet")
        save_json(f"{run_dir}/cell_{tag}.json", rec)
    return rec
