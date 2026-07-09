"""Eval — the run harness and aggregation. One agent should own this to avoid schema drift.

Implement (see EXECUTION_PLAN Phases 5-7):

harness.py
    run_cell(method, encoder, layer, probe, sae, loaders, cfg, seed) -> RunRecord
        # for ONE (model, artifact, rho, opacity, selection∈{oracle,discovered}, method):
        #   - compute per-image: f_art, f_removed, f_feat(ablate), f_feat(steer@coeffs)
        #   - compute metrics (carve.metrics) with bootstrap CIs
        #   - write per-image parquet + metrics json into carve.utils.new_run_dir(...)
        # METHODS share one interface: carve.interventions + carve.baselines plug in here.
    RunRecord schema (columns): model, artifact, rho, opacity, selection, method, seed,
        layer, R_median, R_ci_lo, R_ci_hi, cause, isolation, selectivity, off_target,
        detection_auroc, n_images, run_dir, git_commit

figures.py
    recovery_vs_rho(...) ; selectivity_vs_offtarget_scatter(...) ; detection_bars(...)

aggregate.py
    load_records(runs_dir) / benchmark_table(runs_dir) -> DataFrame   # read ALL run dirs
        # (never memory), build the headline table; mean±std over seeds.
"""

from .harness import RUNRECORD_COLUMNS, run_cell, run_steer_bestcase  # noqa: F401
from .aggregate import benchmark_table, load_records  # noqa: F401
