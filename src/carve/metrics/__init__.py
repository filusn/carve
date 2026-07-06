"""Metrics — exact definitions in docs/METRICS.md. Unit-test on toy tensors first.

Signatures (do NOT redefine after PREREGISTRATION is frozen):

causal.py
    bias_gap(acc_aligned, acc_conflicting) -> float                       # METRICS §1
    input_effect(f_art, f_removed) -> Tensor                              # METRICS §2  e_in
    causal_recovery(f_art, f_feat, f_removed, eps) -> Tensor              # METRICS §3  R, [-,1]
        # R = 1 - |f_feat - f_removed| / |f_art - f_removed| ; skip |e_in|<eps
    selectivity(f_art, f_feat_art, f_clean, f_feat_clean) -> dict
        # returns {cause, isolation, selectivity}                         # METRICS §4
    off_target(acc_clean_before, acc_clean_after) -> float                # METRICS §5
    detection_auroc(feature_scores, artifact_present) -> float            # METRICS §6
    steering_frontier(cause_by_coeff, off_target_by_coeff) -> ndarray     # METRICS §7

stats.py
    bootstrap_ci(values, n=1000, ci=0.95, rng=None, statistic=median) -> (lo, hi)
    paired_bootstrap_diff(a, b, n=1000, ci=0.95, rng=None) -> (mean_diff, lo, hi)
        # never claim a win whose CI overlaps 0
"""

from . import causal, stats  # noqa: F401
