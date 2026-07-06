"""carve.data — dataset loaders, seeded disjoint splits, controlled artifact injection.

Modules:
    datasets   make_splits (seeded, pairwise-disjoint partition), assign_artifact_presence
               (the spurious-correlation ρ knob), realized_rho (audit the realized bias).
    artifacts  inject/remove controlled artifacts (ruler/marker_ink/dark_corner) with masks,
               make_biased_set (materialize a ρ-biased set with per-image metadata).
    isic       load_isic_binary — ISIC-2018 Task3 (HAM10000) mel-vs-nevus loader over the
               locally-provisioned (symlinked, git-ignored) data. No download needed here.

CPU-only, numpy-based, deterministic per seed — see tests/test_splits.py and
tests/test_injection.py for the exact contracts these implement.
"""
