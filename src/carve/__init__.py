"""CARVE — Causal ARtifact Validation of Encodings.

A benchmark for the causal faithfulness of sparse-autoencoder interpretability in
medical foundation models. See docs/EXECUTION_PLAN.md, docs/METRICS.md, docs/INTEGRITY.md.

Subpackages:
    data          load datasets, build seeded disjoint splits, inject controlled artifacts
    models        load (frozen, hookable) encoders + train probes
    sae           train/load sparse autoencoders, discover candidate artifact features
    interventions activation-level ablation/steering hooks + input-vs-feature mediation
    baselines     raw-neuron, CAV/Reveal2Revise, CDEP, random-feature, input-removal oracle
    metrics       causal recovery, selectivity, off-target, detection AUROC, bias gap
    eval          run harness + aggregation into benchmark tables
"""

__version__ = "0.0.1"

from . import utils  # noqa: F401
