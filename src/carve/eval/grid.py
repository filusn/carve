"""Shared scaffolding for the phase-5/6 grid runners (scripts/40, scripts/50).

Both grid scripts share a large amount of near-identical, BEHAVIOR-PRESERVING setup:
config load + run dir + resolving image_size/layer/seeds, pushing the FULL/QUICK size dict
into ``cfg.sae``, and — per seed — seeding, disjoint splits, index caps, loading images,
training the SAE ONCE, computing the mean-activation baseline vector, and exposing the
select/eval tensors. The method-specific loop (which interventions/baselines each script
runs per artifact) stays in the scripts; only this scaffolding lives here.

Public API
    setup_run(cfg, name, N)              -> (run, layer, size)
    resolve_seeds(cfg, quick)            -> list[int]
    iter_seed_contexts(cfg, ds, enc, N, seeds, layer, size) -> Iterator[SeedContext]
    finish_grid(run, records, health_by_seed, title=...)    -> DataFrame
    SeedContext                          (dataclass carrying the per-seed tensors)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from ..data.datasets import make_splits
from ..utils import assert_disjoint, new_run_dir, save_json, set_seed

# Columns printed in the mean±std benchmark table (shared by both scripts).
BENCHMARK_COLS = [
    "artifact", "selection", "method", "detection_auroc_mean",
    "R_median_mean", "R_median_std", "selectivity_mean", "off_target_mean",
]


@dataclass
class SeedContext:
    """Everything a grid script needs for ONE seed, after the shared per-seed setup.

    The SAE is trained once per seed (artifact-agnostic; injected artifacts appear only in
    select/eval, never in sae_train) and reused across all artifacts. ``baseline_vec`` is
    the mean-activation vector (mean-ablation target for raw-neuron baselines); ``d`` is the
    activation dimension. ``sel_imgs``/``sel_labels`` feed make_biased_set on the *select*
    split; ``clean``/``labels``/``ev_idx`` are the DISJOINT *eval* split.
    """

    seed: int
    sae: Any
    health: dict
    sel_imgs: list
    sel_labels: Any
    clean: list
    labels: Any
    ev_idx: Any
    baseline_vec: Any
    d: int


def setup_run(cfg: Any, name: str, N: dict) -> tuple[Path, int, int]:
    """Create the run dir and resolve image size / layer, pushing the size dict into cfg.

    Mirrors the head of both scripts: ``new_run_dir(...)``, read ``image_size`` and the SAE
    ``layer``, then set ``cfg.sae.width/k/train.steps`` from the FULL/QUICK dict ``N``.
    """
    run = new_run_dir(cfg.paths.runs_dir, name, cfg, int(cfg.get("seed", 0)))
    size = int(cfg.dataset.image_size)
    layer = int(cfg.sae.get("layer", 12))
    cfg.sae.width, cfg.sae.k, cfg.sae.train.steps = N["width"], N["k"], N["steps"]
    return run, layer, size


def resolve_seeds(cfg: Any, quick: bool) -> list[int]:
    """One seed under ``--quick``, else the configured grid (default [0, 1, 2])."""
    return [0] if quick else list(cfg.get("seeds", [0, 1, 2]))


def iter_seed_contexts(
    cfg: Any, ds: Any, enc: Any, N: dict, seeds: list[int], layer: int, size: int,
) -> Iterator[SeedContext]:
    """Yield a ready :class:`SeedContext` per seed, doing the shared per-seed setup.

    Per seed: ``set_seed`` → disjoint splits → cap the sae_train/select/eval indices →
    load the SAE-train images and train the SAE once → ``sae_health`` (+ printed health
    line) → mean-activation baseline vector → load the select/eval images. Heavy SAE
    imports are deferred so this module stays importable without a GPU/MONET.
    """
    from ..sae.train_sae import sae_health, train_sae

    for seed in seeds:
        set_seed(seed)
        splits = make_splits(ds.labels, dict(cfg.dataset.splits), seed=seed, stratify=True)
        assert_disjoint(**{k: set(v.tolist()) for k, v in splits.items()})
        sae_idx = splits["sae_train"][: N["sae_train"]]
        sel_idx = splits["select"][: N["select"]]
        ev_idx = splits["eval"][: N["eval"]]

        print(f"\n[seed {seed}] training SAE ...")
        acts = enc.activations([ds.load_image(int(i), size=size) for i in sae_idx], layer, pool=None)
        acts = acts.reshape(-1, acts.shape[-1])
        sae = train_sae(acts, cfg, seed=seed)
        health = sae_health(sae, acts[int(0.8 * len(acts)):])
        print(f"[seed {seed}] SAE R²={health['r2']:.3f}  dead={health['dead_feature_frac']*100:.1f}%")
        baseline_vec = acts.mean(dim=0).cpu().numpy()   # mean-ablation target for raw neurons
        d = acts.shape[-1]

        sel_imgs = [ds.load_image(int(i), size=size) for i in sel_idx]
        clean = [ds.load_image(int(i), size=size) for i in ev_idx]
        labels = ds.labels[ev_idx]
        sel_labels = ds.labels[sel_idx]

        yield SeedContext(
            seed=seed, sae=sae, health=health, sel_imgs=sel_imgs, sel_labels=sel_labels,
            clean=clean, labels=labels, ev_idx=ev_idx, baseline_vec=baseline_vec, d=d,
        )


def finish_grid(
    run: Path, records: list, health_by_seed: dict,
    title: str = "benchmark table (mean±std over seeds)",
):
    """Write ``metrics.json``, build/print the mean±std benchmark table, return it.

    Reads the per-cell json back through :func:`carve.eval.aggregate.benchmark_table`
    (never from memory — INTEGRITY.md), selects the shared headline columns, prints the
    table under ``title`` and the run path, and returns the DataFrame.
    """
    from .aggregate import benchmark_table

    save_json(run / "metrics.json", {"health_by_seed": health_by_seed, "records": records})
    tbl = benchmark_table(run)
    cols = [c for c in BENCHMARK_COLS if c in tbl.columns]
    print(f"\n=== {title} ===")
    print(tbl[cols].to_string(index=False))
    print(f"\n[run] {run}")
    return tbl
