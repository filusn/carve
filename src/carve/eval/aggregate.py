"""Aggregate RunRecords across run dirs into the benchmark table (Phase 7).

Reads the per-cell ``cell_*.json`` written by carve.eval.harness.run_cell (never from
memory — INTEGRITY.md), concatenates them, and (optionally) averages headline metrics over
seeds. Figures live in carve.eval.figures.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .harness import RUNRECORD_COLUMNS


def load_records(runs_dir: str | Path) -> pd.DataFrame:
    """Load every cell_*.json under runs_dir/** into one DataFrame."""
    rows = [json.loads(p.read_text()) for p in Path(runs_dir).rglob("cell_*.json")]
    return pd.DataFrame(rows)


def benchmark_table(runs_dir: str | Path) -> pd.DataFrame:
    """Mean±std over seeds of the headline metrics per (model, artifact, ρ, opacity,
    selection, method)."""
    df = load_records(runs_dir)
    if df.empty:
        return df
    keys = ["model", "artifact", "rho", "opacity", "selection", "method"]
    metrics = ["R_median", "cause", "isolation", "selectivity", "off_target", "detection_auroc"]
    present = [k for k in keys if k in df.columns]
    agg = df.groupby(present)[[m for m in metrics if m in df.columns]].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()
