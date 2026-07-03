"""Figures for the CARVE benchmark (Phase 7) — all regenerated from committed run dirs via
carve.eval.aggregate (never from memory; INTEGRITY.md). matplotlib is imported lazily so
importing carve.eval stays dependency-light.

  * detection_vs_recovery  — THE headline: detection AUROC (x) vs causal recovery R (y).
                             Points hug high-detection / low-R ⇒ "detect but don't control".
  * recovery_bars          — R_median per artifact, grouped by method, with cross-seed std.
  * selectivity_vs_offtarget — the control frontier: selectivity (y) vs off-target damage (x).
  * detection_bars         — detection AUROC per method (sanity: everything detects well).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .aggregate import benchmark_table, load_records

# stable, colour-blind-friendly method styling
_METHOD_STYLE = {
    "sae_ablate": ("CARVE SAE (oracle ablate)", "#0072B2", "o"),
    "sae_steer_c2.0": ("CARVE SAE (steer c=2)", "#56B4E9", "^"),
    "sae_ablate_topk": ("DermFM-Zero (top-k)", "#009E73", "s"),
    "raw_ablate": ("Raw neuron", "#E69F00", "D"),
    "input_remove": ("Input-removal oracle", "#000000", "*"),
}


def _label(method: str) -> str:
    return _METHOD_STYLE.get(method, (method, "#999999", "x"))[0]


def _style(method: str):
    return _METHOD_STYLE.get(method, (method, "#999999", "x"))


def _fig(out: str | Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    return plt


def detection_vs_recovery(runs_dir, out: str | Path):
    """Headline dissociation: per (artifact, method), detection AUROC vs causal recovery R."""
    plt = _fig(out)
    df = load_records(runs_dir)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for method, g in df.groupby("method"):
        name, colour, marker = _style(method)
        d = g["detection_auroc"].to_numpy(dtype=float)
        r = g["R_median"].to_numpy(dtype=float)
        ok = ~np.isnan(d)
        if ok.any():
            ax.scatter(d[ok], r[ok], c=colour, marker=marker, s=70, label=name,
                       edgecolors="k", linewidths=0.4, alpha=0.9, zorder=3)
    ax.axhline(0.0, color="grey", lw=0.8, ls=":")
    ax.axhline(1.0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("detection AUROC  (does a feature/neuron SEE the artifact?)")
    ax.set_ylabel("causal recovery R  (does ablating it UNDO the effect?)")
    ax.set_title("Detection ≠ control")
    ax.set_ylim(-0.3, 1.15)
    ax.legend(fontsize=8, loc="center left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def recovery_bars(runs_dir, out: str | Path):
    """Grouped bars: median causal recovery R per artifact × method, ±cross-seed std."""
    plt = _fig(out)
    tbl = benchmark_table(runs_dir)
    artifacts = sorted(tbl["artifact"].unique())
    methods = [m for m in _METHOD_STYLE if m in set(tbl["method"])]
    methods += [m for m in sorted(tbl["method"].unique()) if m not in methods]
    x = np.arange(len(artifacts))
    w = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for j, method in enumerate(methods):
        name, colour, _ = _style(method)
        means, stds = [], []
        for a in artifacts:
            row = tbl[(tbl["artifact"] == a) & (tbl["method"] == method)]
            means.append(float(row["R_median_mean"].iloc[0]) if len(row) else np.nan)
            stds.append(float(row["R_median_std"].iloc[0]) if len(row) else 0.0)
        ax.bar(x + j * w, means, w, yerr=stds, capsize=2, color=colour, label=name,
               edgecolor="k", linewidth=0.3)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(artifacts)
    ax.set_ylabel("causal recovery R (median, ±std over seeds)")
    ax.set_title("Causal recovery by method")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def selectivity_vs_offtarget(runs_dir, out: str | Path):
    """Control frontier: selectivity (want high) vs off-target damage (want ~0), per method."""
    plt = _fig(out)
    tbl = benchmark_table(runs_dir)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for method, g in tbl.groupby("method"):
        name, colour, marker = _style(method)
        ax.scatter(g["off_target_mean"], g["selectivity_mean"], c=colour, marker=marker,
                   s=70, label=name, edgecolors="k", linewidths=0.4, alpha=0.9, zorder=3)
    ax.set_xlabel("off-target damage (clean-task, lower better)")
    ax.set_ylabel("selectivity (higher better)")
    ax.set_title("Control frontier")
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def detection_bars(runs_dir, out: str | Path):
    """Detection AUROC per artifact × method (sanity: features/neurons detect artifacts well)."""
    plt = _fig(out)
    tbl = benchmark_table(runs_dir)
    tbl = tbl[~tbl["detection_auroc_mean"].isna()]
    artifacts = sorted(tbl["artifact"].unique())
    methods = sorted(tbl["method"].unique())
    x = np.arange(len(artifacts))
    w = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for j, method in enumerate(methods):
        name, colour, _ = _style(method)
        vals = []
        for a in artifacts:
            row = tbl[(tbl["artifact"] == a) & (tbl["method"] == method)]
            vals.append(float(row["detection_auroc_mean"].iloc[0]) if len(row) else np.nan)
        ax.bar(x + j * w, vals, w, color=colour, label=name, edgecolor="k", linewidth=0.3)
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax.set_ylim(0.4, 1.02)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(artifacts)
    ax.set_ylabel("detection AUROC")
    ax.set_title("Artifact detectability")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def make_all(runs_dir, out_dir: str | Path) -> list:
    """Regenerate every headline figure into out_dir. Returns the written paths."""
    out_dir = Path(out_dir)
    return [
        detection_vs_recovery(runs_dir, out_dir / "detection_vs_recovery.png"),
        recovery_bars(runs_dir, out_dir / "recovery_bars.png"),
        selectivity_vs_offtarget(runs_dir, out_dir / "selectivity_vs_offtarget.png"),
        detection_bars(runs_dir, out_dir / "detection_bars.png"),
    ]
