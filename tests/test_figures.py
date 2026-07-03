"""Figures + aggregation smoke test on synthetic cell_*.json (no MONET/data).
Writes fake RunRecords, then checks aggregate + every figure regenerate without error."""
import json

import pytest

from carve.eval.aggregate import benchmark_table, load_records

pytest.importorskip("matplotlib")
from carve.eval import figures  # noqa: E402


def _write_cells(runs_dir):
    """Two seeds × 3 artifacts × 3 methods of plausible RunRecords."""
    methods = {  # method: (detection_auroc, R, selectivity, off_target)
        "sae_ablate": (0.99, 0.03, 0.98, 0.01),
        "raw_ablate": (0.95, 0.10, 0.90, 0.03),
        "input_remove": (float("nan"), 1.0, 1.0, 0.0),
    }
    for seed in (0, 1):
        d = runs_dir / f"run_seed{seed}"
        d.mkdir()
        for artifact in ("ruler", "marker_ink", "dark_corner"):
            for method, (det, r, sel, ot) in methods.items():
                rec = {
                    "model": "monet", "artifact": artifact, "rho": 0.9, "opacity": 1.0,
                    "selection": "oracle" if "sae" in method or method == "input_remove" else "raw_neuron",
                    "method": method, "seed": seed, "layer": 12,
                    "R_median": r + 0.01 * seed, "R_ci_lo": r - 0.1, "R_ci_hi": r + 0.1,
                    "cause": 1.0, "isolation": 1 - sel, "selectivity": sel,
                    "off_target": ot, "detection_auroc": det, "n_images": 50,
                }
                (d / f"cell_s{seed}_{artifact}_{method}.json").write_text(json.dumps(rec))


def test_aggregate_and_figures(tmp_path):
    _write_cells(tmp_path)

    df = load_records(tmp_path)
    assert len(df) == 2 * 3 * 3
    tbl = benchmark_table(tmp_path)
    assert {"R_median_mean", "R_median_std", "selectivity_mean"} <= set(tbl.columns)
    # input-removal oracle aggregates to full recovery
    orc = tbl[tbl["method"] == "input_remove"]["R_median_mean"]
    assert (orc > 0.99).all()

    paths = figures.make_all(tmp_path, tmp_path / "figs")
    assert len(paths) == 4
    for p in paths:
        assert p.exists() and p.stat().st_size > 0
