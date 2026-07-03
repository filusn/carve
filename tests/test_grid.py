"""Unit tests for the shared grid scaffolding in carve.eval.grid.

Pure helpers only — no MONET, no GPU. ``iter_seed_contexts`` (which trains a real SAE) is
NOT exercised here; we cover ``setup_run`` (run dir + resolved layer/size + size-dict push),
``resolve_seeds`` (the --quick flag), and ``finish_grid`` (metrics.json + printed table built
from synthetic cell_*.json).
"""
import json

from omegaconf import OmegaConf

from carve.eval.grid import finish_grid, resolve_seeds, setup_run

FULL = dict(sae_train=1200, select=300, eval=250, width=16384, k=32, steps=3000)
QUICK = dict(sae_train=400, select=200, eval=120, width=4096, k=32, steps=800)


def _cfg(runs_dir):
    return OmegaConf.create({
        "seed": 0,
        "seeds": [0, 1, 2],
        "paths": {"runs_dir": str(runs_dir)},
        "dataset": {"image_size": 224},
        "sae": {"layer": 12, "width": 1, "k": 1, "train": {"steps": 1}},
    })


def test_setup_run_creates_dir_and_pushes_size_dict(tmp_path):
    cfg = _cfg(tmp_path / "runs")
    run, layer, size = setup_run(cfg, "interventions_grid", FULL)

    assert run.exists() and run.is_dir()
    assert run.name.endswith("_interventions_grid")
    assert layer == 12
    assert size == 224
    # size dict pushed into cfg.sae
    assert (cfg.sae.width, cfg.sae.k, cfg.sae.train.steps) == (16384, 32, 3000)
    # provenance + resolved config written by new_run_dir
    assert (run / "config.yaml").exists()
    assert (run / "provenance.json").exists()


def test_setup_run_defaults_layer_when_missing(tmp_path):
    cfg = _cfg(tmp_path / "runs")
    del cfg.sae.layer
    _run, layer, _size = setup_run(cfg, "baselines_grid", QUICK)
    assert layer == 12
    assert (cfg.sae.width, cfg.sae.k, cfg.sae.train.steps) == (4096, 32, 800)


def test_resolve_seeds():
    cfg = _cfg("unused")
    assert resolve_seeds(cfg, quick=True) == [0]
    assert resolve_seeds(cfg, quick=False) == [0, 1, 2]
    # falls back to [0, 1, 2] when unset
    assert resolve_seeds(OmegaConf.create({}), quick=False) == [0, 1, 2]


def _write_cell(run, seed, artifact, method, rec):
    (run / f"cell_s{seed}_{artifact}_{method}.json").write_text(json.dumps(rec))


def _mk_record(seed, artifact, method, det, r, sel, ot):
    return {
        "model": "monet", "artifact": artifact, "rho": 0.9, "opacity": 1.0,
        "selection": "oracle", "method": method, "seed": seed, "layer": 12,
        "R_median": r, "selectivity": sel, "off_target": ot, "detection_auroc": det,
        "n_images": 50,
    }


def test_finish_grid_writes_metrics_and_returns_table(tmp_path, capsys):
    run = tmp_path / "run"
    run.mkdir()
    records, health_by_seed = [], {}
    for seed in (0, 1):
        health_by_seed[seed] = {"r2": 0.9, "dead_feature_frac": 0.1}
        for artifact in ("ruler", "marker_ink"):
            rec = _mk_record(seed, artifact, "sae_ablate", 0.99, 0.03 + 0.01 * seed, 0.98, 0.01)
            records.append(rec)
            _write_cell(run, seed, artifact, "sae_ablate", rec)

    tbl = finish_grid(run, records, health_by_seed, title="unit test table")

    # metrics.json written with both keys
    saved = json.loads((run / "metrics.json").read_text())
    assert set(saved) == {"health_by_seed", "records"}
    assert len(saved["records"]) == len(records)
    assert set(saved["health_by_seed"]) == {"0", "1"}  # json stringifies int keys

    # mean±std aggregated over the 2 seeds, per (artifact, selection, method)
    assert len(tbl) == 2  # two artifacts, one method
    assert "R_median_mean" in tbl.columns
    assert "detection_auroc_mean" in tbl.columns

    out = capsys.readouterr().out
    assert "=== unit test table ===" in out
    assert str(run) in out
