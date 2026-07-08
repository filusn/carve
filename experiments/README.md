# experiments/

- `runs/` — one dir per run, **git-ignored**. Each contains `config.yaml`,
  `provenance.json` (git commit + seed + utc), per-image outputs (`*.parquet`), and
  computed `metrics.json`. Created by `carve.utils.new_run_dir`. **Never hand-edit.**
- `tables/` — curated, committed result tables (CSV/markdown) produced by
  `scripts/60_aggregate.py` from `runs/`. These are what the paper cites.
- `figures/` — committed final figures.
- `artifact_examples/` — before/after injection previews, **git-ignored, not committed**.
  They are pixel derivatives of real ISIC-2018 / HAM10000 patient images (CC BY-NC-SA 4.0)
  and must not be redistributed. Regenerate locally with
  `python scripts/20_make_artifact_examples.py`.

Reproducibility rule (INTEGRITY.md): every number in the paper traces to a `runs/` dir
with its config + commit hash. Aggregation reads from `runs/`, never from memory.
