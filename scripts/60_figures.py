#!/usr/bin/env python
"""Phase-7: regenerate the headline figures from committed run dirs (never from memory).
Reads every cell_*.json under --runs (default: the whole runs dir, so it pools the Phase-5
interventions grid and the Phase-6 baselines grid onto shared axes) and writes PNGs.

    docker exec carve-dev python3 scripts/60_figures.py                 # all runs → figures/
    docker exec carve-dev python3 scripts/60_figures.py --runs <dir>    # one run dir
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from carve.eval import figures  # noqa: E402
from carve.utils import load_config  # noqa: E402


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(str(root / "configs" / "default.yaml"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=cfg.paths.runs_dir, help="run dir(s) to aggregate")
    ap.add_argument("--out", default=str(root / "figures"), help="output dir for PNGs")
    args = ap.parse_args()

    paths = figures.make_all(args.runs, args.out)
    print(f"[figures] aggregated {args.runs}")
    for p in paths:
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
