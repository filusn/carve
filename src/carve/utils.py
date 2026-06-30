"""Shared utilities: device selection, seeding, config, run-dir bookkeeping.

These are implemented (not stubs) so every script can produce a reproducible run dir
exactly as INTEGRITY.md requires: config + git commit + seed + outputs in one place.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def device(pref: str = "auto") -> str:
    """Return 'cuda' > 'mps' > 'cpu' (or an explicit preference). Never hardcode 'cuda'."""
    import torch

    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    """Seed python, numpy, torch (CPU+CUDA) for determinism. Log the seed into the run dir."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> Any:
    """Load a YAML config via OmegaConf (supports CLI overrides downstream)."""
    return OmegaConf.load(str(path))


def git_commit() -> str:
    """Current git commit hash (or 'nogit'). Recorded in every run dir for provenance."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"


def new_run_dir(runs_dir: str | Path, name: str, cfg: Any, seed: int) -> Path:
    """Create experiments/runs/<utc>_<name>/ and write resolved config + provenance.

    Returns the run dir. Per-image outputs (parquet) and metrics (json) are written here
    by the caller. Aggregation reads from these dirs, never from memory (INTEGRITY.md).
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = Path(runs_dir) / f"{ts}_{name}"
    run.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run / "config.yaml")
    save_json(run / "provenance.json", {"git_commit": git_commit(), "seed": seed, "utc": ts})
    return run


def save_json(path: str | Path, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def assert_disjoint(**splits: set) -> None:
    """Assert all named index splits are pairwise-disjoint (no leakage). Use in data code."""
    names = list(splits)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = splits[names[i]], splits[names[j]]
            overlap = set(a) & set(b)
            assert not overlap, f"split leakage: {names[i]} ∩ {names[j]} = {len(overlap)} items"
