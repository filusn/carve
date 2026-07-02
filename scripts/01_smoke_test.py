#!/usr/bin/env python
"""CARVE Phase-0 day-1 go/no-go smoke test.

Purpose (see docs/EXECUTION_PLAN.md Phase 0): in the EASIEST setting — one artifact
(ruler), rho=1.0, opaque — confirm there is something to recover and that an SAE feature
can begin to recover it, BEFORE building the full grid.

This file ships as a runnable skeleton: it checks the environment and prints the exact
go/no-go procedure. Replace each TODO with calls into the implemented package, then the
script should print GO or a documented NO-GO/pivot.
"""
from __future__ import annotations

import importlib


def _check_imports():
    status = {}
    for mod in ["torch", "vit_prisma", "open_clip", "crp", "sklearn", "omegaconf"]:
        try:
            importlib.import_module(mod)
            status[mod] = "ok"
        except Exception as e:  # noqa: BLE001
            status[mod] = f"MISSING ({type(e).__name__})"
    return status


def main():
    print("=" * 72)
    print("CARVE — Phase-0 day-1 go/no-go smoke test")
    print("=" * 72)

    print("\n[env] optional dependency check:")
    for k, v in _check_imports().items():
        print(f"   - {k:12s} {v}")

    try:
        from carve.utils import device, load_config

        cfg = load_config("configs/default.yaml")
        print(f"\n[env] device = {device(cfg.get('device', 'auto'))}")
        print(f"[env] task   = {cfg.dataset.name}/{cfg.dataset.task}")
    except Exception as e:  # noqa: BLE001
        print(f"\n[env] carve/config not ready yet: {e}")

    print(
        "\n[procedure] implement and run these steps (EXECUTION_PLAN Phase 0):\n"
        "   1. inject ruler @ rho=1.0, opacity=1.0 on a small subset      # carve.data.artifacts\n"
        "   2. train probe; measure BIAS GAP + input effect e_in          # carve.models.probe\n"
        "        -> if bias_gap ~ 0: injection invalid, FIX before going on\n"
        "   3. load/train SAE on layer ℓ                                  # carve.sae\n"
        "   4. detection AUROC of best feature for ruler present/absent   # carve.sae.discovery\n"
        "   5. ablate that feature; causal_recovery R vs input removal    # carve.interventions\n"
    )
    print(
        "[decision]\n"
        "   GO    if clear bias gap AND a feature with high detection AUROC whose ablation\n"
        "         yields nonzero recovery  -> proceed to Phases 1-7.\n"
        "   NO-GO if even here the SAE finds nothing / ablation does nothing -> this is a\n"
        "         FINDING: reframe as the negative-result benchmark (still Track 1). Tell PI.\n"
    )
    print("[status] skeleton only — wire in the package calls above, then re-run.\n")


if __name__ == "__main__":
    main()
