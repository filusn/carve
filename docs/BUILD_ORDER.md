# CARVE — Build Order (code-level)

This turns `docs/EXECUTION_PLAN.md` (phased research plan) into an **ordered coding
checklist** against the *current* repo state. Read `CLAUDE.md`, `docs/EXECUTION_PLAN.md`,
`docs/METRICS.md`, and `docs/INTEGRITY.md` first — this file does not restate them, it
sequences the implementation. Primary FM = **MONET** (CLIP ViT-L/14); data = ISIC 2018 +
Atlas (Bissoto-labelled), task = melanoma-vs-nevus.

## Repo state snapshot (2026-07-02)

- **Implemented:** `carve/utils.py` only (device/seed/run-dir/`assert_disjoint`).
- **Specs, not code:** every subpackage `__init__.py` holds a function-level spec —
  implement against it. `scripts/01_smoke_test.py` (Phase-0 gate) and
  `scripts/10_inject.py` (injector visual check) are runnable skeletons.
- **Tests are the contract (TDD):** `tests/test_injection.py` and `tests/test_splits.py`
  already import `carve.data.artifacts` / `carve.data.datasets`, which **do not exist yet**.
  Making those tests green is the definition of done for the data layer.
- **Not started:** all of `src/carve/` except utils; the `src/carve/data/` package.
- **Env:** **Docker** (not venv, per PI). Base image `research-base:2026-07`
  (`docker/base.Dockerfile`, CUDA 12.4 + torch, GPU passthrough verified). Persistent dev
  container `carve-dev` (`--gpus all`, repo at `/workspace`, `PYTHONPATH=/workspace/src`,
  `HF_HOME=/workspace/weights/hf`) with `requirements.txt` installed once; iterate via
  `docker exec carve-dev ...`. `docker-run.sh` remains for one-off ephemeral runs.
  GPU = RTX 4080 Laptop (**12 GB**) — fine for MONET *inference* + SAE training on
  **cached** activations (cache to disk; don't hold model + SAE-training together).

## The gating unknown (resolve before the SAE arm)

The intervention design assumes we can **read and write** MONET's ViT layer-ℓ activations
via hooks (ViT-Prisma `HookedViT`, else raw PyTorch forward/backward hooks). Confirm this
first with the MONET hook probe (Stage 1 below). If it fails, the intervention API changes.

---

## Build order

Two independent arms after the data layer lands (matches EXECUTION_PLAN parallelization):
**Arm A = FM + bias measurement**, **Arm B = SAE**. Critical path to a *reportable* result
is Stage 0 → 2 → 3 → the Phase-0 GO/NO-GO gate.

### Stage 0 — Environment bootstrap (Docker)
- [x] Base image `research-base:2026-07` present; GPU passthrough verified
      (`torch.cuda.is_available() == True`, RTX 4080 Laptop).
- [ ] Persistent `carve-dev` container up with `requirements.txt` installed;
      `open_clip / vit_prisma / crp` import cleanly.
- [ ] `docker exec carve-dev pytest -q` → injection/splits tests **fail on import**.
- [ ] Confirm MONET weights reachable (HF `suinleelab/monet` / MONET repo).

### Stage 1 — MONET hook probe  ✅ **DONE (GO, 2026-07-02)**
- [x] `scripts/02_monet_hook_probe.py`: loads MONET (HF `chanwkim/monet`, CLIP ViT-L/14),
      reads + writes residual stream via HF forward hooks; write moves the zero-shot margin.
- [x] Logged: 24 vision blocks, hidden **1024**, residual `[B,257,1024]` → SAE input dim
      1024; `layer_sweep [4,6,8,10]` valid; **HF forward hooks (no Prisma dependency)**.
- **Gate PASSED:** activations readable AND writable; decision controllable at ℓ.

### Stage 2 — Data layer (unblocks both arms)
- [x] `src/carve/data/artifacts.py` → **green** `tests/test_injection.py` (17/17).
      `ARTIFACT_KINDS`, `inject`, `remove`, `make_biased_set`. Contact sheet verified.
- [x] `src/carve/data/datasets.py` → **green** `tests/test_splits.py` (9/9).
      `make_splits`, `assign_artifact_presence`, `realized_rho`.
- [x] `src/carve/data/isic.py` (`load_isic_binary`) — ISIC-2018 Task3 (HAM10000) mel-vs-nevus
      loader over locally-symlinked data; **green** `tests/test_isic.py` (data-gated).
      Canvas RESOLVED = HAM10000 (no Bissoto labels locally); clean-pool count moot — see
      LOG. Container bind-mounts `/home/barmon/ceai/datasets` (ro) so symlinks resolve.
      Real-data check: loader→injector→MONET runs; injected ruler shifts zero-shot margin.
- **Gate:** `pytest tests/test_injection.py tests/test_splits.py` green ✅;
      `python scripts/10_inject.py` yields a plausible contact sheet ✅.

### Stage 3 — Arm A: FM + bias measurement
- [x] `src/carve/models/encoders.py`: `MonetEncoder`/`load_encoder` (frozen, hookable;
      `.activations(layer)`, `.zero_shot_margin()`, `.hooks()`), `pick_layer` (CV-AUROC
      layer sweep). **Green** `tests/test_encoders.py` (real MONET). NOTE: opaque α=1.0
      ruler is AUROC 1.0 at every layer → layer rule needs graded opacity + a tie-break
      (pre-register). Detection≠control.
- [x] `src/carve/models/probe.py`: `train_probe` (class-weighted linear), `f_decision` =
      logit margin z_pos−z_neg (probe=None → **zero-shot arm**), `probe_accuracy`.
- [x] `src/carve/metrics/causal.py` + `stats.py`: `bias_gap`, `input_effect`,
      `causal_recovery`, `selectivity`, `off_target`, `detection_auroc`, `steering_frontier`,
      bootstrap CIs. **Green** `tests/test_metrics.py` (toy tensors); full suite 48/0.
- [x] **Gate = Phase-0 GO/NO-GO** (`scripts/01_smoke_test.py`): **GO (2026-07-03)**. All 3
      artifacts significant in both arms — zero-shot e_in CI≠0 (ruler −0.51→nevus,
      marker_ink +0.96→mel, dark_corner +0.45→mel); induced bias_gap ≈1.0 (saturated at
      α=1/ρ=1). Bias is directional & artifact-specific. SAE-recovery half → Stage 4.

### Stage 4 — Arm B: SAE
- [ ] `src/carve/sae/load_sae.py`: Prisma pretrained CLIP SAE (dev path for CLIP generality).
- [ ] `src/carve/sae/train_sae.py`: TopK SAE on cached MONET layer-ℓ activations;
      `sae_health`. Set `sae.width`/`sae.k` → PREREGISTRATION.
- [ ] `src/carve/sae/discovery.py`: `select_oracle` (top detection-AUROC on `select`) vs
      `discover_unsupervised`. Enforce select/eval disjointness.

### Stage 5 — Interventions & core results
- [ ] `src/carve/interventions/hooks.py`: `ablate`, `steer` (input unchanged).
- [ ] `src/carve/interventions/mediation.py`: `input_effect`, `feature_effect`, `f_removed`
      → causal recovery R (drop to baseline = causal; stays high = mere correlation) +
      the clean-image off-target check.
- [ ] `src/carve/eval/harness.py`: `run_cell` → `RunRecord` into `experiments/runs/`
      (single owner; schema-drift risk).

### Stage 6 — Baselines, aggregation, figures
- [ ] `src/carve/baselines/*` (raw_neuron, cav, cdep, dermfmzero_suppress, random_ctrl,
      input_oracle) — same interface, same eval split.
- [ ] `src/carve/eval/aggregate.py` + `figures.py`; real-artifact slice.
- [ ] **Freeze `PREREGISTRATION.md` before the final grid.**

## Critical path

```
Stage0 ─▶ Stage1(hook probe)
                 └─▶ Stage2(data) ─┬─▶ Stage3(FM+bias) ─┐
                                   └─▶ Stage4(SAE) ──────┴─▶ Stage5 ─▶ Stage6
```

Deadline MI4MedFM **2026-07-15**. Hit the Stage-3 GO/NO-GO gate first — a documented
NO-GO ("SAEs fail to causally validate injected artifacts") is itself a Track-1 result.
Metric code (Stage 3, toy-tensor-testable) can be written now, before any model loads.
