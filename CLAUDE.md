# CARVE — agent operating instructions

**CARVE = Causal ARtifact Validation of Encodings.**
A benchmark that tests whether sparse-autoencoder (SAE) feature interpretability and
steering on a **medical foundation model** can correctly **recover and control a known,
injected visual artifact's causal effect** — and whether it beats incumbent methods
(CAV / Reveal2Revise, CDEP, raw-neuron ablation).

- **Author / PI:** Filip Noworolnik (AGH Kraków). PhD thesis spine: *reliable visual
  decision support in expert domains*. This is a Pillar-3 (reliability/interpretability)
  contribution with a Pillar-1 (data-centric/synthetic) method.
- **Submission target:** MI4MedFM workshop @ MICCAI 2026 — **deadline 2026-07-15 (AoE)**,
  **Track 1 = 8–10 pp LNCS, archival**. Fallback: Track 2 = 4 pp non-archival.
- **Hard data rule:** **public + synthetic data only.** No private/clinical data, no new
  expert annotation. (Do not pull from iCardio or any non-public source.)

> Read `docs/EXECUTION_PLAN.md` before doing anything. Read `docs/INTEGRITY.md` and treat
> it as binding. `docs/METRICS.md` defines every number you will report.

---

## Prime directives (non-negotiable)

1. **Never fabricate or "fill in" results.** Every number in a table or the paper must
   come from a script that produced it, with a saved run directory and a git commit.
   If an experiment didn't run, say so. A missing result is fine; an invented one is misconduct.
2. **Never invent citations.** Every reference must be verified against the real
   paper before it is cited. `docs/RELATED_WORK.md` marks which are VERIFIED vs
   VERIFY-BEFORE-CITING. Do not upgrade a status without checking the actual paper.
3. **Pre-register before final runs.** `PREREGISTRATION.md` (metrics, thresholds,
   model/layer-selection rule) must be filled and committed **before** the final
   experiment grid is launched. No changing metrics after seeing results (no HARKing).
4. **No split leakage.** The split used to *select* the artifact feature must be disjoint
   from the split used to *evaluate* its causal effect, which must be disjoint from probe
   training. Enforce and assert this in code.
5. **Fair baselines.** Implement CAV/Reveal2Revise and CDEP from the authors' methods,
   tuned with the same budget as our method. Report honestly when a baseline wins.
6. **Negative results are valid and expected.** "SAE steering does not reliably recover
   the known artifact" is a publishable, honest CARVE result. Do not torture data to
   avoid it.
7. **Determinism.** Set and log seeds; report mean ± std over ≥3 seeds for headline numbers.
8. **Stay in scope.** Dermatology only for the MVP. Do not add CXR / a second modality
   until the derm pipeline is complete and the deadline buffer allows it. Ask before
   scope changes.

When in doubt about any of the above, **stop and ask the PI** rather than guessing.

---

## Environment & setup

- Python **3.10+**, managed with **`uv`**. Real runs need a **CUDA GPU** (≈16–24 GB).
  Apple-Silicon/MPS and CPU are for tiny smoke tests only.
- Setup:
  ```bash
  uv venv && source .venv/bin/activate
  uv pip install -r requirements.txt
  python scripts/01_smoke_test.py   # day-1 go/no-go, see EXECUTION_PLAN Phase 0
  ```
- Device selection must be automatic: `cuda` → `mps` → `cpu`. Never hardcode `cuda`.
- All large artifacts (datasets, weights, SAE checkpoints, run outputs) live under
  paths in `configs/default.yaml` and are **git-ignored**. Never commit data or weights.

## Repository conventions

- Code is **config-driven**. No magic constants in scripts — read `configs/*.yaml`.
- Package: `src/carve/...`. Each subpackage `__init__.py` documents the functions it
  must expose; implement against those specs.
- Numbered `scripts/NN_*.py` are the pipeline entrypoints; they orchestrate, the package
  holds the logic.
- Every run writes to `experiments/runs/<timestamp>_<name>/` containing: the resolved
  config, git commit hash, seed, logs, raw per-image outputs (parquet/csv), and the
  computed metrics (json). Aggregation reads from there — never from memory.
- Tests in `tests/` (pytest). At minimum: artifact-injection determinism, hook
  ablation correctness, split-disjointness assertions, metric unit tests on toy inputs.

## Git discipline

- Branch per phase (`phase1-injection`, etc.). Small, labelled commits.
- Commit the **config + commit-hash + metrics json** alongside any result you report.
- Do **not** push or open PRs unless the PI asks.
- Commit message footer: see the PI's global convention if pushing; otherwise local commits.

## What to do / not do

- ✅ Do: implement to the specs, run the smoke test first, log everything, report
  variance, flag surprises, keep a `docs/LOG.md` of decisions.
- ❌ Don't: change metric definitions mid-flight, cite unverified papers, commit data,
  add modalities/datasets not in the plan, claim clinical validity, or report a number
  you can't regenerate from a committed run.

## Pointers

- `docs/EXECUTION_PLAN.md` — the phased, task-level plan (start here).
- `docs/METRICS.md` — exact definitions of every metric.
- `docs/INTEGRITY.md` — the research-integrity protocol (binding).
- `docs/RELATED_WORK.md` — references + novelty framing + the two killer objections.
- `docs/DATASETS.md` — datasets, licenses, how to obtain.
- `PREREGISTRATION.md` — fill and freeze before final runs.
