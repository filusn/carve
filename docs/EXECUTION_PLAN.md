# CARVE — Execution Plan

This is the task-level plan for executing CARVE end-to-end. Read `CLAUDE.md` and
`docs/INTEGRITY.md` first. Every metric here is defined precisely in `docs/METRICS.md`.

**Goal of the MVP paper (Track 1, MI4MedFM, 2026-07-15):** a reusable benchmark that, on
one medical foundation model and dermatology data, measures the **causal recovery,
selectivity, and off-target damage** of SAE-feature steering against injected artifacts,
compared head-to-head with CAV/Reveal2Revise, CDEP, DermFM-Zero-style top-k suppression, raw-neuron ablation, a
random-feature control, and the input-removal oracle — across **artifact type × correlation ρ × opacity**,
with multi-seed variance.

## The scientific design in one picture

```
                  inject artifact (ruler/marker/dark-corner/text)
                  at correlation ρ, opacity α            ┌─ INPUT-LEVEL intervention (GOLD):
 derm image ───────────────────────────────────────────►│   effect = f(x_art) − f(remove(x_art))
      │                                                  └─  = the artifact's TRUE causal effect
      ▼
 medical FM encoder (frozen) ──► activations at layer ℓ ──► classifier/probe head ──► logit
      │                                   │
      │                                   ▼
      │                            SAE (trained/loaded)
      │                                   │  candidate artifact feature(s) S
      ▼                                   ▼
 FEATURE-LEVEL intervention: ablate/steer S in activations, input UNCHANGED
      effect_feat = f(x_art) − f_{−S}(x_art)
      RECOVERY = how well ablating S reproduces removing the artifact from the input
```

If feature `S` truly mediates the artifact, `f_{−S}(x_art) ≈ f(remove(x_art))`. That
mediation comparison — only possible because we injected the artifact — is CARVE's core.

---

## Phase 0 — Setup & decisions (Day 0–1)

**Tasks**
- [ ] `uv venv`; install `requirements.txt`; verify CUDA. Device helper `carve.utils.device`.
- [ ] Confirm the three open decisions (record answers in `docs/LOG.md`):
  - **D1 — Base model.** Default: **primary = a dermatology/medical FM** (candidates:
    **MONET**, **BiomedCLIP**, or a derm-finetuned CLIP) **+ CLIP ViT-B/16 as a
    generality check** (ViT-Prisma ships pretrained SAEs for it → saves SAE-training time).
    Pick by: (a) does it expose ViT activations cleanly via HookedViT, (b) is it public,
    (c) does it actually do derm classification well. Document the choice + why.
  - **D2 — SAE source.** Reuse a pretrained CLIP SAE (Prisma) for method development;
    **train an SAE on the medical FM** for the headline result. TopK SAE.
  - **D3 — Track.** Default **Track 1 (8–10 pp, archival)**. Track 2 (4 pp) is the
    fallback only if Phase 5 slips past ~Jul 10.
- [ ] **Clean-pool count** (gates canvas choice): count all-flags-zero images in Bissoto's
  `isic_bias.csv`+`atlas_bias.csv`; if too small, canvas = HAM10000 and Bissoto → real-slice
  only. Log counts + decision (`docs/DATASETS.md`, `docs/LOG.md`).
- [ ] **Day-1 go/no-go smoke test** (`scripts/01_smoke_test.py`), see below. **Do not
  build the full grid until this passes.**

**Day-1 go/no-go (the single most important early check)**
Easiest possible setting — one artifact (ruler), ρ = 1.0, opaque:
1. Inject ruler so it perfectly predicts the label on a small subset.
2. **Zero-shot arm (no training):** paste/remove the ruler, measure the shift in frozen
   MONET's own mel-vs-nevus output (purest causal GT). **Induced arm:** train the probe and
   **confirm shortcut reliance** via the *bias gap*
   (accuracy on artifact-aligned vs artifact-conflicting test; see METRICS). If there is
   no bias gap, the model isn't using the artifact → fix injection before proceeding.
3. Load/train an SAE on layer ℓ; check whether **any** feature detects the ruler
   (detection AUROC) and whether **ablating it** moves the logit toward the
   artifact-removed prediction (nonzero recovery).
- **GO** if there is a clear bias gap *and* at least one feature with high detection AUROC
  whose ablation yields nonzero recovery → scale up (Phases 1–7).
- **NO-GO / pivot** if even here the SAE finds nothing or ablation does nothing → this is
  itself a finding; reframe the paper as *"SAEs fail to causally validate injected
  artifacts — a benchmark quantifying it"* (still Track-1-worthy). Tell the PI.

**Acceptance:** environment reproducible; D1–D3 logged; smoke test returns GO or a
documented NO-GO/pivot.

---

## Phase 1 — Data & controlled artifact injection (Day 1–3)

**Files:** `src/carve/data/datasets.py`, `src/carve/data/artifacts.py`,
`src/carve/data/download.py`, `scripts/10_inject.py`, `tests/test_injection.py`.

**Tasks**
- [ ] Loaders for dermatology data (HAM10000/ISIC; binary task e.g. melanoma-vs-nevus to
  start). Fixed, seeded splits: `probe_train / sae_train / select / eval / test`
  (all disjoint — assert it). See `docs/DATASETS.md`.
- [ ] Artifact injectors, each returning `(image, artifact_mask)`:
  - `ruler`, `marker_ink`, `dark_corner`, `text_overlay` (≥3 required for MVP).
  - **Controlled axes:** opacity α ∈ {graded}, plus **real overlays** (extracted real
    rulers/hair) in addition to synthetic shapes (to counter the "toy" critique).
- [ ] **Controlled spurious correlation ρ:** a function that, given a clean labelled set,
  produces a biased set where artifact presence correlates with the label at strength
  ρ ∈ {0.5, 0.7, 0.9, 1.0}. Keep per-image metadata: `{label, artifact_type, present,
  alpha, rho_bucket, mask_path}`.
- [ ] Determinism test: same seed ⇒ identical injected pixels & masks.

**Acceptance:** for each (artifact, ρ, α) you can materialize a biased train set + a
clean test set + an artifact-toggled test set, with masks and metadata, reproducibly.

---

## Phase 2 — Induce & confirm the shortcut (Day 2–4)

**Files:** `src/carve/models/encoders.py`, `src/carve/models/probe.py`,
`scripts/20_train_probe.py`.

**Tasks**
- [ ] Load the chosen encoder(s) wrapped so activations at layer ℓ are hookable
  (HookedViT / forward hooks). Freeze the encoder.
- [ ] Train a linear (or light MLP) probe on `probe_train` of each **biased** dataset.
- [ ] **Confirm reliance:** report the **bias gap** and the **input-level artifact causal
  effect** (Δlogit from toggling the artifact on test images) per (artifact, ρ, α).
  This quantifies how much there is to recover; it is also a reported result.
- [ ] **Zero-shot arm (no probe):** also report MONET's own decision shift from toggling the
  artifact (no trained head) — the purest causal ground truth; compare to the induced arm.
- [ ] Pick the analysis layer ℓ by a **pre-committed rule** (e.g., layer maximizing
  artifact linear-decodability on `select`, fixed in PREREGISTRATION) — and still report
  the full layer sweep for transparency.

**Acceptance:** measurable bias gap and nonzero input-level effect for the headline
(artifact, ρ, α) cells; layer choice justified by the pre-registered rule.

---

## Phase 3 — Sparse autoencoder (Day 3–6)

**Files:** `src/carve/sae/train_sae.py`, `src/carve/sae/load_sae.py`,
`scripts/30_train_sae.py`.

**Tasks**
- [ ] Load a pretrained CLIP SAE (Prisma) for the generality model; **train a TopK SAE**
  on `sae_train` activations of the medical FM at layer ℓ.
- [ ] Report SAE health: reconstruction error, fraction of dead features, sparsity (k),
  and **feature stability across seeds** (don't rely on one lucky seed).
- [ ] Sweep SAE width and k modestly; pre-commit the selection rule.

**Acceptance:** an SAE per (model, seed) with logged health metrics and reproducible
checkpoints (git-ignored, path in config).

---

## Phase 4 — Feature discovery (Day 5–7)

**Files:** `src/carve/sae/discovery.py` (or in `eval/`), `scripts/40_run_interventions.py`.

**Tasks** — report **two settings, kept strictly separate** (integrity):
- [ ] **Oracle selection** (upper bound on steering): on `select`, choose feature(s) `S`
  by highest detection AUROC for artifact-present vs absent. Allow a small top-m set.
- [ ] **Unsupervised discovery** (realistic): without using injected labels, do the
  top-activating / highest-variance features on artifact images correspond to the
  artifact? Report discovery precision@k and whether the oracle feature is found.
- [ ] Enforce: selection on `select`, evaluation on `eval` — **disjoint** (assert).

**Acceptance:** `S_oracle` and `S_discovered` per cell, with detection AUROC and a clear
statement of which split produced them.

---

## Phase 5 — Interventions & metrics (Day 6–11) — **the core results**

**Files:** `src/carve/interventions/hooks.py`,
`src/carve/interventions/mediation.py`, `src/carve/metrics/causal.py`,
`src/carve/eval/harness.py`, `scripts/40_run_interventions.py`.

**Tasks**
- [ ] Implement activation-level **ablation** (zero feature), **steering** (subtract
  scaled decoder direction), via hooks on the residual stream at ℓ.
- [ ] Compute, per image and aggregated (definitions in METRICS):
  - **Causal recovery** — fraction of the input-level artifact effect reproduced by
    ablating `S`.
  - **Steering selectivity** — effect on artifact-contaminated vs clean cases.
  - **Off-target damage** — clean-test accuracy drop after the intervention.
- [ ] Run for `S_oracle` and `S_discovered`, across **artifact × ρ × α**, over ≥3 seeds,
  with bootstrap CIs.

**Acceptance:** a tidy results table (one row per
model×artifact×ρ×α×selection×seed → metrics) materialized under `experiments/runs/...`.

---

## Phase 6 — Baselines on identical ground truth (Day 8–12)

**Files:** `src/carve/baselines/{cav,cdep,raw_neuron,random_ctrl,dermfmzero_suppress}.py`,
`scripts/50_baselines.py`.

**Tasks** — evaluate with the **same metrics, same cells, same eval split**:
- [ ] **Raw-neuron ablation** — ablate the most artifact-correlated raw activation
  neuron(s). (Tests "what does the SAE add over raw neurons?" — the key objection.)
- [ ] **CAV / Reveal2Revise** — artifact concept-activation vector; suppress along it.
  Use the authors' method faithfully (zennit-crp).
- [ ] **CDEP** — contextual-decomposition penalty using the injected masks.
- [ ] **DermFM-Zero-style suppression** — zero the top-k SAE neurons most activated by the
  artifact (reimplement from arXiv 2602.10624; see `docs/NOVELTY.md`). The incumbent we validate.
- [ ] **Random-feature control** — must do ≈nothing (sanity).
- [ ] **Input-removal oracle** — removing the artifact at the input = achievable ceiling.

**Acceptance:** every baseline reported with CIs alongside CARVE/SAE on the same axes;
honest win/loss stated.

---

## Phase 7 — Realism, robustness, aggregation, figures (Day 11–14)

**Files:** `scripts/60_aggregate.py`, `src/carve/eval/figures.py`, `experiments/`.

**Tasks**
- [ ] **Real-artifact slice:** repeat the headline evaluation using **real** ruler/marker
  annotations (e.g., Bissoto's ISIC artifact labels) to provide partial external validity.
- [ ] Seed/variance aggregation; significance of "method A > B" (don't claim within-noise).
- [ ] Figures: recovery-vs-ρ curves; selectivity vs off-target scatter (per method);
  detection-AUROC bars; the headline benchmark table.
- [ ] Write `docs/RESULTS.md` summarizing findings (including any negatives).

**Acceptance:** publication-ready table + 3–4 figures, fully regenerable from committed
runs; results narrative drafted.

---

## Deliverables → paper mapping

| Deliverable | Paper element |
|---|---|
| Injector + splits + harness (released) | the **benchmark** contribution |
| Phase-2 bias gap / input-effect table | "models do rely on the artifact" (motivation) |
| Phase-5 recovery/selectivity/off-target | core results table |
| Phase-6 baseline comparison | "does SAE steering beat CAV/CDEP/raw-neuron?" |
| Phase-7 real-artifact slice | external-validity / limitations |
| `docs/INTEGRITY.md` adherence | reproducibility statement |

## Parallelization (for multiple agents)

Mostly sequential, but these can proceed in parallel once Phase 1 lands:
- **A:** Phase 2 (probe/encoder) ‖ **B:** Phase 3 (SAE) — both depend only on Phase 1.
- **C:** Phase 6 baselines (CAV/CDEP/raw-neuron) can be built against the Phase-2 model
  while Phase 5 metrics are implemented.
- Metric code (Phase 5) and figure code (Phase 7) can be written test-first in parallel
  using toy tensors before real activations exist.
Keep one agent as integrator owning `eval/harness.py` and the run schema to avoid drift.

## Risk register & fallbacks

| Risk | Trigger | Mitigation / fallback |
|---|---|---|
| SAE doesn't recover the artifact even at ρ=1, opaque | Phase-0 NO-GO | Pivot to the negative-result benchmark framing (still Track 1). |
| "Synthetic = toy" reviewer objection | always | ρ<1 partial correlation + graded opacity + **real overlays** + the Phase-7 real slice; lead with *benchmark*, not "SAE wins". |
| Polysemantic feature / no clean single S | Phase 4 | Allow top-m feature sets; report selectivity honestly; target middle layers. |
| Influence of compute/time | Phase 5 slips past ~Jul 10 | Drop to 2 artifacts + ρ∈{0.7,1.0}; Track 2 (4 pp) "position + pilot". |
| Baseline beats SAE | Phase 6 | Report it; the benchmark + the finding is still the contribution. |

## Definition of done (MVP)

Headline benchmark table (≥3 artifacts × ≥3 ρ, ≥3 seeds, all methods, CIs) +
real-artifact slice + released injector/harness, all regenerable from committed runs,
PREREGISTRATION frozen before the final grid, RELATED_WORK citations verified.
