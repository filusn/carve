# CARVE — Pre-registration

> **Frozen BEFORE the final experiment grid (Phase 5–7 ρ×α sweep).** After freezing,
> metric definitions and selection rules do not change in response to results. Exploratory
> analyses are allowed but must be labelled exploratory. Any deviation is logged in §9.

- **Frozen on:** `2026-07-03` — **git commit:** `0840de8` (freeze commit adds this file) —
  **by:** Bartlomiej Moniak (with Claude Code)
- **Status:** ☑ FROZEN

## 1. Hypotheses (pre-specified)
- **H1 (recovery):** On injected artifacts, the oracle SAE artifact-feature achieves median
  causal recovery `R ≥ 0.5` at ρ=1.0 / high opacity.
- **H2 (selectivity):** SAE steering reaches Selectivity `≥ 0.8` with off-target `≤ 2%`.
- **H3 (vs baselines):** SAE steering's (recovery, selectivity @ matched off-target) is
  compared to CAV/Reveal2Revise, DermFM-Zero top-k suppression, and raw-neuron ablation.
  Directional prediction: **no linear activation-space method (SAE feature, raw neuron, CAV)
  attains selective causal control** (R≈0 despite high detection); the input-removal oracle
  is the only method reaching R≈1.
- **H4 (detection ≠ control):** high detection AUROC does **not** imply high recovery (report
  the dissociation).
- **H5 (degradation):** recovery/selectivity degrade as ρ↓ and opacity↓ (characterize).

## 2. Metrics
As defined in `docs/METRICS.md` (causal recovery R; Cause/Isolation/Selectivity; off-target;
detection AUROC; bias gap; steering frontier). **No redefinition after freeze.**
- Decision signal `f` = **logit margin `z_pos − z_neg`** (`model.decision_signal`).
- Recovery guard **ε = 1e-3** (per-image |e_in| < ε are returned NaN and skipped in aggregation).
- Intervention ops: **ablate** = subtract the feature's reconstructed contribution
  `z_f·W_dec[f]`; **steer** = subtract `c · Σ unit-decoder(S)`. The recovery direction is
  subtract with **positive** c; the config list `interventions.steer_coeffs` stores magnitudes,
  applied as the recovery-direction subtraction (sign fixed here, not re-chosen after results).

## 3. Success thresholds (interpretation only; report full numbers regardless)
- **Substantial mediation:** median R ≥ 0.5.
- **Selective:** Selectivity ≥ 0.8 **and** off-target ≤ 2%.
- **"Beats baseline":** paired-bootstrap 95% CI of the metric difference excludes 0.

## 4. Models, layer, SAE — selection rules (pre-committed)
- **Base model:** MONET (HF `chanwkim/monet`, CLIP ViT-L/14, 24 blocks, d=1024), frozen
  encoder. Generality check (if time): CLIP ViT-B/16.
- **Layer ℓ rule:** ℓ = layer maximizing artifact linear-decodability on `select`; **ℓ=12**
  used for the MVP; full layer sweep still reported.
- **SAE:** TopK in the raw residual-stream space, k=32, b_dec init = data mean, unit-norm
  decoder, 3000 steps, **AuxK dead-feature revival on** (`train.aux_k=256`, `dead_window=200`;
  Gao et al. 2024) so the dictionary is not starved.
- **SAE width — selection rule:** the **widest** dictionary whose held-out **dead-feature
  fraction ≤ 15% and R² ≥ 0.98** at k=32; a width sweep {4096, 8192, 16384 ± AuxK} with
  cross-seed decoder-cosine stability is reported (`scripts/31_sae_health_sweep.py`). On the
  evidence at freeze, **width 4096** meets the bar (dead ≈9.6%, R² ≈0.99, ruler detection
  AUROC ≈1.0); it is the registered primary. A wider dictionary is promoted **only if** the
  sweep shows it meets the same ≤15%-dead / ≥0.98-R² bar (rule fixed; winner data-determined
  and reported, not re-chosen to favor any hypothesis).
- **Feature-set S size m = 1.** `S_oracle` = top detection-AUROC feature on `select` (uses the
  injected present/absent labels). `S_discovered` = unsupervised top activation-variance on the
  artifact set, no labels. Kept and reported separately.

## 5. Data, splits, artifacts
- **Dataset/task:** HAM10000 / ISIC-2018 Task3, melanoma(+) vs nevus(−), n=7,818 (MEL 1,113 /
  NV 6,705). Public + synthetic only.
- **Splits (seeded, stratified, pairwise-disjoint, asserted in code):** probe_train 0.40,
  sae_train 0.25, select 0.10, eval 0.15, test 0.10. Selection (`select`) ⟂ effect (`eval`) ⟂
  probe training — enforced by `assert_disjoint`.
- **Artifacts:** ruler, marker_ink, dark_corner. **ρ grid:** {0.5, 0.7, 0.9, 1.0}.
  **opacity grid:** {0.4, 0.7, 1.0}. Gold effect via the exact inject/remove counterfactual.
- **Real-overlay slice (Bissoto):** annotations **not available locally** → omitted; logged as
  a deviation (docs/LOG.md, docs/DATASETS.md). External-validity limitation stated in the paper.

## 6. Baselines (same metrics, cells, eval split, tuning budget)
Raw-neuron ablation (budget-matched to the SAE); CAV / Reveal2Revise (concept-vector clamp);
DermFM-Zero top-k activation suppression (incumbent); random-feature/neuron control;
input-removal oracle (ceiling). CDEP (contextual-decomposition penalty, Rieger 2020) added if
the deadline allows; its omission is not a change to any hypothesis or metric.

## 7. Analysis & statistics
Seeds **[0, 1, 2]** for headline numbers (report mean ± std). Bootstrap **95% CIs, ≥1000
resamples**; **paired bootstrap** for method comparisons; no claim with a CI overlapping the
null; report effect sizes + n. SAE health (dead-feature fraction, R², cross-seed decoder
stability) reported alongside.

## 8. What would falsify / produce a negative result
- No bias gap ⇒ injection invalid (fix before proceeding; not a finding about SAEs).
- Oracle feature exists (high AUROC) but median R ≈ 0 across cells ⇒ "SAEs detect but don't
  causally control injected artifacts" — the registered headline negative result (H4).
- Raw-neuron / CAV ≥ SAE everywhere ⇒ "SAE adds no causal-control value here" (report honestly).

## 9. Deviations log
- `2026-07-03` — Real-overlay (Bissoto) slice omitted (annotations unavailable locally).
- `[date]` — `[any post-freeze deviation + reason]`
