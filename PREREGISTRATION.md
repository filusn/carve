# CARVE — Pre-registration

> **Freeze this file (commit) BEFORE launching the final experiment grid (Phase 5–7).**
> After freezing, metric definitions and selection rules do not change in response to
> results. Exploratory analyses are allowed but must be labelled exploratory.

- **Frozen on:** `[DATE]`  — **git commit:** `[HASH]`  — **by:** `[name]`
- **Status:** ☐ DRAFT ☐ FROZEN

## 1. Hypotheses (pre-specified)
- H1 (recovery): On injected artifacts, the oracle SAE artifact-feature achieves median
  causal recovery `R ≥ [threshold]` at ρ=1.0/high-opacity.
- H2 (selectivity): SAE steering reaches Selectivity `≥ [t]` with off-target `≤ [t]%`.
- H3 (vs baselines): SAE steering's (recovery, selectivity@matched-off-target) is `≥` /
  `<` CAV/Reveal2Revise, CDEP, raw-neuron (state the directional prediction per method).
- H4 (detection ≠ control): high detection AUROC does **not** imply high recovery
  (report the dissociation).
- H5 (degradation): recovery/selectivity degrade as ρ↓ and opacity↓ (characterize).

## 2. Metrics
As defined in `docs/METRICS.md` (causal recovery R; Cause/Isolation/Selectivity;
off-target; detection AUROC; bias gap; steering frontier). **No redefinition after freeze.**
- Decision signal `f` = `[logit margin z_pos−z_neg]`.
- Recovery guard ε = `[value]` (skip |e_in|<ε).

## 3. Success thresholds (interpretation only; report full numbers regardless)
- Substantial mediation: median R ≥ `[0.5]`.
- Selective: Selectivity ≥ `[0.8]` and off-target ≤ `[2%]`.
- "Beats baseline": paired-bootstrap CI of the metric difference excludes 0.

## 4. Models, layer, SAE — selection rules (pre-committed)
- Base model(s): `[medical FM = …]` (+ `[CLIP ViT-B/16]` generality check).
- Layer ℓ rule: `[e.g., layer maximizing artifact linear-decodability on `select`]`;
  full layer sweep still reported.
- SAE: TopK, width `[…]`, k `[…]`; selection rule `[…]`. Pretrained vs trained: `[…]`.
- Feature set S size m: `[…]`; `S_oracle` rule (top AUROC on `select`); `S_discovered`
  rule (unsupervised top-activation/variance on artifact images, no labels).

## 5. Data, splits, artifacts
- Dataset/task: `[HAM10000, melanoma vs nevus]`. Splits seeded & disjoint (asserted).
- Artifacts: `[ruler, marker_ink, dark_corner]` (+ text_overlay if time).
- ρ grid: `[0.5, 0.7, 0.9, 1.0]`; opacity grid: `[…]`; real-overlay slice: `[Bissoto]`.

## 6. Baselines
Raw-neuron ablation; CAV/Reveal2Revise; CDEP; random-feature control; input-removal
oracle. Same metrics, cells, eval split, tuning budget.

## 7. Analysis & statistics
≥3 seeds (`[list]`); bootstrap 95% CIs (≥1000 resamples); paired bootstrap for
comparisons; no claim with CI overlapping null; report effect sizes + n.

## 8. What would falsify / produce a negative result
- No bias gap ⇒ injection invalid (fix before proceeding, not a finding about SAEs).
- Oracle feature exists (high AUROC) but median R ≈ 0 across cells ⇒ "SAEs detect but
  don't causally control injected artifacts" — report as the headline negative result.
- Raw-neuron/CAV ≥ SAE everywhere ⇒ report "SAE adds no causal-control value here".

## 9. Deviations log
Record any post-freeze deviation with reason and date. `[none yet]`
