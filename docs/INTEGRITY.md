# CARVE — Research-integrity protocol (binding)

CARVE is a benchmark paper; its entire value is **methodological honesty**. These rules
are binding on every agent and on the PI. If following them makes the result weaker, the
result was weaker — report it.

## 1. Pre-registration (no HARKing, no metric-hacking)
- Fill and **commit `PREREGISTRATION.md` before launching the final experiment grid**:
  hypotheses, exact metrics (already in METRICS.md), success thresholds, the
  **model/layer/SAE selection rules**, datasets/splits, baselines, analysis & stats plan,
  seeds, and what outcome would falsify the claim.
- Header must record the freeze date and git commit hash. After freezing, **do not change
  metric definitions or selection rules** in response to results. Exploratory analyses are
  allowed but must be **labelled exploratory** and separated from the pre-registered ones.

## 2. No fabrication
- Every reported number traces to a committed run dir (`experiments/runs/...`) with config,
  commit hash, seed, raw per-image outputs, and metric json. If you can't regenerate it,
  it doesn't go in the paper. Never hand-edit a results file.

## 3. Split discipline (no leakage)
- Splits `probe_train / sae_train / select / eval / test` are disjoint. **Feature
  selection happens on `select`; causal evaluation on `eval`.** Assert disjointness in code
  and in a test. Selecting `S` and evaluating it on the same images invalidates the result.

## 4. Discovery vs control are reported separately
- `S_oracle` (label-selected) bounds *what steering can do*; `S_discovered` (unsupervised)
  reflects *what you'd actually find*. Never report oracle numbers as if they were
  unsupervised. State which split and which selection produced every number.

## 5. Fair baselines
- Implement CAV/Reveal2Revise and CDEP from the authors' descriptions/code, with the
  **same tuning budget** as our method. No strawmen. Include the **raw-neuron** baseline
  (the "what does the SAE add?" test), the **random-feature control** (must do ≈nothing),
  and the **input-removal oracle** (ceiling). If a baseline wins, the paper says so.

## 6. Negative results are valid
- "SAEs do not reliably recover/steer the injected artifact" is a legitimate CARVE finding
  and a Track-1 paper. Do not tune ρ/opacity/layer/seed to manufacture a positive result.
  Report the regimes where it fails.

## 7. Claim boundary: synthetic → real
- State explicitly that injected artifacts are a **controlled testbed**, not clinical
  validation. Do **not** claim the model is "made safe for clinical use."
- Provide partial external validity via the **real-artifact slice** (Bissoto annotations).
  Frame conclusions as about *the method's behavior under known ground truth*.

## 8. Honest novelty & attribution
- We are **not** "first SAE in dermatology" (DermFM-Zero, Mammo-SAE, MedSAE, CytoSAE, etc.
  exist). Our claim is **first controlled causal validation + reusable benchmark**. Cite
  DermFM-Zero, RAVEL, SynthSAEBench, "Sanity Checks for SAEs", Reveal2Revise prominently.
- Verify every citation against the real paper (`docs/RELATED_WORK.md` status column).
  Never cite from memory or from a search snippet alone.

## 9. Statistics
- Bootstrap CIs on image-level metrics; mean ± std over seeds; paired bootstrap for
  comparisons. No claim whose CI overlaps the null. Report effect sizes and n.

## 10. Reproducibility & licensing
- Release injector, fixed splits (indices/seeds), SAE checkpoints, eval harness, configs.
- **Do not redistribute datasets**; ship a prep script that runs on user-downloaded data.
  Respect licenses (HAM10000 = CC BY-NC; ISIC per-image terms). Record provenance in
  `docs/DATASETS.md`.
- Pin dependency versions; log environment (`pip freeze`) into each run dir.

## 11. AI-assistance disclosure
- If AI tools materially assisted code or writing, disclose per MICCAI/MI4MedFM policy in
  the paper. Keep `docs/LOG.md` of substantive design decisions and who/what made them.

## Before you report ANY number — checklist
- [ ] Produced by a committed script; run dir saved (config + commit + seed + raw + json).
- [ ] Selection on `select`, evaluation on `eval`, both disjoint from `probe_train`.
- [ ] Oracle vs discovered clearly labelled.
- [ ] ≥3 seeds; bootstrap CI attached; comparison uses paired bootstrap.
- [ ] Metric matches `docs/METRICS.md` exactly (no silent redefinition).
- [ ] Pre-registered vs exploratory clearly marked.
- [ ] Any citation involved is VERIFIED in `docs/RELATED_WORK.md`.
