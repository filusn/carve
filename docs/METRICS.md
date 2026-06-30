# CARVE — Metric definitions

Every number reported in CARVE is defined here. Implement in `src/carve/metrics/causal.py`
with unit tests on toy tensors (`tests/test_metrics.py`). Do not redefine these after the
PREREGISTRATION is frozen.

## Notation

- `x_art` — an image **with** the injected artifact. `remove(x_art)` — the *same* image
  with the artifact removed (available because we injected it: the clean source, or
  mask-inpainted region). `add(x)` — a clean image with the artifact injected.
- `f(·)` — the model's scalar **decision signal**: the binary-task **logit margin**
  `z_pos − z_neg` (specify and fix in config). All effects are differences in `f`.
- `ℓ` — the analysis layer. `S` — the candidate artifact feature(s) (SAE features, or a
  CAV direction, or raw neurons, depending on method).
- `f_{−S}(·)` — forward pass with `S` **ablated** (zeroed) or **steered** (decoder
  direction subtracted with coefficient `c`) at `ℓ`, **input unchanged**.

## 1. Bias gap (reliance — Phase 2)

How much the model relies on the artifact shortcut.
```
bias_gap = acc(artifact-ALIGNED test) − acc(artifact-CONFLICTING test)
```
where *aligned* = artifact points to the true label, *conflicting* = artifact points to
the wrong label. `bias_gap ≈ 0` ⇒ no reliance ⇒ nothing to recover.

## 2. Input-level (gold) artifact effect (Phase 2)

The artifact's **true** causal effect on the prediction, by intervening on the input:
```
e_in(x_art) = f(x_art) − f(remove(x_art))
```
Report its distribution over artifact-present eval images. This is the ground truth the
feature-level interventions are judged against.

## 3. Causal recovery (Phase 5 — headline)

Does ablating `S` reproduce removing the artifact from the input? Per image (normalized
mediation residual; bounded, interpretable):
```
R(x_art) = 1 − |f_{−S}(x_art) − f(remove(x_art))|
               ────────────────────────────────────
               |f(x_art)     − f(remove(x_art))|        (= |e_in|, guard ε)
```
`R = 1` perfect mediation; `R = 0` no recovery; `R < 0` wrong-direction/overshoot.
Aggregate = median R over eval images (+ bootstrap 95% CI). Also report the raw ratio
`e_S/e_in` as a secondary view. Skip images with `|e_in| < ε` (pre-register ε).

## 4. Steering selectivity (Phase 5 — headline)

Effect should land on artifact-contaminated cases, not clean ones (RAVEL-style
cause/isolation).
```
Cause     = mean over ARTIFACT cases of  |f(x_art) − f_{−S}(x_art)|        (want high)
Isolation = mean over CLEAN    cases of  |f(x_clean) − f_{−S}(x_clean)|    (want ~0)
Selectivity = Cause / (Cause + Isolation)            ∈ [0,1], 1 = perfectly selective
```
Report Cause, Isolation, and Selectivity separately (don't hide the trade-off).

## 5. Off-target damage (Phase 5 — headline)

Collateral cost on the clean task:
```
off_target = acc_clean(before intervention) − acc_clean(after intervention)
```
Report also effect on other classes / unrelated concepts if available. Lower = better.

## 6. Detection AUROC (Phase 4)

Can the feature *detect* the artifact at all? Use the feature activation (or CAV
projection / neuron activation) as the score for artifact-present vs absent on the
designated split. Report for both `S_oracle` (selected on `select`) and the unsupervised
`S_discovered`. **Detection ≠ control:** high AUROC with low recovery is a key, reportable
dissociation.

## 7. Steering response curve (Phase 5, for steering methods)

When steering with coefficient `c` (not just ablation), sweep `c` and plot
`Cause(c)` vs `off_target(c)`; report the achievable selectivity/off-target frontier per
method. Compare frontiers across methods rather than single points where possible.

## Aggregation & statistics

- **Per cell** = (model, artifact, ρ, opacity, selection∈{oracle,discovered}, method).
- Image-level metrics → **bootstrap 95% CIs** (≥1000 resamples).
- Headline numbers → **mean ± std over ≥3 seeds**.
- Method comparisons ("SAE > CAV") → **paired bootstrap**; do **not** claim a win whose
  CI overlaps zero. State effect sizes, not just significance.

## Pre-registered success thresholds (fill in PREREGISTRATION before final runs)

Proposed defaults (freeze or change *before* seeing final results):
- "Substantial mediation": median `R ≥ 0.5`.
- "Selective": `Selectivity ≥ 0.8` with `off_target ≤ 2%`.
- "Beats baseline": paired-bootstrap CI of the metric difference excludes 0.

These thresholds are for interpretation only; report the full numbers regardless.
