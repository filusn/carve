# CARVE — progress summary

**As of 2026-07-03.** Target: MI4MedFM @ MICCAI 2026, deadline **2026-07-15**, Track 1.
CARVE = *Causal ARtifact Validation of Encodings* — a benchmark testing whether SAE
interpretability can **causally control** (not merely detect) a medical foundation model's
known, injected artifact bias, head-to-head against incumbent methods.

This file summarizes (1) how the injected artifacts affect classification, (2) the Stage-6
benchmark results, (3) the data pipeline and how data is treated, and (4) the CAV status.
Every number below comes from a committed script + saved run dir (no fabricated results).

---

## Stage status

| stage | what | status |
|---|---|---|
| 1 | MONET hook probe (activation read/write) | ✅ |
| 2 | data: artifact injection + splits | ✅ |
| 3 | encoders + probe + zero-shot arm + metrics + Phase-0 gate | ✅ |
| 4 | TopK SAE: train + detection/discovery | ✅ |
| 5 | interventions (ablate/steer) + eval harness | ✅ |
| 6 | **core-4 baselines + aggregation + figures** | ✅ |
| — | CAV / Reveal2Revise + CDEP (2nd baseline pass) | ⬜ deferred (task #12) |
| — | SAE-health fix + freeze PREREGISTRATION | ⬜ gate before final (task #13) |
| 7 | ρ×α sweep + real-artifact slice + paper figures | ⬜ |

---

## 1. How the injected artifacts affect MONET's classification

**Decision is controllable (Stage-1 hook probe).** Ablating layer 12 shifts the zero-shot
mel−nevus logit margin by **1.56** (random steer 1.83) → the residual stream causally drives
the decision. This is the pre-condition for the whole study.

**Bias magnitude & direction (Phase-0 gate, ρ=1, α=1, layer 12).** Decision signal
`f` = logit margin `z_pos − z_neg`; input effect `e_in = f(x_art) − f(x_clean)`.

| artifact | zero-shot `e_in` (median, CI95) | direction | induced-probe **bias_gap** |
|---|---|---|---|
| ruler | **−0.51** [−0.66, −0.34] | → **benign / nevus** | 1.00 |
| marker_ink | **+0.49** [+0.39, +0.64] | → **malignant / melanoma** | 0.945 |
| dark_corner | **+0.45** [+0.30, +0.57] | → **malignant** | 0.995 |

Findings:
- **Directional and artifact-specific.** Ink and dark corners push toward *malignant*
  (classic Winkler-style dermoscopy confound); the ruler pushes the *opposite* way, toward
  benign. "Artifact bias" is not one effect — sign and size depend on the mark.
- **Near-total at the easy setting.** `bias_gap ≈ 1.0` means the induced classifier's call is
  almost entirely decided by artifact presence (clean-task accuracy 0.92 for reference). It
  **saturates** at ρ=1/α=1 → the real grid needs graded ρ∈{0.5,0.7,0.9} and α.
- **Physically sane.** Shrinking marker_ink to a small dot halved its effect
  (`e_in` +0.96→+0.49) — smaller footprint ⇒ smaller cause.

This is the *known bias* CARVE then asks interpretability to undo.

---

## 2. Stage-6 benchmark — detection vs. control (full 3-seed grid)

MONET, **zero-shot arm** (purest causal GT: no trained head), ρ=0.9, α=1.0, layer 12,
width-16384 SAE, eval n=250, 3 seeds. All methods on the **same** cells/metrics/eval split.
`R` = causal recovery (1 = fully undoes the artifact effect, 0 = does nothing).

| method | selects on | detection AUROC | **recovery R** (mean±std) | selectivity | off-target |
|---|---|---|---|---|---|
| **input-removal oracle** (ceiling) | — | — | **1.00 ± 0.00** | 1.00 | 0.00 |
| **SAE oracle-ablate** (ours) | top-AUROC feature | 0.994–1.000 | **≈0** (−.03 / +.03 / +.01) | **0.98–0.998** | 0.00 |
| **raw neuron** (budget-matched) | top-AUROC neuron | 0.973–0.995 | **≈0** (+.03 / +.00 / +.01) | 0.44–0.67 | ~0 |
| **DermFM-Zero** top-5 (incumbent) | top-activation | — | **erratic** (+.07 / **−.76** / **−.48**) | 0.42–0.47 | **0.125** |
| random raw (control) | random | — | ≈0 | ~0.45 | ~0 |

*(three R values = ruler / marker_ink / dark_corner)*

**Headline:** detection is saturated for **both** SAE features and raw neurons, yet recovery
sits at **zero** — the *detection ≠ control* dissociation, robust across seeds and artifacts.
- The SAE's only edge over a single raw neuron is **selectivity**, not control — this answers
  the killer objection "what does the SAE add over raw neurons?": near-perfect selectivity,
  but no more causal control (neither controls the artifact).
- The one method that *moves* the decision (DermFM-Zero suppression) does so
  **non-selectively** (sel ~0.45), at **12.5% clean-task cost**, and **in the wrong direction
  on 2/3 artifacts** — nowhere near the input-removal ceiling of 1.0. Its published one-way
  AUROC gain hides this.
- **Robust to SAE health:** the same dissociation holds on a *healthy* width-4096 SAE
  (9.6% dead-features) — it is not an artifact of the 55%-dead wide dictionary.

Figures in `figures/phase6/` (regenerable via `scripts/60_figures.py`):
`detection_vs_recovery.png` (headline), `recovery_bars.png`,
`selectivity_vs_offtarget.png`, `detection_bars.png`.
Run dir: `experiments/runs/20260703T134217Z_baselines_grid`.

---

## 3. Data pipeline — how data is treated, and why

**Source.** HAM10000 / ISIC-2018 Task3, binarized to melanoma(+) vs nevus(−): **n=7,818**
(MEL 1,113 / NV 6,705, ~6:1 imbalance). Public + synthetic-only. Bissoto real-artifact
labels unavailable locally → artifacts are **injected** onto HAM10000 as-is.

**Why inject rather than use real artifacts.** Injection hands us the **ground-truth causal
effect for free**: for every clean image we produce its exact counterfactual —
`remove(source=clean)` returns the *known* clean canvas — so
`e_in = f(x_art) − f(x_clean)` is the artifact's true per-image effect, **measured, not
estimated**. Every method's recovery `R` is scored against this gold quantity. Real
pre-existing artifacts have no counterfactual, so "undoing" them could never be certified.

**The 5-way disjoint split** (stratified by label, seeded):

| split | frac | role | reasoning |
|---|---|---|---|
| `probe_train` | 0.40 | trains the induced-arm linear probe | the "biased classifier" whose reliance we probe |
| `sae_train` | 0.25 | trains the SAE | **on clean activations only** |
| `select` | 0.10 | **picks** the artifact feature/neuron | uses injected present/absent labels |
| `eval` | 0.15 | **measures** causal recovery | the counterfactual is applied here |
| `test` | 0.10 | held out, untouched | reserved for final numbers |

**Non-negotiable rule (INTEGRITY §4, enforced by `assert_disjoint`):** the split you *select*
the feature on is disjoint from the split you *measure* its effect on, which is disjoint from
probe training. Otherwise "this feature controls the artifact" is memorized selection —
leakage.

**The SAE never sees the injected artifact.** Trained on `sae_train` clean activations and
artifact-agnostic; injected artifacts appear only later in `select`/`eval`. If the SAE trained
on injected images we would be *planting* the feature we then "discover" — circular. Instead
we learn MONET's natural feature basis and ask whether one of those features *happens* to
detect and control the injected cause.

**The ρ-biased set.** `make_biased_set` injects so presence correlates with the label at
strength ρ: `P(present | melanoma) = ρ`, `P(present | nevus) = 1 − ρ`. This manufactures the
spurious correlation a shortcut-learner latches onto, and gives `select` a present/absent
signal for detection AUROC. At **eval** we do not rely on correlation — we inject on *every*
clean eval image and compare to its own clean version (the per-image counterfactual).

**Same cells for every method.** A "cell" = (artifact, ρ, α, seed) fixes one `x_art`/`x_clean`
pair and one selection split. `run_cell` applies SAE-ablate, raw-neuron, DermFM-Zero, random,
and the input oracle through **one interface** (activation-editor `act_fn` | SAE `op/S` |
`oracle=True`) to that identical pair — which is what makes the Stage-6 table directly
comparable rather than five separate experiments.

**Determinism.** Every injection is seeded by image index (`rng(i)`), so `x_art` is
byte-identical across runs; seeds vary only the split/SAE. Reproducibility is a property of
the pipeline, not luck. Every run dir records config + git commit + seed + per-image parquet.

**Two decision arms.** Stage-6 numbers use the **zero-shot arm** (`probe=None`, MONET text
prompts, no trained head) — purest causal GT. The **induced-probe arm** (a class-weighted
logistic probe on a ρ-biased set) is the stress-test variant; the 6:1 imbalance is handled by
stratified splits + class weighting.

---

## 4. CAV (Concept Activation Vector) — NOT yet run

**No CAV results exist** — it is the deferred second baseline pass (task #12). Per the
project's no-fabrication rule, no mitigation number is reported here. What follows is the
plan and an explicit *hypothesis*, not a measurement.

**What CAV does / how it's meant to mitigate.** A CAV (Kim et al., TCAV; used for correction
in Reveal2Revise / ClArC, Anders et al.) fits a linear classifier at layer ℓ separating
**artifact-present vs artifact-absent** activations; its normal is the "artifact direction."
Mitigation edits activations at inference to **remove the component along that direction**
(project onto its orthogonal complement). In our harness this is just another `act_fn` on the
same cells → it lands directly in the Section-2 table.

**Hypothesis (prereg H3 — untested).** A single raw neuron and the SAE feature both detect at
~1.0 AUROC yet recover R≈0; CAV is likewise a **single linear direction** in the same
activation space, so it is expected to also land near **R≈0 (detect-but-don't-control)**,
plausibly with selectivity between the raw neuron (~0.5) and the SAE (~0.98). If CAV instead
recovers substantially, that is itself a publishable twist. Either outcome is why it must be
run before any number is stated.

---

## Immediate next steps

1. **CAV + CDEP** (task #12) — `fit_cav` + projection-removal `act_fn`; one grid pass on the
   identical cells → completes the head-to-head table.
2. **Pre-final gate** (task #13) — fix SAE dead-feature fraction (resample or commit to
   width-4096) + report cross-seed decoder stability; **freeze `PREREGISTRATION.md`**
   (thresholds, layer rule, SAE width/k, recovery ε, steer-coeff sign) — hard rule before the
   final grid.
3. **Phase 7** — full ρ×α sweep (currently fixed ρ=0.9/α=1.0) + real-artifact slice + paper
   figures.

_Branches:_ Stage-6 work on `phase6-baselines` (commits `b97d368`, `e8e1981`).
Not pushed — awaiting PI, per repo git discipline.
