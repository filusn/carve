# CARVE — progress summary

**As of 2026-07-03.** Target: MI4MedFM @ MICCAI 2026, deadline **2026-07-15**, Track 1.

CARVE = *Causal ARtifact Validation of Encodings*. In plain words: skin-cancer AI models
sometimes make their decision based on junk in the picture (like a ruler or pen mark) instead
of the actual skin spot. People have tools that claim to find the "junk detector" inside the
model and switch it off. CARVE tests whether those tools can really **switch the junk off**
(control it), not just **spot** it (detect it) — and it puts the popular tools in a fair race
against each other.

Every number below comes from a real script that ran, with its output folder and code version
saved. Nothing here is made up.

---

## The one-line takeaway (what works, what doesn't yet)

- ✅ **Works:** We can inject a fake artifact, and it clearly steers the model's decision (we
  measured exactly how much and in which direction). Every tool can **detect** the artifact
  almost perfectly (near-perfect AUROC — a detection score where 1.0 is perfect).
- ❌ **Doesn't work (and that's the finding):** None of the interpretability tools can actually
  **undo** the artifact's effect. "Recovery" — how much of the effect gets removed — sits at
  about **0**. So detecting the artifact is easy; controlling it is not. Detecting ≠
  controlling.
- 🔁 **Replicated on realistic marks:** the "detects but can't control" result also holds for a
  **new artifact set** — real ruler + arrow photo overlays and a hard dermoscope-style black circle
  (`black_corner`). See **§5** (originals in §1–§2 kept for comparison).
- 🛡️ **Survives a fair-shot stress test:** even switching off 20× more features, or handing steering a
  per-image *best-case* strength (an upper bound no real method beats), still doesn't recover the answer
  (control caps at ~0.1 for multi-feature ablation, ~0.3–0.8 for oracle steering, vs 1.0 for erasing the
  mark) — and only by hurting selectivity. Detecting ≠ controlling. See **§5.3**.
- 🚧 **Not done yet:** the final big experiment sweep (varying artifact strength), one leftover
  baseline (CDEP), and finishing one health-check on the SAE.

---

## Stage status

| stage | what | status |
|---|---|---|
| 1 | MONET hook probe (read/write the model's internal activity) | ✅ |
| 2 | data: artifact injection + splits | ✅ |
| 3 | encoders + probe + zero-shot arm + metrics + Phase-0 gate | ✅ |
| 4 | TopK SAE: train + detection/discovery | ✅ |
| 5 | interventions (switch features off / nudge them) + eval harness | ✅ |
| 6 | **core-4 baselines + aggregation + figures** | ✅ |
| 6b | **CAV / Reveal2Revise baseline** | ✅ |
| — | CDEP (remaining 2nd-pass baseline) | ⬜ (task #12) |
| 4c | SAE-health fix (AuxK) + **PREREGISTRATION frozen** | ✅ (AuxK-on-MONET sweep unverified) |
| 7 | ρ×α sweep + real-artifact slice + paper figures | ⬜ |

(SAE = sparse autoencoder, a tool that breaks the model's internal activity into many simple,
readable "features.")

---

## 1. How the injected artifacts change MONET's answer

**The decision can be steered from inside the model (Stage-1 hook probe).** When we edit the
model's internal activity at layer 12, the melanoma-vs-nevus score shifts by **1.56** (a random
edit shifts it 1.83). So the inside of the model really does drive the final answer. That's the
basic requirement for the whole study to make sense.

**How big the bias is, and which way it points (Phase-0 gate, ρ=1, α=1, layer 12).** The
"decision signal" `f` is the score gap between the two classes (`z_pos − z_neg`). The artifact's
effect on the input is `e_in = f(x_art) − f(x_clean)` — the answer with the artifact minus the
answer without it.

| artifact | zero-shot `e_in` (median, CI95) | direction | induced-probe **bias_gap** |
|---|---|---|---|
| ruler | **−0.51** [−0.66, −0.34] | → **benign / nevus** | 1.00 |
| marker_ink | **+0.49** [+0.39, +0.64] | → **malignant / melanoma** | 0.945 |
| dark_corner | **+0.45** [+0.30, +0.57] | → **malignant** | 0.995 |

What we found:

- **Each artifact pushes its own way.** Ink and dark corners push the model toward *malignant*
  (the classic real-world trap where doctors' marks bias the read). The ruler pushes the other
  way, toward *benign*. So "artifact bias" isn't one single thing — the direction and size
  depend on the mark.
- **At the easy setting the bias is almost total.** `bias_gap ≈ 1.0` means the model's call is
  decided almost entirely by whether the artifact is there (for comparison, its accuracy on the
  clean task is 0.92). The effect maxes out at ρ=1/α=1, so the real experiment needs weaker,
  graded settings: ρ∈{0.5,0.7,0.9} and varying α.
- **It behaves sensibly.** Shrinking the ink mark to a small dot roughly halved its effect
  (`e_in` +0.96→+0.49). Smaller mark, smaller effect — as you'd expect.

This is the *known bias* that CARVE then asks the interpretability tools to undo.

---

## 2. Stage-6 benchmark — detecting vs. controlling (full 3-seed grid)

Setup: MONET, **zero-shot arm** (no trained classifier head, so the ground truth is as clean as
possible), ρ=0.9, α=1.0, layer 12, width-16384 SAE, 250 eval images, 3 random seeds. Every method
runs on the **same** images, same metrics, same eval split. `R` = recovery (1 = fully undoes the
artifact's effect, 0 = does nothing).

| method | selects on | detection AUROC | **recovery R** (mean±std) | selectivity | off-target |
|---|---|---|---|---|---|
| **input-removal oracle** (ceiling) | — | — | **1.00 ± 0.00** | 1.00 | 0.00 |
| **SAE oracle-ablate** (ours) | top-AUROC feature | 0.994–1.000 | **≈0** (−.03 / +.03 / +.01) | **0.98–0.998** | 0.00 |
| **raw neuron** (budget-matched) | top-AUROC neuron | 0.973–0.995 | **≈0** (+.03 / +.00 / +.01) | 0.44–0.67 | ~0 |
| **CAV** (Reveal2Revise / ClArC) | learned linear direction | 1.000 | **≈0 / +.17** (−.03 / +.01 / **+.17**) | 0.55–0.74 | ~0 |
| **DermFM-Zero** top-5 (incumbent) | top-activation | — | **erratic** (+.07 / **−.76** / **−.48**) | 0.42–0.47 | **0.125** |
| random raw (control) | random | — | ≈0 | ~0.45 | ~0 |

*(the three R values = ruler / marker_ink / dark_corner; CAV std ≤0.03 across seeds)*

**Headline:** every method **detects** the artifact almost perfectly (AUROC ≈1.0), but **recovery
is ≈0**. So detecting the artifact and controlling it are two different things — and this gap
holds up across seeds, across artifacts, and across different families of method.

- The SAE's only advantage over a single raw neuron is **selectivity** (it changes only what it
  should), **not** control. This answers the obvious challenge, "what does the fancy SAE add over
  a plain neuron?" Answer: near-perfect selectivity — but no more actual control. Neither one
  controls the artifact.
- **CAV** (a full *learned* direction, not just one coordinate) also fails to control the
  artifact: R≈0 on the ruler and ink, and only **+0.17 on dark_corner** (a slow, whole-image
  change that lives more neatly along a straight line, so it's a bit easier) — still far below the
  1.0 ceiling. Its selectivity of 0.55–0.74 lands between the raw neuron and the SAE. Bottom line:
  no straight-line method — sparse feature, raw neuron, or concept vector — gets selective control.
- The one method that actually *moves* the decision (DermFM-Zero suppression) does it **sloppily**
  (selectivity ~0.45), damages the clean task by **12.5%**, and pushes the **wrong way on 2 of 3
  artifacts** — nowhere near the 1.0 ceiling. Its published "AUROC went up" number hides all of
  this.
- **Not a fluke of a broken SAE:** the same gap shows up on a *healthy* width-4096 SAE (only 9.6%
  dead features), so it isn't caused by the wide dictionary that had 55% dead features.

Figures live in `figures/phase6/` (rebuild with `scripts/60_figures.py`):
`detection_vs_recovery.png` (the headline), `recovery_bars.png`,
`selectivity_vs_offtarget.png`, `detection_bars.png`.
Run dir: `experiments/runs/20260703T134217Z_baselines_grid`.

---

## 3. The data pipeline — how the data is handled, and why

**Source.** HAM10000 / ISIC-2018 Task3, boiled down to melanoma(+) vs nevus(−): **n=7,818**
(MEL 1,113 / NV 6,705, about 6:1 imbalance). Public + synthetic data only. The Bissoto
real-artifact labels aren't available locally, so we **paste** artifacts onto the HAM10000 images
ourselves.

**Why paste fake artifacts instead of using real ones.** Pasting gives us the exact truth for
free. For every clean image we can make its perfect twin: `remove(source=clean)` hands back the
*known* clean picture. So `e_in = f(x_art) − f(x_clean)` is the artifact's true effect on that one
image — **measured, not guessed**. Every method's recovery `R` is graded against this gold number.
Real, pre-existing artifacts have no clean twin, so you could never prove you truly "undid" them.

**The 5-way split** (kept separate, balanced by label, fixed by seed):

| split | frac | role | why |
|---|---|---|---|
| `probe_train` | 0.40 | trains the induced-arm linear probe | the "biased classifier" we then inspect |
| `sae_train` | 0.25 | trains the SAE | **on clean activations only** |
| `select` | 0.10 | **picks** the artifact feature/neuron | uses present/absent labels |
| `eval` | 0.15 | **measures** the recovery | the clean-twin comparison happens here |
| `test` | 0.10 | held out, untouched | saved for the final numbers |

**The rule we won't break (INTEGRITY §4, checked in code by `assert_disjoint`):** the images you
use to *pick* a feature must not overlap with the images you use to *measure* its effect, and
neither may overlap with probe training. Otherwise "this feature controls the artifact" is just
the model remembering the images we picked on — that's leakage (cheating by overlap).

**The SAE never sees a pasted artifact.** It's trained only on clean `sae_train` activity, so it
knows nothing about the artifact; the pasted artifacts only show up later, in `select`/`eval`. If
we trained the SAE on artifact images, we'd be *planting* the very feature we later claim to
"discover" — circular. Instead we learn MONET's own natural set of features and ask whether one of
them just *happens* to both spot and control the artifact.

**The ρ-biased set.** `make_biased_set` pastes artifacts so that "artifact present" lines up with
the label at strength ρ: `P(present | melanoma) = ρ`, `P(present | nevus) = 1 − ρ`. This is how we
manufacture the fake correlation that a lazy model would grab onto, and it gives `select` a
present/absent signal for the detection score. At **eval** we don't lean on any correlation — we
paste the artifact onto *every* clean eval image and compare it to its own clean twin.

**Same images for every method.** A "cell" = (artifact, ρ, α, seed) and it locks in one
`x_art`/`x_clean` pair and one selection split. `run_cell` then pushes SAE-ablate, raw-neuron,
DermFM-Zero, random, and the input oracle through **one shared interface** onto that exact same
pair (`act_fn` | SAE `op/S` | `oracle=True`). That's what makes the Stage-6 table a true
apples-to-apples comparison instead of five separate experiments.

**Repeatable by design.** Each paste is seeded by the image's index (`rng(i)`), so `x_art` comes
out byte-for-byte identical every run; seeds only change the split and the SAE. Reproducibility is
built in, not luck. Every run folder saves the config + code version + seed + per-image data.

**Two ways to read the model.** The Stage-6 numbers use the **zero-shot arm** (`probe=None`,
MONET's text prompts, no trained head) — the cleanest ground truth. The **induced-probe arm** (a
class-weighted logistic classifier trained on a ρ-biased set) is the tougher stress-test version;
the 6:1 imbalance is handled with balanced splits and class weighting.

---

## 4. CAV (Concept Activation Vector) — DONE

**What CAV is and how it's meant to fix bias.** A CAV (Kim et al., TCAV; used to correct bias in
Reveal2Revise / ClArC, Anders et al.) trains a class-weighted logistic classifier at layer ℓ to
tell **artifact-present from artifact-absent** activity; the direction it points is the "artifact
direction." To mitigate, you push the activity's amount along that direction back to its clean
level at test time (`a′ = a − (a·û − b)·û`). In our setup it's just one more `act_fn` on the same
cells — it's the `cav` row in the Section-2 table.
Code: `src/carve/baselines/cav.py`; run `experiments/runs/20260703T151648Z_baselines_grid`
(commit `9c95270`).

**Result (3-seed grid).** CAV detects the artifact at **1.000 AUROC** but recovers **≈0 on the
ruler (−.03±.01) and marker (+.01±.03)**, and only **+0.17±.02 on dark_corner**; selectivity
**0.55–0.74**, off-target ~0. So our pre-registered guess H3 holds: a full *learned* direction,
just like the SAE feature and the raw neuron, **detects but does not selectively control** the
artifact. The small dark-corner recovery fits the pattern — a slow, whole-image change is the most
straight-line-friendly of the three — and it's still far from the 1.0 ceiling.

---

## 5. New artifact set — real ruler + arrow + black_corner (added 2026-07-09)

Everything in §1–§2 used *hand-drawn* fake marks (a synthetic ruler, an ink dot, a smooth dark
vignette). We re-ran the **same tests** on a **new, more realistic set of marks**, and kept the old
results above for comparison. The new set:

- **ruler** — a *real* dermoscopy-ruler photo (a PNG overlay) pasted on at a random spot, size, and angle.
- **arrow** — a *real* annotation arrow pasted on, pointing at the lesion.
- **black_corner** — a **hard circular cutoff**: everything outside a centred circle is turned solid
  black with a sharp edge (no fade), mimicking a real dermoscope's round field-of-view. (The old
  smooth vignette is still available under the name `dark_corner`.)

Same pipeline, same metrics, config-driven (`configs/default.yaml` → `artifacts.types`); driver
`scripts/run_new_artifact_grid.sh`.

### 5.1 How each new mark bends the answer (Phase-0 gate, ρ=1, α=1, layer 12)

| artifact | zero-shot `e_in` (median, CI95) | direction | induced-probe **bias_gap** |
|---|---|---|---|
| ruler | **+0.14** [+0.08, +0.25] | → **melanoma** (72% of images) | 0.19 |
| arrow | **+0.71** [+0.58, +0.83] | → **melanoma** (93%) | 0.68 |
| black_corner | **+0.72** [+0.53, +0.87] | → **melanoma** (88%) | **1.00** |

What changed vs. the old set:
- **All three now push toward *melanoma*** (positive `e_in`). Note the flip: the *old synthetic*
  ruler pushed toward benign (−0.51), but the *real* ruler photo pushes the other way (+0.14).
- **The real ruler is a weak, subtle cue** (small effect, bias_gap only 0.19) — real rulers are thin
  and sit off to the side, so the model leans on them far less than on the bold synthetic one.
- **black_corner is a very strong cue** (bias_gap 1.00 — the model's call is basically decided by
  whether the corners are black). A big, blunt, whole-image change is the easiest thing to latch onto.

### 5.2 Detecting vs. controlling — same benchmark, new marks

Setup as in §2 (ρ=0.9, α=1.0, layer 12, width-16384 SAE, 250 eval images, 3 seeds, same images/splits
for every method). `R` = recovery (1 = fully undoes the mark's effect, 0 = does nothing). The three R
values = ruler / arrow / black_corner.

| method | detection AUROC | **recovery R** (mean±std) | selectivity | off-target |
|---|---|---|---|---|
| **input-removal oracle** (ceiling) | — | **1.00 ± 0.00** | 1.00 | 0.00 |
| **SAE oracle-ablate** (ours) | 0.83–1.00 | **≈0** (+.00 / +.00 / +.00) | 0.91–1.00 | 0.00 |
| **raw neuron** (budget-matched) | 0.75–1.00 | **≈0** (+.00 / −.02 / +.02) | 0.44–0.50 | ~0–.02 |
| **CAV** (Reveal2Revise) | 1.000 | **≈0, worse on black** (−.04 / +.02 / **−.16**) | 0.47–0.81 | ~0 |
| **DermFM-Zero** top-5 (incumbent) | — | **erratic / harmful** (**−1.25** / −.39 / −.04) | 0.40–0.50 | **0.13** |
| random raw (control) | — | ≈0 | ~0.49 | ~0 |

**Headline: the same result holds.** Every interpretability tool **detects** the mark well (AUROC up
to 1.0), but **none can undo it** (R ≈ 0), while simply erasing the mark from the image undoes it
perfectly (R = 1.0). Detecting ≠ controlling — now confirmed on realistic photo overlays and a
dermoscope-style black circle, not just hand-drawn marks.

Extra notes:
- **Nudging the SAE feature (steering) doesn't help — it overshoots.** Turning the feature down harder
  pushes the answer *past* clean into the wrong direction (e.g. ruler at the strongest setting:
  R = **−2.7**; black_corner: R = **−0.8**). So neither switching the feature off nor nudging it
  recovers the clean decision.
- **CAV actively makes black_corner *worse*** (R = −0.16): a strong whole-image change drags the
  learned direction, so the "fix" overcorrects.
- **The real ruler is hard even to detect** (AUROC as low as 0.75–0.83, vs ~0.99 for the bold
  synthetic one) — a weaker mark is both a weaker cause and a fainter signal.
- **The SAE was healthy:** R² ≈ 0.99, ~22% dead features (width 16384, k=32), consistent across all 3 seeds.

Figures: `experiments/runs/20260708T233117Z_baselines_grid/figures/` (`detection_vs_recovery.png`,
`recovery_bars.png`, `selectivity_vs_offtarget.png`, `detection_bars.png`). Run dirs (2026-07-08, kept,
nothing overwritten): Phase-0 `…224940Z_phase0_gate`, interventions `…225108Z_interventions_grid`,
baselines `…233117Z_baselines_grid`. Code: commit `403377e` on `phase6-baselines` (the run dirs record
`git_commit: nogit` because git isn't installed inside the run container). Width note: this grid uses
the scripts' width-16384 setting to stay comparable to §2; the PREREGISTRATION-frozen width is 4096.

### 5.3 Fair-shot stress test — more features? best-case steering? (added 2026-07-09)

The obvious challenge to "detects but can't control" is: *"you only switched off ONE feature, and
you picked a bad steering strength."* So we gave the interpretability tools their **fairest shot**
and re-ran the new artifact set (ρ=0.9, α=1.0, width-16384, 3 seeds):

- **#1 — switch off more features.** Instead of the single best feature, ablate the top-**m** artifact
  features together, for m = 1, 3, 5, 10, 20.
- **#2 — best-case steering.** For steering, don't fix one strength: give each image its *own* best
  steering strength — the one that drives that image's answer closest to clean (a per-image **oracle**
  coefficient). No real single-strength method can beat this; it's an upper bound.

**#1 — ablating more features (recovery R, mean±std over seeds; selectivity in italics):**

| artifact | m=1 | m=5 | m=20 | selectivity m=1→m=20 |
|---|---|---|---|---|
| arrow | +0.00 | +0.04 | **+0.12** ±0.06 | *0.99 → 0.91* |
| ruler | +0.00 | +0.02 | **+0.07** ±0.01 | *0.87 → 0.82* |
| black_corner | +0.00 | −0.03 | **−0.01** ±0.05 | *1.00 → 0.98* |

Switching off 20× as many features nudges recovery to **at most +0.12** (arrow) and does nothing for
black_corner — still nowhere near the input-removal ceiling of **1.0** — while **selectivity erodes**
(you start damaging clean images). More features ≠ control.

**#2 — steering (recovery R):**

| artifact | fixed-strength curve (peak → overshoot) | **best-case** (per-image oracle) |
|---|---|---|
| arrow | peaks +0.10 (c≈1) → −0.38 (c=16) | **+0.78** ±0.09  (*sel 0.66*) |
| ruler | −0.12 (c=0.5) → **−3.4** (c=16) | **+0.33** ±0.44  (*sel 0.54*) |
| black_corner | +0.01 (c=0.5) → −0.78 (c=8) | **+0.29** ±0.32  (*sel 0.45*) |

No single steering strength recovers the answer: the curve barely rises then **overshoots hard past
clean into the wrong direction** (ruler → −3.4). Even the **best-case per-image oracle** strength — an
upper bound no real method can reach — recovers only **+0.78 (arrow) / +0.33 (ruler) / +0.29
(black_corner)**, never the full 1.0, is **unreliable across seeds** (ruler ±0.44), and gets there only
by **wrecking selectivity** (0.45–0.66 vs 0.87–0.99 for ablation).

🔁 **Takeaway:** giving interpretability its fairest shot — 20× the features, or a per-image oracle
steering strength — **still does not give reliable, selective control.** Detecting ≠ controlling
survives the stress test.

Run dir (2026-07-09, kept): `experiments/runs/20260709T133850Z_interventions_grid` (144 cells + 3-seed
`summary_m_sweep_bestcase.csv`). Code: `scripts/40_run_interventions.py` + `carve.eval.harness.run_steer_bestcase`,
config-driven via `interventions.feature_set_sizes` / `interventions.steer_grid`. Ruler overlay also
enlarged this day to 60–100% of the image on a tangential orbit (see `src/carve/data/artifacts.py`).

---

## Immediate next steps

1. **Finish the SAE-health sweep** — `scripts/31_sae_health_sweep.py` got cut off (battery). It
   checks dead-feature fraction, R², detection, and decoder stability for {4096, 16384,
   16384+AuxK}, and whether the frozen width rule prefers a wider dictionary. Run it with
   `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` when the box has no network.
2. **Phase-7 final grid** — the ρ×α sweep (right now fixed at ρ=0.9/α=1.0), all methods, using the
   **frozen** SAE config (width 4096, AuxK). PREREGISTRATION is frozen, so this is cleared to run.
3. **CDEP** (task #12) — the last baseline, if the deadline allows.

_Branches:_ Stage-6 + prereg on `phase6-baselines`
(`b97d368`, `e8e1981`, `9c95270`, `0840de8`, `5d57d94`, + this docs commit).
Not pushed — waiting on the PI, per repo git rules.
