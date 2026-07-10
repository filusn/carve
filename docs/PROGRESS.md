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
  **new artifact set** — real ruler + arrow photo overlays and a realistic soft dermoscope-corner
  vignette (`black_corner`). See **§5** (originals in §1–§2 kept for comparison).
- 🛡️ **Survives a fair-shot stress test:** even switching off 20× more features, or handing steering a
  per-image *best-case* strength (an upper bound no real method beats), still doesn't recover the answer
  (control caps at ~0.1 for multi-feature ablation, ~0.3–0.8 for oracle steering, vs 1.0 for erasing the
  mark) — and only by hurting selectivity. Detecting ≠ controlling. See **§5.3**.
- ✅ **Confirmed grid-wide:** the pre-registered ρ×α sweep (varying correlation strength *and*
  artifact opacity) shows detection stays high and recovery stays ≈0 in *every* cell — see **§6**.
- 🔬 **We know WHY now:** the artifact's causal effect is essentially *one* activation direction
  (rank ≈1), but the feature/CAV a tool picks to *detect* it is nearly orthogonal (|cos| ≈ 0.02–0.12)
  to that causal direction — so it ablates the wrong vector. Detection ≠ control, literally. See **§7**.
- 🚧 **Not done yet:** CDEP baseline (deferred — it is a *training-time* method, out of our
  inference-time scope; noted as future work), and a leftover SAE-health check.

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
- **black_corner** — a **realistic dermoscope field-of-view**: circular optics on a rectangular
  sensor, so a *soft* vignette darkens only the corners that fall outside the large optical circle
  (off-centre ⇒ 2 corners, centred ⇒ 4), with coverage capped at **≤2.8%** of the image.

Same pipeline, same metrics, config-driven (`configs/default.yaml` → `artifacts.types`); driver
`scripts/run_new_artifact_grid.sh`.

> **black_corner redesign (2026-07-09).** The first version was a *hard inscribed-circle* cutoff that
> blacked out ~21% of the image — an unrealistically blunt cue that saturated the bias (bias_gap 1.0).
> Prompted by checking the training set (only **1.9%** of the 10,015 HAM10000 images carry any real
> dermoscope vignette, and a *full* circle is essentially absent — **0.03%**), we replaced it with the
> physically faithful **circular-optics-over-sensor** model above (soft, ≤2.8% coverage). The old hard
> disc is kept for reference as `black_corner_circle`; the smooth gradient vignette is `dark_corner`.
> §5.1–§5.3 below are the **re-run on this realistic version**: the zero-shot effect is far weaker
> (§5.1), yet "detection ≠ control" still holds (§5.2–§5.3).

### 5.1 How each new mark bends the answer (Phase-0 gate, ρ=1, α=1, layer 12)

| artifact | zero-shot `e_in` (median, CI95) | direction | induced-probe **bias_gap** |
|---|---|---|---|
| ruler | **+0.18** [+0.09, +0.30] | → **melanoma** (68% of images) | 0.32 |
| arrow | **+0.71** [+0.58, +0.83] | → **melanoma** (93%) | 0.68 |
| black_corner | **+0.23** [+0.14, +0.32] | → **melanoma** (68%) | 0.87 |

What we found:
- **All three push toward *melanoma*** (positive `e_in`). (The *old synthetic* ruler in §1 pushed the
  other way, toward benign −0.51; the *real* ruler photo flips it to +0.18.)
- **The real ruler is a weak, subtle cue** (small effect, bias_gap 0.32) — real rulers are thin and
  sit off to the side, so the model leans on them far less than on the bold synthetic one.
- **The realistic black_corner is now only a *moderate* cue.** Making the vignette faithful
  **collapsed its zero-shot effect from +0.72 (old blunt disc) to +0.23** — MONET's own call barely
  moves. But a *trained probe* still exploits it heavily (bias_gap 0.87): a soft corner darkening is a
  reliable, easily-detected tag, so a lazy classifier grabs it even though it shifts the raw decision
  little. So the old disc's "total bias" was mostly an artifact of its bluntness.

### 5.2 Detecting vs. controlling — same benchmark, new marks

Setup as in §2 (ρ=0.9, α=1.0, layer 12, width-16384 SAE, 250 eval images, 3 seeds, same images/splits
for every method). `R` = recovery (1 = fully undoes the mark's effect, 0 = does nothing). The three R
values = ruler / arrow / black_corner.

| method | detection AUROC | **recovery R** (mean±std) | selectivity | off-target |
|---|---|---|---|---|
| **input-removal oracle** (ceiling) | — | **1.00 ± 0.00** | 1.00 | 0.00 |
| **SAE oracle-ablate** (ours) | 0.88–0.99 | **≈0** (+.00 / +.00 / +.03) | 0.87–0.99 | 0.00 |
| **raw neuron** (budget-matched) | 0.82–0.95 | **≈0** (−.01 / −.01 / −.01) | 0.48–0.52 | ~0 |
| **CAV** (Reveal2Revise) | 1.000 | **≈0** (−.05 / +.02 / +.05) | 0.52–0.69 | ~0–.03 |
| **DermFM-Zero** top-5 (incumbent) | — | **erratic / harmful** (**−1.07** / −.39 / **−.92**) | 0.46–0.50 | **0.13** |
| random raw (control) | — | ≈0 (+.00 / +.00 / +.00) | 0.27–0.35 | ~0 |

**Headline: the same result holds.** Every interpretability tool **detects** the mark well (AUROC up
to 1.0), but **none can undo it** (R ≈ 0), while simply erasing the mark from the image undoes it
perfectly (R = 1.0). Detecting ≠ controlling — now confirmed on realistic photo overlays *and* a
physically faithful soft dermoscope vignette, not just hand-drawn marks.

Extra notes:
- **Nudging the SAE feature (steering) doesn't help — it overshoots.** Turning the feature down harder
  pushes the answer *past* clean into the wrong direction (at the strongest setting c=16: ruler
  R = **−3.4**, black_corner R = **−2.2**; arrow peaks at only +0.10 then falls to −0.38). So neither
  switching the feature off nor nudging it recovers the clean decision.
- **The realistic black_corner is no longer a special case.** On the old blunt disc CAV *worsened* it
  (R = −0.16); on the soft version every straight-line method is back to R ≈ 0 (SAE +.03, CAV +.05,
  raw −.01) — detectable (AUROC 0.95–1.0) but not controllable, exactly like ruler and arrow.
- **The real ruler is the hardest to detect** (AUROC 0.82–0.88, vs 0.96–1.0 for arrow/black_corner) —
  a weaker mark is both a weaker cause and a fainter signal.
- **The SAE was healthy:** R² ≈ 0.991, ~22–23% dead features (width 16384, k=32), consistent across all 3 seeds.

Figures: `experiments/runs/20260709T195504Z_baselines_grid/figures/` (`detection_vs_recovery.png`,
`recovery_bars.png`, `selectivity_vs_offtarget.png`, `detection_bars.png`). Run dirs (2026-07-09, kept,
nothing overwritten): Phase-0 `…185640Z_phase0_gate`, interventions `…185808Z_interventions_grid`,
baselines `…195504Z_baselines_grid`. Code: `phase6-baselines` (the run dirs record `git_commit: nogit`
because git isn't installed inside the run container). Width note: this grid uses the scripts'
width-16384 setting to stay comparable to §2; the PREREGISTRATION-frozen width is 4096.

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
| black_corner | +0.03 | +0.07 | **+0.14** ±0.05 | *0.95 → 0.92* |

Switching off 20× as many features nudges recovery to **at most +0.14** (black_corner) — still nowhere
near the input-removal ceiling of **1.0** — while **selectivity erodes** (you start damaging clean
images). More features ≠ control.

**#2 — steering (recovery R):**

| artifact | fixed-strength curve (peak → overshoot) | **best-case** (per-image oracle) |
|---|---|---|
| arrow | peaks +0.10 (c≈1) → −0.38 (c=16) | **+0.78** ±0.09  (*sel 0.66*) |
| ruler | −0.12 (c=0.5) → **−3.4** (c=16) | **+0.33** ±0.44  (*sel 0.54*) |
| black_corner | +0.01 (c=0.5) → **−2.2** (c=16) | **+0.50** ±0.21  (*sel 0.56*) |

No single steering strength recovers the answer: the curve barely rises then **overshoots hard past
clean into the wrong direction** (ruler → −3.4, black_corner → −2.2). Even the **best-case per-image
oracle** strength — an upper bound no real method can reach — recovers only **+0.78 (arrow) / +0.50
(black_corner) / +0.33 (ruler)**, never the full 1.0, is **unreliable across seeds** (ruler ±0.44), and
gets there only by **wrecking selectivity** (0.54–0.66 vs 0.87–0.99 for ablation).

🔁 **Takeaway:** giving interpretability its fairest shot — 20× the features, or a per-image oracle
steering strength — **still does not give reliable, selective control.** Detecting ≠ controlling
survives the stress test, now on the realistic soft black_corner too.

Run dir (2026-07-09, kept): `experiments/runs/20260709T185808Z_interventions_grid`. Code:
`scripts/40_run_interventions.py` + `carve.eval.harness.run_steer_bestcase`, config-driven via
`interventions.feature_set_sizes` / `interventions.steer_grid`. Same-day artifact work (see
`src/carve/data/artifacts.py`): ruler enlarged to 60–100% on a tangential orbit; black_corner replaced
by the realistic soft circular-optics vignette (≤2.8% cov), old hard disc kept as `black_corner_circle`.

---

## 6. Phase-7 final grid — the pre-registered ρ×α sweep (added 2026-07-10)

Everything above sits at a single operating point (ρ=0.9, α=1.0). The **pre-registered confirmatory
experiment** (PREREGISTRATION §1) is the ρ×α sweep: does "detection ≠ control" hold across the whole
grid of spurious-correlation strength ρ and opacity α? **It does.**

**Width note — resolves the prereg-compliance question.** PREREG §4 freezes the SAE width by a *rule*:
the widest dictionary with **≤15% dead features** and R²≥0.98. The §1–§5 grids ran at width **16384**,
which is **~22% dead — it FAILS that bar**. This confirmatory sweep therefore runs at the rule-compliant
**width 4096** (dead **6.9–8.9%** across seeds, R²=0.990). So §5's 16384 numbers are now labelled
**exploratory / robustness**; §6 (4096) is the frozen-compliant headline. (The dissociation is identical
at both widths.) A direct width sweep at the grid config (AuxK-on, k=32, 3000 steps) confirms 4096 is
the *widest* ≤15%-dead width — dead-feature fraction **4096 → 8.7%, 8192 → 18.1%, 16384 → 23.2%** (only
4096 clears the bar), so the frozen "widest ≤15%-dead" rule selects 4096 unambiguously.

Setup: pre-registered grid ρ∈{0.5,0.7,0.9,1.0} × α∈{0.4,0.7,1.0} (12 cells) × 3 artifacts × 3 seeds.
The SAE trains once per seed on clean activations (ρ/α-independent) and is reused across the grid. Per
cell: SAE oracle-ablate **detection AUROC** + **recovery R**, plus the input-removal oracle (R≡1).

**Detection stays high across the whole grid** (rising with opacity α):

| artifact | α=0.4 | α=0.7 | α=1.0 |
|---|---|---|---|
| ruler | 0.78–0.85 | 0.84–0.87 | 0.88–0.89 |
| arrow | 0.88–0.90 | 0.95–0.96 | 0.97–0.98 |
| black_corner | 0.95–0.97 | 0.97–0.99 | 0.98–0.99 |

**Recovery stays ≈0 in every one of the 12 cells** (SAE oracle-ablate R, 3-seed mean; input oracle
= 1.00 in every cell):

| artifact | recovery R across all (ρ,α) |
|---|---|
| ruler | **0.00** (exactly, every cell) |
| arrow | **+0.005 … +0.021** |
| black_corner | **−0.005 … +0.029** |

🔁 **Confirmatory takeaway:** the dissociation is **grid-wide** — at *every* combination of correlation
strength and opacity, the SAE feature detects the artifact (AUROC 0.78–0.99) but ablating it recovers
essentially none of the effect (R ≈ 0), while erasing the mark recovers it perfectly (R = 1.0). The
result is not an artifact of one operating point.

Figure `…rho_alpha_sweep/figures/rho_alpha_dissociation.png`. Run dir (2026-07-10, kept):
`experiments/runs/20260709T213446Z_rho_alpha_sweep` (216 cells, 3 seeds, width 4096). Code:
`scripts/41_rho_alpha_sweep.py`.

---

## 7. Mechanism — WHY linear control fails (added 2026-07-10)

§5–§6 establish that every method detects the artifact but can't control it (R ≈ 0). §7 asks *why*,
and finds a clean geometric answer. At block ℓ=12 the artifact moves the activation by Δa =
a(x_art) − a(x_clean) (per image, per token). We measure (a) the **effective rank** of Δa
(participation ratio of its singular values) and (b) the **alignment** (|cos|) between the artifact's
**causal direction** (top singular vector of Δa) and the **detection directions the tools actually
ablate**: the SAE feature with the highest present/absent AUROC, and the CAV. Frozen-compliant width
4096, 3 seeds.

**Finding 1 — the causal effect is essentially rank-1.** Δa has participation ratio **1.1–1.4** (its
top singular direction carries **84–94%** of the variance). The artifact is *not* smeared across many
dimensions — it moves the activation along essentially one direction, so removing that one direction
should suffice. (This refuted our initial "the effect is high-dimensional" guess.)

**Finding 2 — but the detection direction is a DIFFERENT vector than the causal one.** The SAE feature
that best *detects* the artifact (AUROC 0.89–0.99) is **nearly orthogonal** to the causal direction
(|cos| = **0.05–0.12**); the CAV is essentially orthogonal (|cos| ≈ **0.02**). Even the *best-aligned*
feature in the entire 4096-feature dictionary reaches only |cos| ≈ **0.63**.

| artifact | eff. rank (top-dir var) | detection AUROC | \|cos(causal, SAE feat)\| | \|cos(causal, CAV)\| | best \|cos\| any feat |
|---|---|---|---|---|---|
| ruler | 1.18 (91%) | 0.895 | **0.12** | **0.02** | 0.63 |
| arrow | 1.14 (94%) | 0.974 | **0.06** | **0.02** | 0.63 |
| black_corner | 1.38 (85%) | 0.989 | **0.05** | **0.02** | 0.63 |

🔬 **Mechanism:** "detection ≠ control" is literally true **at the level of vectors.** The artifact's
causal effect is a single activation direction, but the direction a tool picks to *detect* the artifact
is not that direction — so ablating the detection feature (or the CAV) removes the *wrong* direction and
barely moves the decision (R ≈ 0). Detection asks "which direction best *separates* present/absent?";
control needs "which direction *carries the effect*?" — and these are different vectors. This is exactly
why erasing the artifact at the input (which removes the true causal component per image) recovers
perfectly, while every linear feature-space method fails.

Figure `…effect_dimensionality/figures/mechanism_detection_vs_causal_direction.png`. Run dir (2026-07-10,
kept): `experiments/runs/20260709T231727Z_effect_dimensionality`. Code: `scripts/42_effect_dimensionality.py`.

---

## 8. Robustness across layers (added 2026-07-10)

Reviewer-proofing against "you cherry-picked block 12." We re-ran the core dissociation **and** the §7
mechanism at blocks {6, 8, 10, 12}, training a fresh SAE (frozen width 4096) at each, 2 seeds
(supplementary; the main grid is 3-seed at block 12).

**Detection ≠ control holds at every layer.** Detection AUROC stays high (0.83–0.99) and SAE-ablate
recovery stays ≈0 (max +0.06) at all four blocks; the input-removal oracle is 1.0 throughout. The
dissociation is not specific to block 12.

| block ℓ | detection AUROC | SAE recovery R | eff. rank of Δa | \|cos(causal, SAE feat)\| |
|---|---|---|---|---|
| 6 | 0.83–0.98 | −0.00 … +0.05 | ~10–17 | 0.20–0.46 |
| 8 | 0.84–0.99 | +0.00 … +0.06 | ~11–17 | 0.18–0.40 |
| 10 | 0.87–0.99 | +0.00 … +0.05 | ~14–16 | 0.23–0.49 |
| 12 | 0.88–0.99 | −0.01 … +0.02 | **~1.2** | **0.05–0.11** |

**The geometric cause shifts with depth** (an honest nuance): at mid blocks the artifact effect is
*distributed* (effective rank ~10–17), so a single-feature ablation can't capture it even though the
detection feature is moderately aligned (cos ~0.2–0.5); by block 12 the effect *concentrates* to
rank-1 but the detection feature becomes near-orthogonal to it (cos ~0.05–0.11). Either way — too many
dimensions, or the wrong single one — a low-rank linear intervention fails to control the artifact.

*Caveat:* the SAE reconstructs mid blocks less faithfully (R² ~0.84 at ℓ6–10 vs 0.99 at ℓ12), which
also limits mid-block feature quality; block 12 (the pre-registered layer) is where the SAE is
healthiest and the dissociation is cleanest.

Figure `…layer_robustness/figures/layer_robustness.png`. Run dir (2026-07-10):
`experiments/runs/20260709T235403Z_layer_robustness` (2 seeds). Code: `scripts/43_layer_robustness.py`.

---

## Immediate next steps

1. ✅ **Phase-7 ρ×α sweep — DONE** (§6): confirmatory grid at the frozen-compliant width 4096;
   dissociation holds in every (ρ,α) cell. `scripts/41_rho_alpha_sweep.py`.
2. ✅ **Mechanism — DONE** (§7): rank-1 causal direction vs. near-orthogonal detection direction.
   `scripts/42_effect_dimensionality.py`.
3. **Manuscript** — the experimental spine is complete (bias → detection≠control → fair-shot → grid-wide
   → mechanism); the remaining critical-path work is writing. Initial draft in `docs/PAPER_DRAFT.md`.
4. **CDEP** — DEFERRED as future work: it is a *training-time* explanation penalty, outside CARVE's
   inference-time recovery framework; benchmarking it fairly needs a separate setup (note it in the
   paper's Related Work / Limitations rather than rushing an unfair reimplementation).
5. **SAE-health sweep** (`scripts/31_sae_health_sweep.py`, cut off) — would confirm 4096 is the widest
   ≤15%-dead width (8192 unchecked). Not blocking.

_Branches:_ Stage-6 + prereg on `phase6-baselines`
(`b97d368`, `e8e1981`, `9c95270`, `0840de8`, `5d57d94`, + this docs commit).
Not pushed — waiting on the PI, per repo git rules.
