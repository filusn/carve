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
