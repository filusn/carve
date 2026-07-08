# CARVE — Novelty & Scoop Gate

A lasting record of the "has someone already done this?" check we ran before writing code.
Update it if a closer competitor shows up.

## Gate #1 — DermFM-Zero (arXiv 2602.10624) — 2026-06-30

**Question:** does DermFM-Zero already do CARVE's controlled cause-and-effect check?
**Method:** two independent reads — (a) a deep read of the full paper, (b) a look at the GitHub
repo + appendix code — each one had to quote evidence and reach its own verdict. Both agreed.

**Verdict: LOW overlap → PROCEED.** (Meaning: they didn't do our thing. Go ahead.)

### What DermFM-Zero actually does with artifacts
It runs a "find-then-name" SAE pipeline on the frozen vision encoder, feeding an SAE-Concept
Bottleneck Model. Artifacts get one results paragraph + Fig. 6i:
> "we identified the top five neurons most strongly activated by each artifact type and
> suppressed their activations at inference time ... AUROC increases ranging from 12% to
> 38% across artifact categories."

In practice (confirmed in their released code, `4_intervention_CBM.py`): they do
`activations_masked[:, neuron_indices] = 0` on **real** ISIC artifact subsets (rulers n=500,
purple pen markings n=246, hair n=649), bundled into biased splits (`ISIC_*_bias`), and scored by
before/after diagnostic AUROC. No pasting-in of artifacts, no opacity knob, no
correlation-strength knob, no baselines, no selectivity, no off-target — not in the paper, not in
the repo.

### D1–D7 (both passes agree)
| # | CARVE ingredient | In DermFM-Zero? |
|---|---|---|
| D1 | Controlled paste-in (ρ / opacity, synthetic answer key) | ABSENT |
| D2 | Recovery graded against a paste/remove gold standard | ABSENT |
| D3 | Selectivity (contaminated vs. clean) | ABSENT |
| D4 | Off-target damage from steering | ABSENT |
| D5 | Head-to-head baselines (CAV/R2R/CRP/CDEP/raw-neuron/random) | ABSENT |
| D6 | Intervention tied to the final diagnosis | **PRESENT** (their 12–38% AUROC gain) |
| D7 | Zero-shot input-level bias measurement | ABSENT |

### What this means for CARVE (binding)
- **Do NOT claim** "muting SAE artifact-features improves dermatology diagnosis" — DermFM-Zero
  already published that. Cite them for it.
- **DO claim** the controlled cause-and-effect **check/benchmark**: paste in a known cause →
  measure recovery / selectivity / off-target → race the methods head-to-head. That part is
  untouched.
- **Closest real overlap to set ourselves apart from:** they train one biased model and measure
  AUROC once, before vs. after muting. CARVE adds the controlled cause (paste-in + ρ sweep), a
  per-image answer key at the input level, the full set of metrics, and competing baselines. Say
  this difference out loud.
- **Add DermFM-Zero-style top-k neuron muting as a BASELINE** in our benchmark (rebuild it from
  the paper description — don't fork their code: it's CC-BY-NC-ND, and the weights are
  private/on-request). Testing their own method on our controlled answer key also heads off the
  obvious reviewer complaint.
- **Main model = MONET** (public weights). DermFM-Zero's weights aren't available, so we can't and
  don't need to build on it.
- **Reviewer risk (not an overlap issue):** we may well be reviewed by the same group
  (Yan/Tschandl/Kittler/Ge). Plan: cite DermFM-Zero generously as the motivating proof-of-concept,
  and frame CARVE as the rigorous check that their one-way AUROC number is missing.

### Integrity caveat
We couldn't re-verify their muting numbers (weights are private); we cite the published claim
as-is and don't reproduce it. The quotes above are from the arXiv HTML v1 + the raw GitHub files
(fetched 2026-06-30).

### Still-open scoop-watch
Re-run before we submit: controlled/synthetic SAE-feature checks in *medical* imaging; recovery /
selectivity benchmarks tied to clinical decisions; any new MONET-artifact audit. See
`RELATED_WORK.md` searches.
