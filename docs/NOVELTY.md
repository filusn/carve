# CARVE — Novelty & Scoop Gate

Durable record of the pre-code novelty check. Update if a closer competitor appears.

## Gate #1 — DermFM-Zero (arXiv 2602.10624) — 2026-06-30

**Question:** does DermFM-Zero already do CARVE's controlled causal-validation
contribution? **Method:** two independent reads — (a) full paper deep-read, (b) GitHub
repo + appendix code inspection — each required to quote evidence and render its own
verdict. Both converged.

**Verdict: LOW overlap → PROCEED.**

### What DermFM-Zero actually does on artifacts
A "discover-then-name" SAE pipeline on the frozen vision encoder, feeding an SAE-Concept
Bottleneck Model. Artifact handling is one results paragraph + Fig. 6i:
> "we identified the top five neurons most strongly activated by each artifact type and
> suppressed their activations at inference time ... AUROC increases ranging from 12% to
> 38% across artifact categories."

Operationally (confirmed in released code, `4_intervention_CBM.py`):
`activations_masked[:, neuron_indices] = 0` on **natural** ISIC artifact subsets
(rulers n=500, purple pen markings n=246, hair n=649), assembled into biased splits
(`ISIC_*_bias`), scored by before/after diagnostic AUROC. No injection, no opacity, no
correlation-strength knob, no baselines, no selectivity, no off-target — in paper or repo.

### D1–D7 (both passes agree)
| # | CARVE ingredient | In DermFM-Zero? |
|---|---|---|
| D1 | Controlled injection (ρ / opacity, synthetic GT) | ABSENT |
| D2 | Causal-recovery vs. input paste/remove gold standard | ABSENT |
| D3 | Selectivity (contaminated vs. clean) | ABSENT |
| D4 | Off-target damage of steering | ABSENT |
| D5 | Head-to-head baselines (CAV/R2R/CRP/CDEP/raw-neuron/random) | ABSENT |
| D6 | Intervention tied to downstream diagnosis | **PRESENT** (their 12–38% AUROC gain) |
| D7 | Zero-shot input-intervention bias measurement | ABSENT |

### Binding consequences for CARVE
- **Do NOT claim** "suppressing SAE artifact-features improves dermatology diagnosis" —
  DermFM-Zero published it. Cite them for it.
- **DO claim** the controlled causal **validation/benchmark**: injected known cause →
  recovery / selectivity / off-target → head-to-head methods. That is untouched.
- **Closest real overlap to differentiate against:** they train a biased CBM and measure
  AUROC once before/after suppression. CARVE adds the controlled cause (injection + ρ
  sweep), a per-image input-level counterfactual gold standard, the full metric suite, and
  competing baselines. State this delta explicitly.
- **Add DermFM-Zero-style top-k neuron suppression as a BASELINE** in our benchmark
  (reimplement from the paper description — do not fork: code is CC-BY-NC-ND, weights
  private/on-request). Benchmarking their own method on controlled GT also disarms the
  obvious reviewer.
- **Primary FM = MONET** (public weights). DermFM-Zero weights are NOT available, so we
  cannot and need not build on it.
- **Reviewer risk (not overlap):** likely reviewed by the same group (Yan/Tschandl/
  Kittler/Ge). Mitigation: cite DermFM-Zero generously as the motivating proof-of-concept;
  frame CARVE as the rigorous validation its one-way AUROC number lacks.

### Integrity caveat
Their suppression numbers could not be re-verified (weights private); we cite the published
claim as-is and do not reproduce it. Quotes above are from arXiv HTML v1 + raw GitHub files
(fetched 2026-06-30).

### Still-open scoop-watch
Re-run before submission: controlled/synthetic SAE-feature validation in *medical* imaging;
causal recovery / selectivity benchmarks tied to clinical decisions; any new MONET-artifact
audit. See `RELATED_WORK.md` searches.
