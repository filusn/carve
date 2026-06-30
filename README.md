# CARVE - Causal ARtifact Validation of Encodings

*A controlled benchmark for the causal faithfulness of sparse-autoencoder (SAE)
interpretability in medical foundation models.*

## The problem

A growing body of work trains SAEs on medical image encoders and reports
"interpretable" features (e.g., a *dermoscopy-ruler* feature). Almost none **prove that such a feature is *causally* the artifact** - that ablating/steering it actually removes
the model's reliance on that artifact, selectively, without collateral damage. Existing
artifact-mitigation work (CAV/Reveal2Revise, CDEP) and the closest SAE-derm work
(DermFM-Zero) validate against *natural* data with no known causal ground truth.

## The idea

**Inject the artifact yourself.** If we paste a controlled artifact (ruler, marker-ink,
dark-corner, text overlay) into dermatology images at a *known* correlation strength and
opacity, then the artifact's **true causal effect** on the model is measurable by
intervening on the input (paste vs. remove). We can then ask, for the first time under
known ground truth:

1. **Recovery** - does ablating the candidate SAE artifact-feature reproduce the
   input-level removal of the artifact? (mediation)
2. **Selectivity** - does it change predictions on artifact-contaminated cases but *not*
   on clean cases?
3. **Off-target damage** - how much clean-task performance is lost?

…and does the SAE approach beat **CAV/Reveal2Revise, CDEP, raw-neuron ablation**, a
**random-feature control**, and the **input-removal oracle** (ceiling) on the *same*
ground truth? We release the injector, splits, and eval harness as a reusable benchmark.

This is a benchmark contribution: it succeeds by being **correct, fair, and reusable**,
not by our method winning. A clean negative result is a valid outcome.

## Quickstart

```bash
uv venv
uv pip install -r requirements.txt
python scripts/01_smoke_test.py      # Phase-0 day-1 go/no-go
```
Real runs require a CUDA GPU. See `CLAUDE.md` for environment details.

## Layout

```
carve/
├── CLAUDE.md            # agent operating rules (integrity-first)
├── PREREGISTRATION.md   # freeze metrics/thresholds before final runs
├── configs/             # config-driven everything
├── docs/                # EXECUTION_PLAN, METRICS, INTEGRITY, RELATED_WORK, DATASETS
├── src/carve/           # data · models · sae · interventions · baselines · metrics · eval
├── scripts/             # NN_*.py pipeline entrypoints
├── experiments/         # run outputs (git-ignored) + result tables
└── tests/
```

## Data & licenses

Public dermatology data only (HAM10000/ISIC, Derm7pt, Fitzpatrick17k) + synthetic
injection. We **redistribute injection recipes/code, not images.** See `docs/DATASETS.md`
for sources and license terms (HAM10000 is CC BY-NC).

## Integrity

CARVE's value is methodological honesty. Pre-registration, disjoint splits, fair
baselines, multi-seed variance, an explicit synthetic→real claim boundary, and full
reproducibility are required - see `docs/INTEGRITY.md`.

## Citation

TBD upon submission.
