# CARVE — project brief

**CARVE (Causal ARtifact Validation of Encodings)** — a benchmark testing whether
sparse-autoencoder (SAE) interpretability can *causally control* a medical foundation
model's known failure mode, not merely correlate with it.

## TL;DR
Dermatology models cheat on non-diagnostic artifacts (rulers, pen ink, dark corners).
People use interpretability tools to find and switch off the "artifact feature," but on real
data you can't verify you found the *right* one. CARVE **injects** the artifact itself — so
its true causal effect is known — then measures whether intervening on an SAE feature
**recovers** that effect, **selectively**, and **without collateral damage**, benchmarked
head-to-head against the established methods. Because we own the ground truth, there is
always a reportable result.

## The problem
Skin-cancer classifiers latch onto artifacts instead of the lesion (e.g. Winkler 2019:
adding surgical ink dropped specificity 84%→46%; in ISIC, colored patches sit on ~46% of
benign and 0% of malignant images). The field now uses SAEs / concept vectors to find and
suppress the responsible feature — but on natural images the artifact's true causal effect
is unknown, so claims stay correlational.

## The idea (the one trick)
We **inject** the artifact at a controlled strength and opacity. Because we paste it in, we
know its true causal effect — just remove it from the input and measure the change. That is
our gold-standard "truth meter." We then intervene on the candidate SAE feature and ask:
- **Recovery** — does flipping the feature reproduce the input-level effect?
- **Selectivity** — does it change only contaminated cases, not clean ones?
- **Off-target** — what else breaks?

We run this for SAE steering **and** the incumbents — CAV/Reveal2Revise, CDEP, raw-neuron
ablation, DermFM-Zero-style top-k suppression, plus random-feature and input-removal controls
— on the **same** ground truth, and report who wins, honestly.

## Why it's worth doing
- **Low risk:** we own the ground truth, so a result is guaranteed — even "SAEs don't beat a
  single neuron" is a publishable finding.
- **Novel:** prior work (incl. DermFM-Zero, Feb 2026) *uses* SAEs to suppress artifacts and
  shows accuracy improves — an *application*. Nobody has *validated* whether the
  interpretability claim is causally true, how selective it is, or which method wins. CARVE
  is that controlled validation/benchmark. (We did a two-pass overlap check: LOW overlap.)
- **Fits the thesis:** reliable, auditable visual decision support.

## Setup
- **Data:** Bissoto-labelled ISIC 2018 + Atlas (artifact labels: ruler/ink/hair/…) as core —
  gives both a verified-clean injection canvas and a real-artifact reality-check slice;
  HAM10000 (melanoma-vs-nevus) as the larger reservoir. **Public + synthetic only; no
  private data, no new annotation.**
- **Model:** MONET (dermatology CLIP, public weights) primary; CLIP ViT-B/16 as a generality
  check (ships pretrained SAEs).
- **Method:** TopK SAE on frozen features; activation patching / steering via ViT-Prisma.
- **Two bias arms:** zero-shot (no training — purest causal ground truth) + induced
  (controlled spurious-correlation strength ρ).

## Target
MI4MedFM @ MICCAI 2026 workshop, **Track 1 (8–10 pp, archival)**. Deadline **15 Jul 2026**.

## Integrity guardrails
Pre-register before final runs; disjoint splits (no leakage); fair baselines (we benchmark
the rivals' own method); negative results are valid; synthetic→real claims stay bounded; no
fabricated numbers or citations.

## Work modules (parallelizable once the injector lands)
1. Data loaders + artifact injector (`inject`/`remove`, deterministic) + seeded disjoint splits
2. Encoder/probe + bias measurement (zero-shot & induced arms)
3. SAE training + feature discovery (oracle vs. unsupervised)
4. Interventions + metrics harness (recovery / selectivity / off-target)
5. Baselines (CAV/CDEP/raw-neuron/DermFM-Zero/random/oracle)
6. Aggregation + figures + real-artifact slice

Full detail: `docs/EXECUTION_PLAN.md`, `docs/DATASETS.md`, `docs/METRICS.md`,
`docs/INTEGRITY.md`, `docs/NOVELTY.md`.
