# CARVE — project brief

**CARVE (Causal ARtifact Validation of Encodings)** — a test of whether interpretability tools
can truly *switch off* a medical AI model's known bad habit, not just *notice* it.

## TL;DR
Skin-cancer AI models often cheat: they read junk in the photo (rulers, pen ink, dark corners)
instead of the actual skin spot. People use interpretability tools to find the "junk feature"
inside the model and turn it off — but on real photos you can't prove you found the *right* one.
CARVE **pastes the junk in itself**, so its true effect is known. Then it checks whether turning
off a candidate feature really **undoes** that effect, does it **only where the junk is**, and
does it **without breaking anything else** — racing the popular tools side by side. Because we own
the answer key, we always get a result worth reporting.

(An SAE — sparse autoencoder — is a tool that splits the model's inner activity into many simple,
readable "features.")

## The problem
Skin-cancer classifiers grab onto artifacts instead of the lesion (e.g. Winkler 2019: adding
surgical ink dropped specificity 84%→46%; in ISIC, colored patches sit on ~46% of benign and 0%
of malignant images). The field now uses SAEs / concept vectors to find and mute the guilty
feature — but on natural photos the artifact's true effect is unknown, so the claims stay
correlational (a matching pattern, not proven cause).

## The idea (the one trick)
We **paste** the artifact in at a chosen strength and see-through-ness. Because we pasted it, we
know its true effect — just remove it from the photo and measure the change. That's our
gold-standard "truth meter." Then we act on the candidate SAE feature and ask:
- **Recovery** — does flipping the feature reproduce the effect we saw at the input?
- **Selectivity** — does it change only the contaminated photos, not the clean ones?
- **Off-target** — what else breaks?

We run this for SAE steering **and** the rivals — CAV/Reveal2Revise, CDEP, raw-neuron ablation,
DermFM-Zero-style top-k muting, plus random-feature and input-removal controls — all on the
**same** answer key, and report who wins, honestly.

## Why it's worth doing
- **Low risk:** we own the answer key, so a result is guaranteed — even "SAEs don't beat a single
  neuron" is worth publishing.
- **New:** earlier work (incl. DermFM-Zero, Feb 2026) *uses* SAEs to mute artifacts and shows
  accuracy goes up — an *application*. Nobody has *checked* whether the interpretability claim is
  actually true, how selective it is, or which method wins. CARVE is that controlled check. (We
  ran a two-pass overlap check: LOW overlap.)
- **Fits the thesis:** reliable, checkable visual decision support.

## Setup
- **Data:** Bissoto-labelled ISIC 2018 + Atlas (artifact labels: ruler/ink/hair/…) as the core —
  it gives both a verified-clean canvas to paste on and a real-artifact reality-check slice;
  HAM10000 (melanoma-vs-nevus) as the bigger pool. **Public + synthetic only; no private data, no
  new labeling.**
- **Model:** MONET (a dermatology CLIP model, public weights) as the main one; CLIP ViT-B/16 as a
  does-it-generalize check (it ships with pretrained SAEs).
- **Method:** TopK SAE on frozen features; edit/steer the activity via ViT-Prisma.
- **Two ways to create bias:** zero-shot (no training — cleanest answer key) + induced (a fake
  correlation we set at strength ρ).

## Target
MI4MedFM @ MICCAI 2026 workshop, **Track 1 (8–10 pp, archival)**. Deadline **15 Jul 2026**.

## Integrity guardrails
Pre-register before final runs; keep splits separate (no leakage); run the rivals' own methods
fairly; negative results are valid; keep synthetic→real claims modest; no made-up numbers or
citations.

## Work modules (can run in parallel once the injector is done)
1. Data loaders + artifact injector (`inject`/`remove`, repeatable) + seeded separate splits
2. Encoder/probe + bias measurement (zero-shot & induced arms)
3. SAE training + feature discovery (oracle vs. unsupervised)
4. Interventions + metrics harness (recovery / selectivity / off-target)
5. Baselines (CAV/CDEP/raw-neuron/DermFM-Zero/random/oracle)
6. Aggregation + figures + real-artifact slice

Full detail: `docs/EXECUTION_PLAN.md`, `docs/DATASETS.md`, `docs/METRICS.md`,
`docs/INTEGRITY.md`, `docs/NOVELTY.md`.
