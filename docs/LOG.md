# CARVE — decision & deviation log

Append-only record of substantive design decisions (and, post-freeze, any deviation from
PREREGISTRATION.md, with reason + date). Keep entries short.

## 2026-06-28 — project bootstrapped
- Repo scaffolded (CLAUDE.md, README, docs/{EXECUTION_PLAN,METRICS,INTEGRITY,RELATED_WORK,
  DATASETS}, PREREGISTRATION, package skeleton, configs, smoke test).
- Target: MI4MedFM @ MICCAI 2026, deadline 2026-07-15, Track 1 (8–10pp LNCS, archival).
- Scope locked to dermatology MVP; public/synthetic data only.

## Open decisions (resolve in Phase 0 — see EXECUTION_PLAN)
- **D1 base model:** `[TODO]` (medical FM primary + CLIP generality). Why: `[…]`
- **D2 SAE source:** `[TODO]` (train on medical FM vs Prisma pretrained for CLIP).
- **D3 track:** default Track 1; revisit ~Jul 10.
- **Layer ℓ rule, SAE width/k, recovery ε, success thresholds:** set in PREREGISTRATION.

## Entries
- `2026-06-30` — **Novelty gate #1 cleared (DermFM-Zero, arXiv 2602.10624): LOW overlap → PROCEED.**
  Two independent reads (paper + repo code) agree it's an artifact-suppression *demo*
  (top-5 neuron zeroing on natural ISIC subsets, one-way AUROC gain) with none of CARVE's
  controlled-validation machinery (D1–D5, D7 ABSENT; only D6 present). Decisions: (i) add
  DermFM-Zero-style top-k suppression as a baseline (reimplement from description — code is
  CC-BY-NC-ND, weights private); (ii) primary FM = MONET (DermFM-Zero weights unavailable);
  (iii) cite DermFM-Zero as motivating proof-of-concept, position CARVE as its missing
  rigorous validation. Full record: `docs/NOVELTY.md`.
- `2026-06-30` — Verified facts folded in: Winkler specificity **84.1%→45.8%** (confirmed);
  Bissoto artifact labels real & public — 3,466 imgs (2,594 ISIC2018 + 872 Atlas), 7 cats
  (`dark_corner,hair,gel_border,gel_bubble,ruler,ink,patches`), repo `alceubissoto/debiasing-skin`,
  **no license** (all-rights-reserved → cite/use, don't redistribute). Nauta 2022 natural
  bias: colored patches on ~46% of benign ISIC images, 0% malignant.
- `[YYYY-MM-DD]` — `[decision / deviation + reason]`
