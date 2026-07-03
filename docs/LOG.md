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
- `2026-07-02` — **Dev environment = Docker (not venv), per PI.** Base image
  `research-base:2026-07` (`docker/base.Dockerfile`, `nvidia/cuda:12.4.1` + torch). GPU
  passthrough verified inside container (`torch.cuda.is_available()==True`, RTX 4080 Laptop
  12 GB, torch 2.12.1+cu130). Persistent dev container `carve-dev` (`--gpus all`, repo at
  `/workspace`, `PYTHONPATH=/workspace/src`, `HF_HOME=/workspace/weights/hf`) with
  `requirements.txt` installed once; iterate via `docker exec carve-dev …`. All deps import
  clean (open_clip, vit_prisma, crp, sklearn, pyarrow…). `docker-run.sh` kept for one-offs.
  Container python = 3.10 (satisfies ≥3.10). Data tests fail on import as expected (no
  `carve.data` yet) → TDD target set. Build order: `docs/BUILD_ORDER.md`.
- `2026-07-02` — **Stage-1 MONET hook probe = GO** (`scripts/02_monet_hook_probe.py`,
  run `experiments/runs/20260702T212841Z_monet_hook_probe`). Load path confirmed: HF
  `chanwkim/monet` via `transformers.AutoModelForZeroShotImageClassification` (CLIP
  **ViT-L/14**). Module tree: `vision_model.encoder.layers[ℓ]`, **24 vision blocks**,
  **hidden dim 1024**; residual stream at ℓ = `[B, 257, 1024]` (256 patches + CLS) →
  **SAE input dim = 1024**. Plain HF **forward hooks** both READ activations and WRITE
  them: ablating layer 12 shifts the zero-shot mel−nevus logit margin by 1.56, a random
  steer by 1.83 → the decision is controllable at the residual stream. ⇒ `carve.interventions`
  builds on HF hooks; **no dependency on ViT-Prisma accepting MONET weights**. Config
  `model.layer_sweep [4,6,8,10]` valid for depth 24. Weights cached at `weights/hf`
  (git-ignored) — no separate download needed.
- `2026-07-03` — **Canvas RESOLVED = ISIC-2018 Task3 (HAM10000), mel-vs-nevus.** Data
  provisioned locally under `data/` via symlinks to `/home/barmon/ceai/datasets/ISIC/ISIC_2018`
  (`isic2018` → ISIC_2018 root; `ham10000` → Task3 training images). Counts: **7,818**
  mel-vs-nevus images (**MEL 1,113 pos / NV 6,705 neg**, ~6:1), 600×450 RGB, labels from
  `ISIC2018_Task3_Training_GroundTruth.csv` (one-hot MEL,NV,BCC,AKIEC,BKL,DF,VASC).
  **Deviation from plan:** the **Bissoto artifact annotations are NOT available** here, so
  (a) no "verified-clean" canvas filter → we inject onto HAM10000 as-is (the Phase-0
  clean-pool count is therefore moot; the inject/remove gold counterfactual still isolates
  each injected artifact's effect regardless of pre-existing real artifacts), and (b) no
  Bissoto real-artifact slice for Phase-7 external validity. NOTE for PI: the dataset tree
  also contains pre-made artifact variants (`..._Test_Input_rulers/_arrows/_both`) and SAM
  lesion masks from a prior project — candidate substitutes for the real-slice / masks;
  flag before relying on them. Loader: `carve.data.isic.load_isic_binary`; config
  `dataset.name=isic2018_task3`. Container must bind-mount `/home/barmon/ceai/datasets` (ro)
  so the absolute symlinks resolve (done for `carve-dev`).
- `2026-07-03` — **Phase-0 GO/NO-GO (bias half) = GO** (`scripts/01_smoke_test.py`, MONET ×
  ISIC-2018 Task3, seeded disjoint splits, α=1.0, ρ=1.0). Bias confirmed for **all three
  artifacts in both arms** (full run: probe_train=500, eval=200, layer ℓ=12,
  run `experiments/runs/20260702T221857Z_phase0_gate`):
  - **Zero-shot** input effect e_in (median, CI95 excludes 0 for all):
    ruler **−0.51** [−0.66,−0.34] (→ nevus, 25% toward mel), marker_ink **+0.49**
    [+0.39,+0.64] (→ melanoma, 88%), dark_corner **+0.45** [+0.30,+0.57] (→ mel, 72%).
    **Finding: artifact bias is directional and artifact-specific** — ink/dark-corner push
    toward *malignant* (Winkler-style), ruler pushes toward *benign* (opposite). Report both.
  - **Induced** bias_gap ≈ **+1.00** (ruler 1.00, dark_corner 0.995, marker_ink 0.945;
    clean-probe acc 0.92). Saturated at the easiest setting → for the real grid use ρ∈{0.5,0.7,0.9} and
    graded α to de-saturate (same saturation seen in layer-AUROC). 2nd gate condition (SAE
    detection+recovery) is Stage 4 — proceed to build it.
  - `2026-07-03b` **marker_ink resized** to a small ink dot (radii ×0.15–0.35; ≈3–17% of
    image diameter). Effect ~halved (zero-shot e_in +0.96→**+0.49**; bias_gap 1.00→**0.945**)
    — smaller footprint ⇒ smaller causal effect, as expected. Still GO. Refreshed run
    `experiments/runs/20260702T223404Z_phase0_gate` (prior larger-marker run `…221857Z`).
- `2026-07-03` — **Stage 4 SAE built + validated (quick run) — SAE gate half: DETECTION ✓.**
  TopK SAE (`carve.sae`, raw layer-12 activation space so Stage-5 interventions act directly
  on the residual stream). Quick run (400 sae_train imgs → 102,800 patch+CLS tokens, width
  4096, k=32, 800 steps; run `…231421Z_train_sae`): **R²=0.990**, FVU 0.010, dead 9.6%.
  Discovery on a ρ=0.9 ruler-biased `select` set: **oracle single feature #993 detects the
  ruler at AUROC 0.997**. Unsupervised (activation-variance) top-5 misses it →
  **precision@5 = 0.00** — naive unsupervised discovery does NOT surface the artifact feature
  (reportable detection-vs-discovery dissociation; a contrast-based unsupervised method may
  do better — try in the full run). Config `sae.width/k` set (16384/32) — freeze in
  PREREGISTRATION. Remaining gate condition (CAUSAL RECOVERY: does ablating #993 reproduce
  e_in?) = Stage 5. Still on track: detection strong, recovery TBD.
- `2026-07-03` — **Stage 5 interventions + harness built → core CARVE result (quick):
  DETECTION ≠ CONTROL.** `carve.interventions` (ablate = subtract z_f·W_dec[f]; steer =
  subtract c·decoder dir) + `carve.eval.harness.run_cell` (writes per-cell json + per-image
  parquet + RunRecord). Quick cell (MONET, ruler, ρ=0.9, α=1.0, layer 12, width-4096 SAE,
  eval n=150; run `…233021Z_interventions`):
  - oracle feature detects ruler at **AUROC 0.997** but **ablation causal recovery R=0.150
    [0.13,0.18]** — far below the 0.5 "substantial mediation" bar. Ablation is highly
    **selective** (0.997) and zero off-target, but recovers little.
  - **Steering** raises the effect magnitude (cause↑ with c) but is **not selective**
    (~0.44–0.50, hits clean images too), recovery stays ≈0 and **overshoots** at c=8
    (R=−0.12); off-target nonzero. Coefficient-sensitive, unstable.
  - **Random-feature control** R=0.000 (sanity ✓).
  ⇒ The headline dissociation CARVE was built to quantify: **an SAE feature can detect an
  injected artifact almost perfectly yet fail to causally control it.** Quick numbers vary
  with eval sample (a diagnostic on a different eval slice gave ablation R≈0.04) → the
  FINAL claim needs the full grid (width 16384, ≥3 seeds, ρ/α sweep, all 3 artifacts) +
  Stage-6 baselines (does CAV/CDEP/raw-neuron do better?). Steer coeff sign: recovery
  direction is subtract POSITIVE c (config `interventions.steer_coeffs` are negative →
  revisit in PREREGISTRATION).
- `2026-07-03` — **Provenance fix:** git runs as root over uid-1000 `/workspace` in the
  container → "dubious ownership" → run dirs recorded `git_commit=nogit`. Fixed with
  `git config --global --add safe.directory /workspace` in `carve-dev`; **bake into the base
  image / docker-run** so every run captures the commit (INTEGRITY.md).
- `2026-07-03` — **Full 3-seed × 3-artifact grid (width 16384) — headline firmed up:
  DETECTION ≈ 1.0, ABLATION RECOVERY ≈ 0.** Run `…234519Z_interventions_grid` (seeds 0/1/2,
  ρ=0.9, α=1.0, layer 12, eval n=250). Ablate(oracle) causal recovery R (mean±std over seeds):
  **ruler −0.03±0.13, marker_ink +0.03±0.10, dark_corner +0.01±0.01** — all indistinguishable
  from 0 despite **detection AUROC 0.994–1.000**. Ablation **selectivity 0.98–0.998** (a pure,
  causally-inert detector). **Steering** noisy/overshoots (R negative at c≥4, off-target rises
  to 0.2–0.56 at c=4–8; only ruler c=2 gives R≈+0.24 with off-target 0.08 and poor selectivity
  ~0.54). **Random control R=0.** ⇒ Robust, multi-seed **detection≠control**: an SAE feature
  detects each injected artifact near-perfectly yet ablating it recovers ~none of the causal
  effect; steering only partially and unstably. **CAVEAT (SAE health):** at width 16384/k32 the
  **dead-feature fraction is high (~55%)** (vs ~5–10% at width 4096) — under-trained/over-wide;
  before the FINAL numbers, add dead-feature resampling or reduce width, and report cross-seed
  decoder stability. Next: Stage-6 baselines (CAV/CDEP/raw-neuron/input-oracle) on the same
  cells — the comparative claim.
- `2026-07-03` — **Stage-6 core-4 baselines on identical ground truth → the dissociation
  survives the comparison; the SAE adds no causal-control value over raw neurons.** New
  keystone: `run_cell` generalized to one interface (activation-editor `act_fn` | SAE `op/S`
  back-compat | input-removal `oracle=True`) so all methods share cells/metrics/eval split.
  Full 3-seed × 3-artifact grid (width 16384, ρ=0.9, α=1.0, layer 12, eval n=250; run
  `…134217Z_baselines_grid`). Causal recovery R (mean±std over seeds) & selectivity / off-target:
  - **input-removal oracle (ceiling):** R=**1.00**, sel 1.00, off-target 0 — the effect IS
    fully removable; every method below is read against this.
  - **SAE oracle-ablate (ours):** detection **0.994–1.000**, R = ruler −0.03±0.13 / marker
    +0.03±0.10 / dark +0.01±0.01 (**≈0**), **selectivity 0.98–0.998**, off-target 0 — a pure,
    causally-inert detector.
  - **Raw-neuron ablate (budget-matched):** detection **0.973–0.995** (raw neurons detect the
    artifact just as well), R ≈0 (+0.03/+0.00/+0.01), selectivity 0.44–0.67. ⇒ **answers the
    killer objection: the SAE dictionary buys near-perfect selectivity but no more causal
    control than a single raw neuron — neither controls the artifact.**
  - **DermFM-Zero top-5 suppression (incumbent):** the only mover, but **non-selective**
    (sel 0.42–0.47), **12.5% off-target** clean-task damage, and **wrong-direction** on 2/3
    artifacts (R = ruler +0.07±0.16 / marker −0.76±0.56 / dark −0.48±0.38) — nowhere near the
    ceiling. Its published one-way AUROC gain hides this.
  - **Random raw control:** R≈0 (sanity ✓).
  ⇒ Robust benchmark claim: **on injected artifacts with a known causal effect, detection
  (SAE & raw neuron ≈1.0 AUROC) does not imply control (R≈0); the incumbent suppression moves
  the decision but non-selectively and often the wrong way.** Figures in `figures/phase6/`
  (detection-vs-recovery, recovery bars, selectivity-vs-off-target, detection bars), all
  regenerable via `scripts/60_figures.py`. Robustness: the SAME dissociation holds on a
  **healthy width-4096 SAE (dead 9.6%)** in the `--quick` run → not an artifact of the 55%-dead
  wide dictionary. Deferred (task #12): CAV/Reveal2Revise + CDEP second pass. Before FINAL
  numbers (task #13): SAE-health fix + freeze PREREGISTRATION.
- `2026-07-03` — **Ops:** `carve-dev` lost GPU after ~16h (NVML "Unknown Error", cuda
  unavailable) — host driver/cgroup desync, not a host GPU fault (RTX 4080 healthy). Fixed by
  `docker restart carve-dev` (non-destructive: deps in image, code/data bind-mounted). Restart
  also wiped the in-container git identity + `safe.directory` → **bake both into the image**
  (already flagged) so runs keep capturing `git_commit`.
- `[YYYY-MM-DD]` — `[decision / deviation + reason]`
