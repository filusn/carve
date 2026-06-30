# CARVE — Datasets

**Public dermatology only.** We **do not redistribute images or third-party labels**;
`src/carve/data/download.py` fetches from official sources into a git-ignored `data/` dir,
and we release the **injection recipe + split indices**, not the data. Record exact
versions/hashes per run. Facts below verified 2026-06-30 (see `docs/LOG.md`).

## Design decision (refined 2026-06-30)

One labelled pool does double duty, avoiding cross-dataset shift between canvas and slice:

- **Core = the Bissoto-labelled pool (ISIC 2018 + Interactive Atlas).** Its per-image
  7-way artifact labels let us, from *one* source, (a) filter a **verified-clean canvas**
  (all artifact flags = 0) to paste synthetic artifacts onto, and (b) carve the
  **real-artifact slice** (a given flag = 1) for external validity.
- **Reservoir = HAM10000** (~10k, the standard melanoma-vs-nevus set) — the larger canvas
  if the clean pool is too small for training the induced-bias probe.
- **Final primary canvas is decided after the Day-1 clean-pool count** (task in Phase 0).
  Do not hardcode it before measuring.

## Sources

| Dataset | Use in CARVE | Source | License / access | Status |
|---|---|---|---|---|
| **Bissoto artifact annotations** | **core** — clean-canvas filter **+** real-artifact slice | repo `alceubissoto/debiasing-skin` (CVPR-W 2020; labels for "(De)Constructing Bias" CVPR-W 2019) | **NO license** (all-rights-reserved → may download & cite, **must not redistribute**) | **verified** |
| **ISIC 2018 (Task 1/2) images** | base images the Bissoto labels point at (2,594) | challenge.isic-archive.com / ISIC Archive | per-image ISIC terms (CC-BY-NC / CC0 mix) — check | confirm at download |
| **Interactive Atlas of Dermoscopy** | base images for the other 872 labelled | Atlas (Lio & Nghiem; via Bissoto refs) | research-use, registration | confirm |
| **HAM10000** | reservoir canvas (mel-vs-nevus) | Harvard Dataverse **DBW86T** / ISIC | **CC BY-NC 4.0** (attribution, non-commercial) | **verified** |
| **ISIC 2019 / 2020** | optional scale-up only (not MVP) | challenge.isic-archive.com | per-image terms — check | optional |
| **Derm7pt / Fitzpatrick17k / SkinCon** | optional concept / skin-tone / off-target probes | resp. official pages | research-use (some Fitzpatrick links rot) | optional |

### Bissoto annotations — exact specifics (verified)
- Repo: `https://github.com/alceubissoto/debiasing-skin` (labels live here, **not** the 2019
  `deconstructing-bias-skin-lesion` repo, which has only splits).
- Files: `artefacts-annotation/isic_bias.csv` (ISIC 2018, 2,594 rows) and
  `artefacts-annotation/atlas_bias.csv` (Atlas, 872 rows) → **3,466 annotated images**.
- **7 binary categories** (canonical CSV header):
  `dark_corner, hair, gel_border, gel_bubble, ruler, ink, patches`.
  (The 2019 paper prose says "color charts"; the released schema calls it `patches`.
  "Purple pen / surgical markings" map to **`ink`**.)
- **License: none** (GitHub reports `license: null`, no LICENSE file). Treat as
  all-rights-reserved: cite and use, do **not** re-host. Contact authors before any release.
- Only the labels are in the repo; images come from ISIC / Atlas separately.

## Models

| Role | Model | Why / status |
|---|---|---|
| **Primary FM** | **MONET** (Kim et al., *Nat. Med.* 2024; CLIP ViT-L/14, ~105k derm image–text pairs) | **public** weights (`github.com/suinleelab/MONET`, HF `suinleelab/monet`); documents artifact behaviour; zero-shot via text prompts → enables the no-training bias arm. **Verify exact LICENSE before quoting.** |
| **Generality check** | **CLIP ViT-B/16** | ViT-Prisma ships **pretrained SAEs** → saves SAE-training time; tests whether findings are MONET-specific. |
| ~~DermFM-Zero~~ | not usable | weights **private/on-request**, code CC-BY-NC-ND → cannot build on it. We instead **reimplement its top-k suppression method from the paper as a baseline** (see `NOVELTY.md`). |

## Task definition (MVP)
**Binary melanoma vs. nevus.** Clean causal signal; expand classes only if time allows.
Fixed in `configs/default.yaml`.

## Two bias-measurement arms (both reported)
1. **Natural, zero training (cleanest ground truth).** Frozen MONET answers mel-vs-nevus
   zero-shot via text prompts; paste/remove the artifact and measure the shift in its *own*
   output (Winkler-style: e.g. 0.16→0.54). No decision layer, so the artifact's effect on
   the frozen model **is** the pure causal quantity to recover.
2. **Induced, controlled (the ρ knob).** Train a light probe on a set where artifact
   presence correlates with the label at strength ρ ∈ {0.5,0.7,0.9,1.0} → guaranteed,
   dial-able bias. (Real natural precedent: colored patches co-occur with ~46% of benign
   but 0% of malignant ISIC images, Nauta et al. 2022 — the bias is real, we just control it.)

## Splits (seeded, disjoint — enforce in code)
`probe_train · sae_train · select (feature selection) · eval (causal eval) · test (final)`.
Persist indices under `data/splits/<dataset>_<seed>.json`; **assert pairwise-disjoint** in
`tests/test_splits.py`.

## Artifact assets
- **Synthetic:** procedurally drawn ruler ticks, marker/pen-ink blobs, dark-corner
  vignettes (text overlay if time) — parameterized by opacity α + placement, deterministic
  per seed. Categories chosen to align with the Bissoto `ruler / ink / hair` taxonomy (and
  DermFM-Zero's ruler / pen-ink / hair subsets) so real-slice comparison stays apples-to-apples.
- **Real overlays:** extract real rulers/hair/ink from Bissoto-flagged images to composite
  at graded α — counters the "synthetic = toy" objection.
- Always store the **artifact mask** alongside the image (CDEP and `remove()` need it).

## Phase-0 measurement (do before locking the canvas)
- [ ] Count the **all-flags-zero** clean pool in `isic_bias.csv` + `atlas_bias.csv` (and a
  relaxed variant: no `ruler/ink/patches/dark_corner`, hair allowed). If large enough for
  probe-train + eval → core canvas = Bissoto-clean ISIC/Atlas; else → canvas = HAM10000,
  Bissoto used only for the real slice. Log the counts + decision in `docs/LOG.md`.

## Provenance & licensing rules
- Log dataset version + file hashes into each run dir.
- Honor CC BY-NC (HAM10000) and all-rights-reserved (Bissoto labels): research use,
  attribution, **no redistribution of images or labels**; cite the source papers. If unsure
  about an image's terms, exclude it from any released artifact.
