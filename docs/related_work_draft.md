# CARVE — Related Work & Positioning (draft)

Drop-in LNCS text with `\cite{}` keys. **Status legend at the bottom marks which
references are VERIFIED vs MUST-VERIFY before submission** (INTEGRITY.md §8). A 2026-06-30
verification pass cleared winkler2019 (figures), bissoto2019/2020, and dermfmzero2026; rows
still marked `?` remain to be opened before final.

---

## Related Work

**Shortcut learning and artifact bias in dermatology.**
Deep dermatology classifiers are well known to exploit spurious image artifacts rather than
lesion morphology. Winkler et al.~\cite{winkler2019} reported that adding surgical (gentian-violet) skin
markings collapsed a melanoma classifier's specificity from 84.1\% to 45.8\%, and
Bissoto et al.~\cite{bissoto2019} showed that models can attain surprisingly high accuracy
using only artifact-bearing image regions, with follow-up analyses cautioning that such
biases are not easily removed~\cite{bissoto2020}; indeed, in ISIC, coloured patches
co-occur with ${\sim}46\%$ of benign but no malignant images, a textbook spurious
correlation~\cite{nauta2022}. A line of mitigation methods therefore
*detects and suppresses* the responsible representations: concept- and relevance-based
approaches such as Reveal2Revise~\cite{pahde2025} localise a spurious concept direction and
unlearn it, while training-time explanation penalties such as CDEP~\cite{rieger2020}
discourage reliance on annotated artifact regions. These methods are our natural baselines.
Critically, they are validated on *natural* data, where the artifact's true causal
contribution to a given prediction is unknown and can only be approximated.

**Sparse autoencoders for interpreting (medical) vision models.**
Sparse autoencoders (SAEs) have become a standard tool for decomposing foundation-model
activations into more monosemantic features, and the medical-imaging community has adopted
them across pathology~\cite{pathai2024}, chest X-ray~\cite{saerad2024},
mammography~\cite{mammosae2025}, hematology~\cite{cytosae2025}, and radiology report
models~\cite{maira2sae2025}, among others~\cite{medsae2025,biomedparsesae2026}. Most
directly related to us, DermFM-Zero~\cite{dermfmzero2026} trains SAEs on a dermatology model
to discover clinical concepts and *suppress* artifact-activated latents, reporting 12--38\%
diagnostic-AUROC gains on naturally artifact-contaminated subsets; this shows suppression
*helps*, not that the suppressed latent is the artifact's causal carrier, how selectively it
acts, or what it damages. SAEs in dermatology are thus **not** new. What remains absent across this literature is *causal validation*:
these works establish that a feature *correlates* with an artifact or concept—qualitatively,
or via ablation on natural images—but not that the feature is *causally* responsible for the
model's artifact reliance to a measurable degree, nor that steering it removes that reliance
*selectively* and without collateral damage.

**Causally validating SAE features.**
Establishing causal rather than correlational claims about SAE features is an active
methodological frontier. Sparse-feature-circuit and circuit-tracing methods compose features
into causal graphs in language, vision-language, and genomics
models~\cite{marks2024,vlmcircuits2026,singlecell2026}, and disentanglement benchmarks such
as RAVEL~\cite{ravel2024} score whether an intervention affects a target attribute (*cause*)
without affecting others (*isolation*). Because natural data lacks ground truth, several
works construct *synthetic* ground truth: SynthSAEBench~\cite{synthsae2026} and
sanity-check studies~\cite{sanitysae2026} insert known features and find that SAEs recover
only a fraction of them, while TAPAScore~\cite{tapascore2026} builds paired images differing
in a single natural attribute to measure causal selectivity. These efforts, however, operate
either at the level of *activation reconstruction* rather than *downstream task
behaviour*~\cite{synthsae2026,sanitysae2026}, or on *natural* object attributes in
non-medical settings without an off-target/steering analysis~\cite{tapascore2026,ravel2024}.
To our knowledge, no prior work provides a controlled causal validation of SAE-feature
steering for a *medical* foundation model that ties a *known, injected* artifact to its
measured effect on clinical predictions.

## Positioning and Contributions

CARVE fills exactly this gap. Rather than asking whether an SAE feature *correlates* with an
artifact, we inject the artifact ourselves at a controlled spurious-correlation strength and
opacity, which makes its true causal effect on the model directly measurable by intervening
on the input. We then test whether intervening on the candidate SAE feature *reproduces*
that input-level effect (recovery), *selectively* (changing artifact-contaminated but not
clean cases), and *without off-target damage*—and we evaluate CAV/Reveal2Revise~\cite{pahde2025},
CDEP~\cite{rieger2020}, DermFM-Zero-style top-$k$ feature suppression~\cite{dermfmzero2026},
raw-neuron ablation, and a random-feature control on identical ground truth. Two clarifications pre-empt the obvious concerns. First, our contribution is a
*benchmark and evaluation*, not a claim that SAEs are superior: whether SAE steering beats a
CAV—or even a single raw neuron—is an empirical question we report honestly, and a negative
result is informative. Second, synthetic ground truth is a deliberate methodological choice,
not a convenience—it is the only setting in which the causal quantity being estimated is
known, which is precisely why it underpins recent SAE-evaluation
work~\cite{synthsae2026,sanitysae2026,tapascore2026}; we further guard against the "toy"
regime by sweeping *partial* correlation ($\rho<1$) and graded opacity, compositing *real*
extracted rulers and hair, and including a real-artifact evaluation slice built from existing
ISIC artifact annotations~\cite{bissoto2019}. Our contributions are: (i) a controlled
artifact-injection protocol that yields per-image causal ground truth for medical-image
models; (ii) three causal metrics—recovery, selectivity, and off-target damage—adapting the
cause/isolation principle~\cite{ravel2024} to downstream clinical predictions; (iii) a
head-to-head benchmark of SAE steering against concept-level (CAV/Reveal2Revise),
training-level (CDEP), suppression-based (DermFM-Zero), and raw-neuron baselines; and
(iv) a released, reusable harness.

---

## Citation legend & verification status

`V` = verified in our research pass · `?` = MUST open the paper and confirm before citing.

| key | reference | id / venue | status |
|---|---|---|---|
| winkler2019 | Winkler et al., skin-marking bias in melanoma classification | JAMA Dermatology 2019;155(10):1135 | **V** — confirmed: specificity 84.1%→45.8% |
| bissoto2019 | Bissoto et al., "(De)Constructing Bias…" (ISIC artifact annotations) | CVPR-W 2019, arXiv 1904.08818 | **V** |
| bissoto2020 | "Debiasing Skin-Lesion Datasets and Models? Not So Fast" | CVPR-W 2020, arXiv 2004.11457 | **V** |
| nauta2022 | Nauta et al., "Uncovering & Correcting Shortcut Learning…" (patches ~46% benign / 0% malignant) | Diagnostics 2022, PMC8774502 | **V** |
| pahde2025 | Reveal2Revise / "Ensuring Medical AI Safety…" (CRP+CAV) | arXiv 2501.13818 | V |
| rieger2020 | CDEP, "Interpretations are useful: penalizing explanations…" | ICML 2020, arXiv 1909.13584 | V |
| pathai2024 | SAEs on a pathology foundation model | arXiv 2407.10785 | V |
| saerad2024 | SAE-Rad (chest X-ray) | arXiv 2410.03334 | ? |
| mammosae2025 | Mammo-SAE | arXiv 2507.15227 | V |
| cytosae2025 | CytoSAE (hematology) | MICCAI 2025, arXiv 2507.12464 | ? |
| maira2sae2025 | MAIRA-2 SAE (radiology MLLM) | arXiv 2507.12950 | V |
| medsae2025 | MedSAE (MedCLIP) | arXiv 2510.26411 | ? |
| biomedparsesae2026 | SAEs on BiomedParse/DINOv3 (CT/MRI) | arXiv 2603.23794 | ? |
| dermfmzero2026 | DermFM-Zero (SAE concepts + artifact suppression, derm) | arXiv 2602.10624 | **V** — 2-pass read 2026-06-30; cite as non-peer-reviewed preprint, weights private |
| marks2024 | Sparse Feature Circuits | arXiv 2403.19647 | V |
| vlmcircuits2026 | Circuit Tracing in Vision-Language Models | CVPR-W/Findings 2026, arXiv 2602.20330 | ? |
| singlecell2026 | Causal circuit tracing in single-cell FMs | arXiv 2603.01752 | ? |
| ravel2024 | RAVEL (cause/isolation disentanglement) | arXiv 2402.17700 | V |
| synthsae2026 | SynthSAEBench (synthetic ground-truth features) | arXiv 2602.14687 | V |
| sanitysae2026 | Sanity Checks for SAEs (synthetic GT; low recovery) | arXiv 2602.14111 | V |
| tapascore2026 | TAPAScore / Concept-Annotations (synCUB/synCOCO selectivity) | ECCV 2026, arXiv 2606.24716 | V |

**Before submission:** (1) open every remaining `?` row and confirm title/authors/venue/ID;
(2) re-run the scoop-watch searches in `docs/RELATED_WORK.md` and the gate log in
`docs/NOVELTY.md`; (3) build `refs.bib` from this legend. Do not cite any row still marked `?`.
