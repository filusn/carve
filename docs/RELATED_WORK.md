# CARVE — Related work, novelty framing, and the objections to pre-empt

**Citation rule (from INTEGRITY.md):** verify every entry against the real paper before
citing. Status column: `V` = fetched/verified by our prior research pass; `?` = surfaced
from search snippets, **must open the paper before citing**. Re-verify all at write time.
arXiv IDs use `YYMM.NNNNN` (so `25xx`/`26xx` are 2025/2026 — not typos).

## The one-sentence novelty claim
> *The first **controlled causal validation** of SAE-feature interpretability in a medical
> foundation model: by **injecting artifacts with known causal effect**, we measure
> recovery, selectivity, and off-target damage of feature steering, benchmark it against
> CAV/Reveal2Revise, CDEP, and raw-neuron ablation, and release the harness.*
We are **not** claiming "first SAE in dermatology / medical imaging."

## A. Closest threat — SAEs + artifacts in dermatology
| Status | Work | ID | Why it's not us |
|---|---|---|---|
| V | **DermFM-Zero** (SAE concept discovery + artifact-bias *suppression*, derm) | 2602.10624 | Ablates artifact features on **natural** ISIC; **no known causal magnitude, no additive steering, no selectivity/off-target, no benchmark**. Our springboard — cite as the clinical motivation, position CARVE as the controlled validation it lacks. (Artifacts: ruler/ink/hair, **not** dark-corner.) |

## B. SAEs on medical encoders (we are not first here — cite as established)
| Status | Work | ID |
|---|---|---|
| V | Pathology FM SAE (PathAI) | 2407.10785 |
| ? | SAE-Rad (CXR, "X-ray is worth 15 features") | 2410.03334 |
| ? | CytoSAE (hematology, MICCAI 2025) | 2507.12464 |
| V | Mammo-SAE (Mammo-CLIP) | 2507.15227 |
| ? | MedSAE (MedCLIP) | 2510.26411 |
| V | MAIRA-2 SAE (radiology MLLM; medical SAE weights released) | 2507.12950 |
| ? | BiomedParse/DINOv3 SAE (CT/MRI) | 2603.23794 |

## C. Causal SAE-feature graphs / circuits (exists in LLM/VLM/genomics, not medical imaging)
| Status | Work | ID |
|---|---|---|
| V | Sparse Feature Circuits (nodes=SAE feats, causal edges; SHIFT editing) | 2403.19647 |
| ? | Circuit Tracing in VLMs (Findings CVPR 2026) | 2602.20330 |
| ? | Causal circuit tracing in single-cell FMs (genomics) | 2603.01752 |

## D. Synthetic / causal validation of SAEs (our methodological neighbors)
| Status | Work | ID | Gap we fill |
|---|---|---|---|
| V | RAVEL (causal "Cause" + "Isolation/off-target" metrics) | 2402.17700 | natural LLM attributes; we do **medical vision, injected artifacts** |
| V | SynthSAEBench (synthetic ground-truth dictionary) | 2602.14687 | **activation-level** recovery only, **no downstream steering** |
| V | Sanity Checks for SAEs (synthetic GT; SAEs recover ~9%) | 2602.14111 | motivates *why* causal validation is needed |
| V | TAPAScore / Concept-Annotations (synCUB/synCOCO, causal selectivity) | 2606.24716 | natural bird/COCO attrs, **no medical, no off-target steering** |
| ? | SAEBench (SCR concept-erasure metric) | 2503.09532 | general SAE eval; not medical/causal-artifact |

## E. Incumbent artifact mitigation = our baselines
| Status | Work | ID / venue | Role |
|---|---|---|---|
| V | Reveal2Revise / "Ensuring Medical AI Safety …" (Pahde et al.; CRP+CAV detect+unlearn) | 2501.13818 | **CAV baseline** |
| V | CDEP — "Interpretations are useful: penalizing explanations…" (Rieger et al., ICML 2020) | 1909.13584 | **CDEP baseline** |
| V | Steering CLIP's ViT with SAEs (spurious-correlation removal, Waterbirds) | 2504.08729 | steering recipe template |

## F. The shortcut is canonical (motivation, not novelty)
| Status | Work | ID / venue |
|---|---|---|
| V | Winkler et al. — surgical-marker bias in dermoscopy (spec. 84%→46%) | JAMA Dermatology 2019 |
| V | Bissoto et al. — "(De)Constructing Bias…" (artifact annotations on ISIC) | CVPR-W 2019 |
| ? | "Debiasing Skin Lesion Datasets… Not So Fast" | 2004.11457 |

## G. Tooling
| Status | Work | ID |
|---|---|---|
| V | ViT-Prisma (HookedViT, activation patching, pretrained CLIP/DINO SAEs) | 2504.19475 |

---

## The two killer objections — and the built-in rebuttals
1. **"A pasted artifact is trivially detectable; recovery is unsurprising and won't
   transfer."** → The hard regime is **ρ<1 partial correlation + graded opacity + real
   overlays**; the deliverable is a **method-agnostic benchmark + cross-method comparison**,
   not "SAEs win"; plus the **Phase-7 real-artifact slice**.
2. **"What does the lossy SAE add over a raw neuron / a CAV?"** → That is exactly the
   **raw-neuron and CAV/Reveal2Revise baselines** measured on identical ground truth. If
   the SAE doesn't add value, we report that — the benchmark is still the contribution.

## Searches to re-run the week before submission (scoop watch)
SAE + medical + (causal OR steering OR artifact OR shortcut); DermFM-Zero follow-ups;
"causal validation" + "sparse autoencoder"; RAVEL/Concept-Annotations for medical imaging.
