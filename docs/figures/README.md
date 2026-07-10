# Paper figures

Final publication figures for the CARVE draft (`docs/PAPER_DRAFT.md`), copied here (committed) from
their git-ignored run dirs so the draft is self-contained. Regenerate any of them by re-running the
listed script/run.

| file | draft slot | what it shows | source |
|---|---|---|---|
| `artifacts.png` | Fig 1 | injected artifacts (ruler / arrow / soft black_corner) on real ISIC images, clean vs. injected | `inject()` on ISIC (current artifact set) |
| `detection_vs_recovery.png` | Fig 2 | "Detection ≠ control" scatter — every method near recovery R≈0 across detection AUROC | `…195504Z_baselines_grid` (§5.2, exploratory width 16384) |
| `rho_alpha_dissociation.png` | Fig 3 | ρ×α heatmaps — detection high, recovery ≈0 in every cell | `…213446Z_rho_alpha_sweep` (§6, confirmatory width 4096) |
| `mechanism.png` | Fig 5 | rank-1 causal effect vs. near-orthogonal detection direction | `…231727Z_effect_dimensionality` (§7) |
| `layer_robustness.png` | (new, for §8) | detection≠control + mechanism across blocks {6,8,10,12} | `…235403Z_layer_robustness` (§8) |

Notes:
- Fig 2 is at the exploratory width 16384 (the full baseline comparison); the confirmatory dissociation
  (Fig 3) is at the frozen-compliant width 4096. Both show the same result (see PROGRESS §6 width note).
- `layer_robustness.png` is a new result (PROGRESS §8) added after the initial draft; the draft still
  needs a short robustness subsection to reference it.
- Draft "Fig 4 (optional)" (recovery bars + selectivity-vs-off-target) is not copied here; regenerate
  from `…baselines_grid/figures/recovery_bars.png` and `selectivity_vs_offtarget.png` if used.
