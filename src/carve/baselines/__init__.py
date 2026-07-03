"""Baselines — evaluated on IDENTICAL ground truth, metrics, cells, eval split.

Each method plugs into carve.eval.harness.run_cell through the same interface as
carve.interventions, so all numbers stay comparable (INTEGRITY.md §5):

  * raw_neuron       — ablate the most artifact-correlated RAW block-ℓ neuron(s).
                       Answers "what does the SAE add over raw neurons?".  → act_fn
  * dermfmzero       — suppress the top-k SAE features most activated by the artifact
                       (DermFM-Zero, arXiv 2602.10624). The incumbent we validate. → op="ablate", S
  * random_ctrl      — random SAE feature / raw neuron; must do ≈nothing (sanity floor).
  * input_oracle     — remove the artifact at the INPUT = achievable ceiling. → oracle=True

CAV/Reveal2Revise and CDEP (heavier faithful reimplementations) are added in a second pass.
"""

from .dermfmzero_suppress import dermfmzero_select  # noqa: F401
from .input_oracle import ORACLE_META  # noqa: F401
from .random_ctrl import random_raw_neurons, random_sae_features  # noqa: F401
from .raw_neuron import (  # noqa: F401
    raw_image_scores,
    raw_neuron_ablate_fn,
    raw_neuron_select,
    reference_mean,
)
