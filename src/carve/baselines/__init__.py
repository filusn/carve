"""Baselines — evaluated on IDENTICAL ground truth, metrics, cells, eval split.

Implement (see EXECUTION_PLAN Phase 6, INTEGRITY.md §5). Each exposes an intervention with
the same interface as carve.interventions so the harness treats all methods uniformly.

raw_neuron.py
    raw_neuron_direction(encoder, layer, select_loader) -> neurons   # most artifact-correlated
    # ablate raw neuron(s); tests "what does the SAE add over raw neurons?"
cav.py
    fit_cav(encoder, layer, select_loader) -> direction             # Reveal2Revise / CRP-style
    # suppress along the artifact CAV; use zennit-crp faithfully (authors' method, fair budget)
cdep.py
    cdep_train(...) -> probe                                        # contextual-decomposition
    # penalty using injected masks (Rieger et al., ICML 2020)
random_ctrl.py
    random_feature(sae) -> S_rand                                   # must do ~nothing (sanity)
input_oracle.py
    oracle_effect(...) -> Tensor                                    # input removal = ceiling
"""
raise NotImplementedError("carve.baselines: implement raw_neuron/cav/cdep/random_ctrl/input_oracle")
