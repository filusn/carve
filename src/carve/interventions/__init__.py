"""Interventions: activation-level ablation/steering + input-vs-feature mediation.

Implement (see EXECUTION_PLAN Phase 5, METRICS.md §3-4-7):

hooks.py
    ablate(encoder, layer, features) -> context manager     # zero feature(s) S during forward
    steer(encoder, layer, features, coeff) -> context manager
        # subtract coeff * decoder_direction(S) from the residual stream at `layer`
    # Implemented via the encoder's hook API; input image is UNCHANGED.

mediation.py
    input_effect(probe, encoder, layer, x_art, x_removed) -> Tensor
        # e_in = f(x_art) - f(remove(x_art))   (the GOLD causal effect)
    feature_effect(probe, encoder, layer, x_art, S, op, coeff=None) -> Tensor
        # e_S  = f(x_art) - f_{-S}(x_art)
    f_removed(probe, encoder, layer, x_art, x_removed) -> Tensor    # f(remove(x_art))
    # These three feed carve.metrics.causal_recovery (normalized mediation residual).
"""
raise NotImplementedError("carve.interventions: implement hooks.py / mediation.py")
