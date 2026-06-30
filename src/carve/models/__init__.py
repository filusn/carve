"""Models: frozen, hookable encoders + linear/MLP probes.

Implement (see EXECUTION_PLAN Phase 2):

encoders.py
    load_encoder(cfg) -> HookedEncoder
        # wrap the chosen medical FM (+ CLIP generality model) so activations at a given
        # layer are readable AND writable via hooks (ViT-Prisma HookedViT or forward hooks).
        # encoder is FROZEN. Expose: .activations(images, layer) and a hook-registration API
        # used by carve.interventions.
    pick_layer(encoder, select_loader, cfg) -> int
        # pre-committed rule (cfg.model.layer_rule), e.g. max artifact linear-decodability
        # on the `select` split; ALSO return/log the full layer sweep.

probe.py
    train_probe(encoder, layer, probe_train_loader, cfg) -> Probe   # linear or light MLP
    f_decision(probe, encoder, images, layer) -> Tensor             # = cfg.decision_signal
        # the scalar decision signal f(.) used by ALL metrics (logit margin z_pos - z_neg)
"""
raise NotImplementedError("carve.models: implement encoders.py / probe.py")
