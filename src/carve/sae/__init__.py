"""SAE: train/load sparse autoencoders + discover candidate artifact features.

Implement (see EXECUTION_PLAN Phases 3-4, INTEGRITY.md §3-4):

train_sae.py
    train_sae(activations_loader, cfg, seed) -> SAE        # TopK SAE on layer-ℓ activations
    sae_health(sae, loader) -> dict                        # recon error, dead-feature frac,
                                                           # sparsity k, cross-seed stability
load_sae.py
    load_pretrained_sae(cfg) -> SAE                        # e.g. ViT-Prisma CLIP SAE

discovery.py
    select_oracle(sae, encoder, layer, select_loader) -> S_oracle
        # top detection-AUROC feature(s) for artifact-present vs absent on `select`
    discover_unsupervised(sae, encoder, layer, artifact_loader) -> S_discovered
        # NO injected labels: top activation/variance features on artifact images
    # CRITICAL: selection on `select`, evaluation later on `eval` (disjoint). Keep
    # S_oracle and S_discovered separate and labelled everywhere (no leakage, no conflation).
"""
raise NotImplementedError("carve.sae: implement train_sae.py / load_sae.py / discovery.py")
