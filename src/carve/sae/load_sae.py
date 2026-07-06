"""Load a pretrained SAE for the generality model (CLIP ViT-B/16 via ViT-Prisma).

Used only for the generality/dev path (docs/EXECUTION_PLAN.md D2); the headline MONET SAE
is trained by carve.sae.train_sae. Kept importable; the vit_prisma import is lazy.
"""
from __future__ import annotations


def load_pretrained_sae(cfg=None, repo_id: str | None = None):
    """Load a ViT-Prisma pretrained CLIP SAE. Requires cfg.sae.pretrained.{repo_id,...} or
    an explicit repo_id. Not needed for the MONET headline result — wire when the generality
    (CLIP ViT-B/16) arm is added."""
    try:
        import vit_prisma  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise ImportError(f"vit_prisma is required to load a pretrained SAE: {e}") from e
    raise NotImplementedError(
        "Pretrained CLIP SAE loading is deferred to the generality arm; the MONET SAE is "
        "trained via carve.sae.train_sae. Provide cfg.sae.pretrained + wire the ViT-Prisma "
        "loader here when CLIP ViT-B/16 is added."
    )
