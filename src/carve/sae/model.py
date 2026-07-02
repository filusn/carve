"""TopK sparse autoencoder over frozen ViT activations (docs/EXECUTION_PLAN.md Phase 3).

Trained in the RAW activation space of MONET's layer-ℓ residual stream (no input
standardization) so that carve.interventions can ablate/steer features directly on the
residual stream in Stage 5. Sparsity is enforced by a hard TopK on the latents (no L1),
following Gao et al. 2024; decoder feature directions are kept unit-norm.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TopKSAE(nn.Module):
    """x → z = TopK(ReLU(W_enc·(x − b_dec) + b_enc)) → x̂ = W_dec·z + b_dec.

    Shapes: W_enc [d, m], W_dec [m, d], b_enc [m], b_dec [d]; d = activation dim, m = width.
    """

    def __init__(self, d_in: int, width: int, k: int):
        super().__init__()
        self.d_in, self.width, self.k = int(d_in), int(width), int(k)
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.b_enc = nn.Parameter(torch.zeros(width))
        self.W_enc = nn.Parameter(torch.empty(d_in, width))
        self.W_dec = nn.Parameter(torch.empty(width, d_in))
        nn.init.kaiming_uniform_(self.W_enc)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t())
            self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))

    def preacts(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Latent activations after ReLU + hard TopK (rest zeroed). Shape [..., m]."""
        acts = torch.relu(self.preacts(x))
        topv, topi = acts.topk(self.k, dim=-1)
        return torch.zeros_like(acts).scatter_(-1, topi, topv)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z

    def decoder_direction(self, feature: int) -> torch.Tensor:
        """Unit decoder vector for a feature (the steering direction used in Stage 5)."""
        return self.W_dec[feature]
