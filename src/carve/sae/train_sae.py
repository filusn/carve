"""Train a TopK SAE on cached MONET layer-ℓ activations + report SAE health (Phase 3).

Contract (carve.sae docstring): train_sae(activations, cfg, seed) -> SAE ;
sae_health(sae, activations) -> dict. Activations are raw residual-stream vectors
(patch + CLS tokens flattened to [N, d]); see scripts/30_train_sae.py for extraction.
"""
from __future__ import annotations

import numpy as np
import torch

from ..utils import device as _device
from .model import TopKSAE


def _as_2d(activations) -> torch.Tensor:
    a = torch.as_tensor(np.asarray(activations), dtype=torch.float32)
    return a.reshape(-1, a.shape[-1]) if a.ndim > 2 else a


def train_sae(activations, cfg=None, seed: int = 0, log=print) -> TopKSAE:
    """Fit a TopK SAE to raw activations [N, d]. Config keys under cfg.sae:
    width, k, and train.{steps,batch,lr}. Decoder kept unit-norm; b_dec init = data mean."""
    torch.manual_seed(seed)
    dev = _device(cfg.get("device", "auto") if cfg is not None else "auto")

    X = _as_2d(activations)
    N, d = X.shape
    scfg = cfg.sae if cfg is not None else {}
    width = int(scfg.get("width", 16384))
    k = int(scfg.get("k", 32))
    tcfg = scfg.get("train", {}) if hasattr(scfg, "get") else {}
    steps = int(tcfg.get("steps", 2000))
    batch = int(min(tcfg.get("batch", 4096), N))
    lr = float(tcfg.get("lr", 1e-3))
    # AuxK dead-feature revival (Gao et al. 2024): reconstruct the main residual with the
    # top-`aux_k` DEAD latents so starved features keep getting a gradient. Default off
    # (aux_k=0) → identical to the plain TopK loss, so prior runs/tests are unchanged.
    aux_k = int(tcfg.get("aux_k", 0))
    aux_coef = float(tcfg.get("aux_coef", 1.0 / 32))
    dead_window = int(tcfg.get("dead_window", 200))  # steps unfired ⇒ "dead" (auxk-eligible)

    sae = TopKSAE(d, width, k).to(dev)
    with torch.no_grad():
        sae.b_dec.copy_(X.mean(0).to(dev))
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    Xdev = X.to(dev) if X.numel() * 4 < 3_000_000_000 else None  # keep on GPU if < ~3GB
    rng = np.random.default_rng(seed)
    since_fired = torch.zeros(width, device=dev)  # steps since each feature last fired
    log(f"[sae] train: N={N} d={d} width={width} k={k} steps={steps} batch={batch} lr={lr} "
        f"aux_k={aux_k} dev={dev}")
    for step in range(steps):
        idx = rng.integers(0, N, size=batch)
        x = (Xdev[idx] if Xdev is not None else X[idx].to(dev))
        z = sae.encode(x)
        recon = sae.decode(z)
        loss = (recon - x).pow(2).mean()

        fired = (z > 0).any(dim=0)                                   # [m]
        since_fired = torch.where(fired, torch.zeros_like(since_fired), since_fired + 1)
        if aux_k > 0:
            dead = since_fired > dead_window
            if int(dead.sum()) > 0:                                   # revive dead latents
                pre = torch.relu(sae.preacts(x)).masked_fill(~dead.unsqueeze(0), 0.0)
                kk = min(aux_k, int(dead.sum()))
                tv, ti = pre.topk(kk, dim=-1)
                z_aux = torch.zeros_like(pre).scatter_(-1, ti, tv)
                err = (x - recon).detach()
                loss = loss + aux_coef * (z_aux @ sae.W_dec - err).pow(2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sae.normalize_decoder()
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            log(f"   step {step:5d}  mse {loss.item():.5f}")
    sae.eval()
    return sae


@torch.no_grad()
def sae_health(sae: TopKSAE, activations, batch: int = 8192) -> dict:
    """Reconstruction quality + sparsity health over a held-out activation set.

    Returns fraction-of-variance-unexplained (FVU) and R², dead-feature fraction, k, width.
    """
    dev = next(sae.parameters()).device
    X = _as_2d(activations)
    N, d = X.shape
    total_var = X.var(0, unbiased=False).sum().item()
    sse, fired = 0.0, torch.zeros(sae.width, device=dev)
    for i in range(0, N, batch):
        x = X[i : i + batch].to(dev)
        recon, z = sae(x)
        sse += (recon - x).pow(2).sum().item()
        fired += (z > 0).sum(0)
    fvu = sse / (total_var * N + 1e-12)
    return {
        "fvu": float(fvu),
        "r2": float(1.0 - fvu),
        "dead_feature_frac": float((fired == 0).float().mean().item()),
        "n_active_features": int((fired > 0).sum().item()),
        "k": sae.k,
        "width": sae.width,
        "n": int(N),
    }


@torch.no_grad()
def decoder_cosine_stability(sae_a: TopKSAE, sae_b: TopKSAE) -> dict:
    """Cross-seed feature stability: mean best-match cosine between two SAEs' decoder dicts
    (each feature in A matched to its most similar feature in B). ~1 ⇒ stable dictionary."""
    A = torch.nn.functional.normalize(sae_a.W_dec, dim=1)
    B = torch.nn.functional.normalize(sae_b.W_dec.to(A.device), dim=1)
    sim = A @ B.t()  # [m_a, m_b]
    best = sim.max(dim=1).values
    return {"mean_best_cosine": float(best.mean()), "median_best_cosine": float(best.median())}
