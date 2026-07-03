"""Unit tests for the TopK SAE core + discovery helpers. Synthetic tensors, no MONET/data.
Runs on whatever device carve.utils.device picks (CPU locally, CUDA in the container)."""
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from carve.sae.discovery import _auroc_columns, discovery_precision_at_k
from carve.sae.model import TopKSAE
from carve.sae.train_sae import sae_health, train_sae


def test_topk_encode_has_at_most_k_nonzeros():
    sae = TopKSAE(d_in=16, width=64, k=5)
    z = sae.encode(torch.randn(10, 16))
    assert z.shape == (10, 64)
    assert int((z > 0).sum(dim=1).max()) <= 5  # hard sparsity


def _synthetic_sparse(n=3000, d=32, true_feats=64, k_true=3, seed=0):
    rng = np.random.default_rng(seed)
    D = rng.normal(size=(true_feats, d)).astype(np.float32)
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    X = np.zeros((n, d), np.float32)
    for i in range(n):
        idx = rng.choice(true_feats, k_true, replace=False)
        X[i] = rng.uniform(0.5, 2.0, k_true).astype(np.float32) @ D[idx]
    return X + rng.normal(scale=0.02, size=(n, d)).astype(np.float32)


def test_train_reduces_reconstruction_error():
    X = _synthetic_sparse()
    cfg = OmegaConf.create({"device": "auto",
                            "sae": {"width": 128, "k": 4, "train": {"steps": 600, "batch": 512, "lr": 1e-3}}})
    untrained = TopKSAE(X.shape[1], 128, 4)
    h0 = sae_health(untrained.to("cpu"), X)
    sae = train_sae(X, cfg, seed=0, log=lambda *a: None)
    h1 = sae_health(sae, X)
    assert set(h1) >= {"fvu", "r2", "dead_feature_frac", "k", "width", "n"}
    assert h1["r2"] > h0["r2"] and h1["r2"] > 0.5     # learns real structure
    assert 0.0 <= h1["dead_feature_frac"] <= 1.0


def test_auxk_reduces_dead_features():
    # AuxK revives starved latents via the reconstruction residual, so it only helps when the
    # SAE actually under-reconstructs. Force that: true sparsity (8) > SAE k (4) ⇒ real residual,
    # and a wide dict (512) over d=32 ⇒ many latents starve under plain TopK.
    X = _synthetic_sparse(n=4000, d=32, true_feats=128, k_true=8)
    base = {"width": 512, "k": 4, "train": {"steps": 1200, "batch": 512, "lr": 1e-3}}
    plain = train_sae(X, OmegaConf.create({"device": "auto", "sae": base}), seed=0, log=lambda *a: None)
    auxk_cfg = {"width": 512, "k": 4,
                "train": {"steps": 1200, "batch": 512, "lr": 1e-3, "aux_k": 64, "dead_window": 50}}
    revived = train_sae(X, OmegaConf.create({"device": "auto", "sae": auxk_cfg}), seed=0, log=lambda *a: None)
    assert sae_health(revived, X)["dead_feature_frac"] < sae_health(plain, X)["dead_feature_frac"]
    assert sae_health(revived, X)["r2"] > 0.5     # reconstruction not sacrificed


def test_auroc_columns_matches_sklearn():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(1)
    scores = rng.normal(size=(200, 3))
    y = (rng.random(200) < 0.4).astype(int)
    got = _auroc_columns(scores, y)
    for f in range(3):
        assert got[f] == pytest.approx(roc_auc_score(y, scores[:, f]), abs=1e-6)


def test_auroc_columns_single_class_is_nan():
    out = _auroc_columns(np.random.rand(10, 2), np.ones(10, int))
    assert np.isnan(out).all()


def test_discovery_precision_at_k():
    disc = {"features": [7, 3, 99, 1, 42]}
    oracle = {"features": [3, 7]}
    assert discovery_precision_at_k(disc, oracle, k=2) == pytest.approx(1.0)  # top-2 both oracle
    assert discovery_precision_at_k(disc, oracle, k=5) == pytest.approx(2 / 5)
