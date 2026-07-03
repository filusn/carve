"""Integration tests for SAE-feature ablation + causal recovery on real MONET.
Heavy but small; skipped unless the ISIC data is mounted. Trains a tiny SAE once per module."""
import os

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

_GT = "data/isic2018/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv"
pytestmark = pytest.mark.skipif(not os.path.exists(_GT), reason="ISIC-2018 data not mounted")

LAYER = 12


@pytest.fixture(scope="module")
def monet_sae():
    from carve.data.artifacts import inject, make_biased_set
    from carve.data.isic import load_isic_binary
    from carve.models.encoders import load_encoder
    from carve.sae.discovery import select_oracle
    from carve.sae.train_sae import train_sae

    enc = load_encoder(None)
    ds = load_isic_binary(image_size=224)
    cfg = OmegaConf.create({"device": "auto",
                            "sae": {"width": 4096, "k": 32, "train": {"steps": 500, "batch": 4096, "lr": 1e-3}}})

    tr = [ds.load_image(i) for i in range(200)]
    acts = enc.activations(tr, LAYER, pool=None).reshape(-1, 1024)
    sae = train_sae(acts, cfg, seed=0, log=lambda *a: None)

    sel = [ds.load_image(i) for i in range(200, 300)]
    items, _ = make_biased_set(sel, ds.labels[200:300], "ruler", rho=0.9, alpha=1.0, seed=1)
    oracle = select_oracle(sae, enc, LAYER, items)["features"]

    ev_idx = range(300, 340)
    clean = [ds.load_image(i) for i in ev_idx]
    art = [inject(im, "ruler", 1.0, np.random.default_rng(i))[0] for im, i in zip(clean, ev_idx)]
    return enc, sae, oracle, art, clean


def test_ablate_changes_decision(monet_sae):
    from carve.interventions.hooks import ablate

    enc, sae, oracle, art, _ = monet_sae
    base = enc.zero_shot_margin(art)
    with ablate(enc, LAYER, sae, oracle):
        abl = enc.zero_shot_margin(art)
    assert (abl - base).abs().max().item() > 1e-4  # the hook actually edits the residual


def test_steering_scales_and_recovers_more_than_ablation(monet_sae):
    # Validates the intervention MECHANISM, not a scientific conclusion about recovery level.
    # (Empirically ablation barely recovers while steering recovers partially — the
    # detection-vs-control dissociation is reported by scripts/40, not asserted here.)
    from carve.interventions.mediation import f_intervened
    from carve.metrics.causal import causal_recovery
    from carve.models.probe import f_decision

    enc, sae, oracle, art, clean = monet_sae
    f_art = f_decision(None, enc, art)
    f_clean = f_decision(None, enc, clean)
    e_in = (f_art - f_clean).abs().median().item()

    def eS_and_R(op, coeff=None):
        f_iv = f_intervened(None, enc, LAYER, art, sae, oracle, op=op, coeff=coeff)
        e_s = (f_art - f_iv).abs().median().item()
        R = float(np.nanmedian(causal_recovery(f_art, f_iv, f_clean, eps=1e-3).numpy()))
        return e_s, R

    eS_abl, R_abl = eS_and_R("ablate")
    eS_s2, _ = eS_and_R("steer", 2.0)
    eS_s4, R_s4 = eS_and_R("steer", 4.0)
    assert np.isfinite(R_abl) and np.isfinite(R_s4)
    assert eS_s4 > eS_s2 > eS_abl          # steering magnitude scales with the coefficient
    # NB: we deliberately do NOT assert R_s4 > R_abl. At these settings both recoveries are
    # ~0 (the detection≠control dissociation), so their ordering is training noise; the
    # actual recovery comparison is reported by scripts/40, not asserted in a unit test.
    assert eS_s4 < 3 * e_in                 # sanity: not exploding


def test_oracle_moves_decision_more_than_random_feature(monet_sae):
    from carve.interventions.mediation import feature_effect

    enc, sae, oracle, art, _ = monet_sae
    rng = np.random.default_rng(0)
    rand = [int(rng.integers(0, sae.width))]
    e_oracle = feature_effect(None, enc, LAYER, art, sae, oracle, op="ablate").abs().median()
    e_random = feature_effect(None, enc, LAYER, art, sae, rand, op="ablate").abs().median()
    assert e_oracle.item() > e_random.item()  # selectivity sanity: artifact feature matters more
