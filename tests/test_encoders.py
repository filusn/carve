"""Encoder + probe tests on real MONET + a few ISIC images. Heavy: skipped unless the
dataset is mounted (the model downloads/caches on first use). Runs in the carve-dev container."""
import os

import numpy as np
import pytest
import torch

_GT = "data/isic2018/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv"
pytestmark = pytest.mark.skipif(not os.path.exists(_GT), reason="ISIC-2018 data not mounted")


@pytest.fixture(scope="module")
def encoder():
    from carve.models.encoders import load_encoder

    return load_encoder(None)


@pytest.fixture(scope="module")
def sample_images():
    from carve.data.isic import load_isic_binary

    ds = load_isic_binary(image_size=224)
    return [ds.load_image(i) for i in range(6)]


def test_monet_shape_facts(encoder):
    assert encoder.n_layers == 24 and encoder.hidden == 1024


def test_activations_shapes(encoder, sample_images):
    cls = encoder.activations(sample_images, layer=12, pool="cls")
    assert cls.shape == (6, 1024)
    full = encoder.activations(sample_images[:2], layer=12, pool=None)
    assert full.shape == (2, 257, 1024)  # CLS + 16x16 patches


def test_zero_shot_margin_and_ablation_moves_it(encoder, sample_images):
    m = encoder.zero_shot_margin(sample_images)
    assert m.shape == (6,)

    def ablate(_m, _i, o):
        return (torch.zeros_like(o[0]),) + tuple(o[1:]) if isinstance(o, tuple) else o * 0

    with encoder.hooks([(12, ablate)]):
        m_abl = encoder.zero_shot_margin(sample_images)
    assert (m_abl - m).abs().max().item() > 1e-4  # writing at ℓ moves the decision


def test_probe_learns_and_f_decision_agrees(encoder, sample_images):
    # tiny separable toy: label images by whether a bright patch was pasted top-left
    from carve.models.probe import f_decision, train_probe

    items = []
    for k, base in enumerate(sample_images * 2):
        img = base.copy()
        lab = k % 2
        if lab:
            img[:40, :40, :] = 1.0
        items.append({"image": img, "label": lab})
    probe = train_probe(encoder, layer=12, probe_train_items=items)
    f = f_decision(probe, encoder, [it["image"] for it in items], layer=12)
    acc = ((f.numpy() > 0).astype(int) == np.array([it["label"] for it in items])).mean()
    assert acc >= 0.75  # a linear probe on frozen feats separates this easy signal
