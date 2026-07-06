"""Loader tests for ISIC-2018 Task3. Skipped automatically when the dataset isn't mounted
(so CI without data stays green); runs for real inside the carve-dev container."""
import os

import numpy as np
import pytest

from carve.data.isic import load_isic_binary

_GT = "data/isic2018/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv"
pytestmark = pytest.mark.skipif(not os.path.exists(_GT), reason="ISIC-2018 data not mounted")


def test_load_binary_mel_vs_nevus():
    ds = load_isic_binary(pos_class="MEL", neg_class="NV", image_size=64)
    s = ds.summary()
    assert s["task"] == "MEL_vs_NV"
    assert s["n"] == s["n_pos"] + s["n_neg"]
    assert s["n_pos"] > 0 and s["n_neg"] > s["n_pos"]  # nevus is the majority class
    assert set(np.unique(ds.labels)).issubset({0, 1})
    assert len(ds.paths) == len(ds) == len(ds.labels)


def test_load_image_shape_and_range():
    ds = load_isic_binary(image_size=64)
    img = ds.load_image(0)
    assert img.shape == (64, 64, 3) and img.dtype == np.float32
    assert 0.0 <= img.min() and img.max() <= 1.0


def test_deterministic_order():
    a = load_isic_binary(image_size=32)
    b = load_isic_binary(image_size=32)
    assert a.ids == b.ids and np.array_equal(a.labels, b.labels)
