"""Smoke tests that run without a GPU. Expand per phase (injection determinism,
split disjointness, metric unit tests on toy tensors) as code lands."""
import pytest


def test_package_imports():
    import carve

    assert isinstance(carve.__version__, str)


def test_assert_disjoint_catches_leakage():
    from carve.utils import assert_disjoint

    assert_disjoint(a={1, 2, 3}, b={4, 5})  # ok
    with pytest.raises(AssertionError):
        assert_disjoint(a={1, 2, 3}, b={3, 4})  # leakage


def test_device_returns_valid():
    pytest.importorskip("torch")
    from carve.utils import device

    assert device("auto") in {"cuda", "mps", "cpu"}


def test_injection_determinism():
    # implemented in Phase 1 — full coverage in tests/test_injection.py
    import numpy as np

    from carve.data.artifacts import inject

    img = np.random.default_rng(0).uniform(0.3, 0.8, (32, 32, 3)).astype("float32")
    a1, m1 = inject(img, "ruler", 0.7, np.random.default_rng(1))
    a2, m2 = inject(img, "ruler", 0.7, np.random.default_rng(1))
    assert np.array_equal(a1, a2) and np.array_equal(m1, m2)


def test_causal_recovery_toy():
    # implemented in Phase 5 — full coverage in tests/test_metrics.py
    from carve.metrics.causal import causal_recovery

    R = causal_recovery([0.55, 0.55], [0.20, 0.55], [0.20, 0.20], eps=1e-6)
    assert R[0].item() == pytest.approx(1.0)  # feat == removed → perfect recovery
    assert R[1].item() == pytest.approx(0.0)  # feat == art     → no recovery
