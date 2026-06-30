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


@pytest.mark.skip(reason="TODO Phase 1: same seed -> identical injected pixels & masks")
def test_injection_determinism():
    ...


@pytest.mark.skip(reason="TODO Phase 5: causal_recovery on toy tensors (R=1 perfect, 0 none)")
def test_causal_recovery_toy():
    ...
