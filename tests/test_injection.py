"""Phase-1 unit tests for the artifact injector. CPU-only, no model, no data."""
import numpy as np
import pytest

from carve.data import artifacts as A
from carve.data.artifacts import (
    ALL_ARTIFACT_KINDS,
    ARTIFACT_KINDS,
    LEGACY_ARTIFACT_KINDS,
    inject,
    remove,
)


def _img(h=64, w=64, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.uniform(0.3, 0.8, (h, w, 3))).astype(np.float32)


@pytest.mark.parametrize("kind", ALL_ARTIFACT_KINDS)
def test_inject_determinism(kind):
    img = _img()
    a1, m1 = inject(img, kind, 0.7, np.random.default_rng(123))
    a2, m2 = inject(img, kind, 0.7, np.random.default_rng(123))
    assert np.array_equal(a1, a2)
    assert np.array_equal(m1, m2)


@pytest.mark.parametrize("kind", ALL_ARTIFACT_KINDS)
def test_inject_shapes_and_range(kind):
    img = _img(48, 80)
    art, mask = inject(img, kind, 0.7, np.random.default_rng(1))
    assert art.shape == img.shape and art.dtype == np.float32
    assert mask.shape == img.shape[:2] and mask.dtype == np.float32
    assert art.min() >= 0.0 and art.max() <= 1.0
    assert mask.min() >= 0.0 and mask.max() <= 1.0
    assert mask.sum() > 0  # the artifact actually covers something


@pytest.mark.parametrize("kind", ALL_ARTIFACT_KINDS)
def test_alpha_zero_is_noop(kind):
    img = _img()
    art, _ = inject(img, kind, 0.0, np.random.default_rng(7))
    assert np.allclose(art, img, atol=1e-6)


@pytest.mark.parametrize("kind", ALL_ARTIFACT_KINDS)
def test_alpha_monotonic_change(kind):
    img = _img()
    art_lo, _ = inject(img, kind, 0.3, np.random.default_rng(42))
    art_hi, _ = inject(img, kind, 0.9, np.random.default_rng(42))
    # higher opacity => larger deviation from the clean image
    assert np.abs(art_hi - img).mean() > np.abs(art_lo - img).mean()


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        inject(_img(), "not_a_kind", 0.5, np.random.default_rng(0))


def test_remove_with_source_is_exact_counterfactual():
    img = _img()
    art, mask = inject(img, "ruler", 1.0, np.random.default_rng(3))
    assert not np.array_equal(art, img)  # injection changed pixels
    recovered = remove(art, mask, source=img)
    assert np.array_equal(recovered, img)  # gold counterfactual is exact


def test_remove_inpaint_changes_masked_region_only_approx():
    img = _img()
    art, mask = inject(img, "marker_ink", 1.0, np.random.default_rng(5))
    out = remove(art, mask, method="inpaint")  # numpy fallback (no cv2 locally)
    m = mask > 0.05
    # outside the mask the image is untouched; inside it no longer equals the ink layer
    assert np.allclose(out[~m], img[~m], atol=1e-6)
    assert np.abs(out[m] - art[m]).mean() > 0.05


def test_accepts_uint8_and_grayscale():
    u8 = (np.random.default_rng(0).integers(0, 256, (32, 32, 3))).astype(np.uint8)
    art, _ = inject(u8, "dark_corner", 0.6, np.random.default_rng(0))
    assert art.dtype == np.float32 and art.max() <= 1.0
    gray = np.random.default_rng(1).uniform(0, 1, (32, 32)).astype(np.float32)
    art2, _ = inject(gray, "ruler", 0.6, np.random.default_rng(0))
    assert art2.shape == (32, 32, 3)


def test_make_biased_set_consistency():
    rng = np.random.default_rng(0)
    imgs = [_img(32, 32, seed=i) for i in range(40)]
    labels = (rng.random(40) < 0.5).astype(int)
    items, summary = A.make_biased_set(imgs, labels, "ruler", rho=0.9, alpha=0.8, seed=1)
    assert len(items) == 40 and summary["n"] == 40
    for it in items:
        # mask is non-empty iff the artifact is present
        assert (it["mask"].sum() > 0) == it["present"]
        assert "clean" in it
    # rho=0.9 => positives much more likely to carry the artifact than negatives
    r = summary["realized"]
    assert r["p_present_given_pos"] > r["p_present_given_neg"]


def test_default_kinds_are_real_overlays():
    # the pipeline default is the real photographic overlays; legacy synthetic is opt-in
    assert ARTIFACT_KINDS == ["ruler", "arrow"]
    assert set(LEGACY_ARTIFACT_KINDS) == {
        "ruler_synthetic",
        "marker_ink",
        "dark_corner",
        "black_corner",
    }
    assert set(ARTIFACT_KINDS).isdisjoint(LEGACY_ARTIFACT_KINDS)


def test_black_corner_is_realistic_dermoscope_fov():
    # black_corner models a REAL dermoscope: circular optics on a rectangular sensor, so the
    # skin fills almost the whole frame and black appears only where the sensor corners poke
    # outside the large optical circle (2 far corners when off-centre, all 4 when centred).
    # Low coverage (soft-vignette only, capped at ≤~3%, not the legacy ~21% inscribed disc),
    # so assert the geometric invariants over seeds rather than exact darkness.
    for seed in range(12):
        _, mask = inject(_img(96, 96), "black_corner", 1.0, np.random.default_rng(seed))
        assert mask[48, 48] == 0.0                       # centre: inside the FOV => skin visible
        assert mask.max() > 0.0                          # something is covered (mask.sum() > 0)
        corners = [mask[0, 0], mask[0, -1], mask[-1, 0], mask[-1, -1]]
        assert max(corners) > 0.0                        # the corners carry the coverage
        assert mask[48, 48] < max(corners)               # corners darker than the centre
        assert 0.0 < mask.mean() < 0.15                  # low coverage, not the old ~21% disc


def test_black_corner_circle_is_hard_disc():
    # the legacy inscribed-circle cutoff is preserved under black_corner_circle:
    # centre visible (mask==0), corners black (mask==1), strictly binary sharp edge.
    _, mask = inject(_img(64, 64), "black_corner_circle", 0.8, np.random.default_rng(0))
    assert mask[32, 32] == 0.0            # centre pixel: inside the circle => fully visible
    assert mask[0, 0] == 1.0              # corner pixel: outside the circle => fully black
    assert set(np.unique(mask)).issubset({0.0, 1.0})  # binary: sharp edge, no gradient


@pytest.mark.parametrize("kind", ["ruler", "arrow", "overlay_both"])
def test_real_overlay_is_per_pixel_and_covers(kind):
    img = _img(120, 160)
    art, mask = inject(img, kind, 1.0, np.random.default_rng(2))
    assert art.shape == img.shape and mask.shape == img.shape[:2]
    assert mask.sum() > 0                                   # overlay actually landed
    changed = np.abs(art - img).sum(-1) > 1e-6
    assert changed.any()                                    # it actually altered the image
    assert not (changed & (mask <= 0.0)).any()              # but only under the overlay footprint
    # exact gold counterfactual holds for the photographic overlays too
    assert np.array_equal(remove(art, mask, source=img), A._to_float_rgb(img))


@pytest.mark.parametrize("kind", ["ruler", "arrow", "overlay_both"])
@pytest.mark.parametrize("hw", [(32, 32), (48, 80), (224, 224)])
def test_real_overlay_nonempty_across_sizes_and_seeds(kind, hw):
    h, w = hw
    for s in range(5):  # footprint guarantee must hold for every seed / canvas size
        _, mask = inject(_img(h, w, seed=s), kind, 1.0, np.random.default_rng(s))
        assert mask.sum() > 0
