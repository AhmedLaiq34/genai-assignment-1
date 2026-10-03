import numpy as np
import pytest
import torch

from genai.common import constants as C
from genai.pets import corruptions as corr


def _img(seed=0):
    return torch.from_numpy(np.random.default_rng(seed).random((3, 128, 128), dtype=np.float32))


def test_salt_fraction_close_to_p_and_shared_across_channels():
    img = torch.full((3, 128, 128), 0.5)
    out = corr.apply(img, {"type": "salt_pepper", "p": 0.1}, np.random.default_rng(0))
    changed = out != 0.5
    assert torch.equal(changed[0], changed[1]) and torch.equal(changed[0], changed[2])
    assert abs(changed[0].float().mean().item() - 0.1) < 0.01
    # changed pixels are only black or white, about half each
    vals = out[0][changed[0]]
    assert set(vals.unique().tolist()) <= {0.0, 1.0}
    assert abs(vals.mean().item() - 0.5) < 0.05


def test_train_param_ranges():
    rng = np.random.default_rng(1)
    for _ in range(200):
        p = corr.sample_params(1, rng)
        assert 0.02 <= p["p"] <= 0.15
        b = corr.sample_params(2, rng)
        assert b["kernel"] in (3, 5, 7) and 0.5 <= b["sigma"] <= 2.5
        o = corr.sample_params(3, rng)
        assert o["n_rects"] in (1, 2, 3) and len(o["rects"]) == o["n_rects"]
        assert 0.10 <= o["achieved_coverage"] <= 0.35
        assert o["achieved_coverage"] == corr.union_coverage(o["rects"])


def test_two_draws_differ():
    a = corr.sample_params(3, np.random.default_rng(1))
    b = corr.sample_params(3, np.random.default_rng(2))
    assert a["rects"] != b["rects"]
    img = _img()
    p = {"type": "salt_pepper", "p": 0.1}
    x = corr.apply(img, p, np.random.default_rng(1))
    y = corr.apply(img, p, np.random.default_rng(2))
    assert not torch.equal(x, y)


def test_union_coverage_with_overlap():
    # two identical rectangles: union is one rectangle, not the sum
    assert corr.union_coverage([[0, 0, 64, 64], [0, 0, 64, 64]]) == pytest.approx(0.25)
    # partial overlap: 64x64 + 64x64 - 32x64 overlap = 6144 px
    cov = corr.union_coverage([[0, 0, 64, 64], [32, 0, 96, 64]])
    assert cov == pytest.approx(6144 / 16384)


def test_occlusion_paints_black_only_inside_rects():
    img = torch.ones(3, 128, 128)
    out = corr.apply(img, {"type": "occlusion", "rects": [[10, 20, 30, 40]]}, np.random.default_rng(0))
    assert (out[:, 20:40, 10:30] == 0).all()
    assert out.sum().item() == 3 * (128 * 128 - 20 * 20)


@pytest.mark.parametrize("s", range(3))
def test_test_occlusion_coverage_and_no_overlap(s):
    n, target = C.TEST_OCC[s]
    for seed in range(20):
        p = corr.test_params(3, C.SEVERITY_NAMES[s], np.random.default_rng(seed))
        assert p["n_rects"] == n and len(p["rects"]) == n
        assert abs(p["achieved_coverage"] - target) <= 0.01
        total = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in p["rects"])
        assert total / 128**2 == pytest.approx(p["achieved_coverage"])  # no overlap


def test_test_salt_and_blur_table():
    for s, name in enumerate(C.SEVERITY_NAMES):
        assert corr.test_params(1, name, None)["p"] == C.TEST_SALT_P[s]
        b = corr.test_params(2, name, None)
        assert (b["kernel"], b["sigma"]) == C.TEST_BLUR[s]


def test_blur_smooths_and_keeps_range_and_shape():
    img = _img()
    out = corr.apply(img, {"type": "gaussian_blur", "kernel": 7, "sigma": 2.5}, np.random.default_rng(0))
    assert out.shape == img.shape and out.dtype == torch.float32
    assert 0 <= out.min() and out.max() <= 1
    assert out.std() < img.std()  # blur removes variance of a noise-like image
    # constant image stays constant (kernel normalised, reflect padding)
    c = torch.full((3, 128, 128), 0.3)
    assert torch.allclose(corr.apply(c, {"type": "gaussian_blur", "kernel": 5, "sigma": 1.5}, None), c, atol=1e-6)


def test_input_not_modified():
    img = _img()
    before = img.clone()
    for cond in range(4):
        corr.apply(img, corr.sample_params(cond, np.random.default_rng(0)), np.random.default_rng(0))
    assert torch.equal(img, before)


def test_severity_label_tertiles():
    assert corr.severity_label(0, {"type": "clean"}) is None
    assert corr.severity_label(1, {"p": 0.03}) == "low"
    assert corr.severity_label(1, {"p": 0.085}) == "medium"
    assert corr.severity_label(1, {"p": 0.14}) == "high"
    assert corr.severity_label(2, {"sigma": 0.6}) == "low"
    assert corr.severity_label(2, {"sigma": 2.4}) == "high"
    assert corr.severity_label(3, {"achieved_coverage": 0.22}) == "medium"
