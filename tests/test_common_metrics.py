"""Tests for genai.common.metrics."""
import math

import torch

from genai.common import metrics as m


class _Identity(torch.nn.Module):
    def forward(self, x):
        return x


def test_identical_images():
    x = torch.rand(2, 3, 128, 128)
    assert torch.allclose(m.l1(x, x), torch.zeros(2))
    assert torch.allclose(m.ssim(x, x), torch.ones(2), atol=1e-5)
    p = m.psnr(x, x)
    assert torch.isfinite(p).all() and (p >= 99).all()
    assert torch.allclose(m.objective_J_batch(x, x), torch.zeros(2), atol=1e-5)


def test_known_l1_and_psnr():
    x = torch.zeros(1, 3, 128, 128)
    y = torch.full_like(x, 0.1)
    assert abs(m.l1(x, y).item() - 0.1) < 1e-6
    assert abs(m.psnr(x, y).item() - 20.0) < 1e-4  # mse=0.01 -> 20 dB


def test_objective_J_formula():
    assert math.isclose(m.objective_J(0.2, 0.8), 0.5 * 0.2 + 0.5 * 0.2)


def test_ssim_drops_for_noise():
    x = torch.rand(1, 3, 128, 128)
    noisy = (x + 0.3 * torch.randn_like(x)).clamp(0, 1)
    assert m.ssim(x, noisy).item() < 0.9


def test_evaluate_restoration_groups_by_condition():
    clean = torch.rand(4, 3, 128, 128)
    corrupted = clean.clone()
    corrupted[2:] = (clean[2:] + 0.1).clamp(0, 1)  # last two are "corrupted"
    cond = torch.tensor([0, 0, 1, 1])
    sev = torch.tensor([-1, -1, 0, 2])
    res = m.evaluate_restoration(_Identity(), [(corrupted, clean, cond, sev)], "cpu")
    assert res["clean"]["count"] == 2 and res["clean"]["l1"] < 1e-6
    assert res["salt_pepper"]["count"] == 2 and res["salt_pepper"]["l1"] > 0.01
    assert res["overall"]["count"] == 4
    assert set(res["by_severity"]) == {"salt_pepper/low", "salt_pepper/high"}
    assert "gaussian_blur" not in res
    for key in ("l1", "ssim", "psnr", "J", "count"):
        assert key in res["overall"]
