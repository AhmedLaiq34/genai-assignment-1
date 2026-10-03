"""L1 / SSIM / PSNR metrics and fixed objective J. Implements CONTRACTS 3.6.

All image tensors are float, shape N x C x H x W, range [0,1].
The per-image functions (l1, ssim, psnr) return a tensor of shape (N,)
(one value per image); call .mean() for a batch average.
"""
from __future__ import annotations

import torch
from pytorch_msssim import ssim as _ssim

from genai.common.constants import CLASS_NAMES, SEVERITY_NAMES

_PSNR_EPS = 1e-10  # keeps PSNR finite (max 100 dB) when two images are identical


def l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean absolute error per image -> (N,)."""
    return (x - y).abs().flatten(1).mean(dim=1)


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """SSIM per image -> (N,). data_range=1 because images are in [0,1].

    size_average=False means: do NOT average over the batch inside the
    library; we get one SSIM per image and average ourselves. We keep the
    library's default Gaussian window (win_size=11), which is fine at 128x128.
    """
    return _ssim(x, y, data_range=1.0, size_average=False)


def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """PSNR in dB per image -> (N,). Identical images give 100 dB (not inf)."""
    mse = ((x - y) ** 2).flatten(1).mean(dim=1).clamp_min(_PSNR_EPS)
    return 10.0 * torch.log10(1.0 / mse)


def objective_J(l1_value, ssim_value):
    """Fixed study objective J = 0.5*L1 + 0.5*(1-SSIM). Works on floats or tensors."""
    return 0.5 * l1_value + 0.5 * (1.0 - ssim_value)


def objective_J_batch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """J per image for a batch of (restored, clean) -> (N,)."""
    return objective_J(l1(x, y), ssim(x, y))


objective_j = objective_J_batch  # scaffold name kept as alias


def _summarise(rows: list) -> dict:
    """rows: list of (l1, ssim, psnr, J) tuples -> mean dict + count."""
    t = torch.tensor(rows, dtype=torch.float64)
    mean = t.mean(dim=0).tolist()
    return {"l1": mean[0], "ssim": mean[1], "psnr": mean[2], "J": mean[3], "count": len(rows)}


@torch.no_grad()
def evaluate_restoration(model, loader, device) -> dict:
    """Evaluate a restoration model on a loader yielding
    (corrupted, clean, cond_id, severity_id) batches (severity_id = -1 if none).

    Returns {cond_name: {l1, ssim, psnr, J, count}, ..., 'overall': {...},
             'by_severity': {'cond_name/low': {...}, ...}}.
    Only conditions that actually occur appear. Model mode is restored afterwards.
    If the model returns a tuple (e.g. MoE output, weights), the first item is used.
    """
    was_training = model.training
    model.eval()
    by_cond, by_sev, everything = {}, {}, []
    for corrupted, clean, cond_id, sev_id in loader:
        corrupted, clean = corrupted.to(device), clean.to(device)
        out = model(corrupted)
        if isinstance(out, (tuple, list)):
            out = out[0]
        a, b, c = l1(out, clean), ssim(out, clean), psnr(out, clean)
        j = objective_J(a, b)
        for i in range(len(a)):
            row = (a[i].item(), b[i].item(), c[i].item(), j[i].item())
            cond = CLASS_NAMES[int(cond_id[i])]
            everything.append(row)
            by_cond.setdefault(cond, []).append(row)
            if int(sev_id[i]) >= 0:
                key = f"{cond}/{SEVERITY_NAMES[int(sev_id[i])]}"
                by_sev.setdefault(key, []).append(row)
    model.train(was_training)

    result = {name: _summarise(rows) for name, rows in by_cond.items()}
    result["overall"] = _summarise(everything)
    result["by_severity"] = {k: _summarise(v) for k, v in by_sev.items()}
    return result
