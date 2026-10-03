"""Salt-and-pepper, Gaussian blur and occlusion corruptions. Implements CONTRACTS section 3.3.

All images are float32 torch tensors of shape [3, H, W] with values in [0, 1].

API (see the bottom half of the file for the dispatchers):
    sample_params(cond_id, rng)            -> params dict (training distribution)
    test_params(cond_id, severity, rng)    -> params dict (fixed test severity: "low"/"medium"/"high")
    apply(img01, params, rng)              -> corrupted tensor (same shape/dtype)
    severity_label(cond_id, params)        -> "low" | "medium" | "high" | None
    union_coverage(rects, size)            -> fraction of pixels covered by the union of rectangles

Each corruption type also has its own functions (sample_salt_pepper_params, apply_salt_pepper, ...).

The params dict is plain JSON (ints, floats, strings, lists). It is exactly what the
manifests store. Randomness that is NOT in params (which pixels get salt/pepper) comes from
the `rng` passed to apply(), so the same (params, seed) always gives the same image.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from genai.common import constants as C

# --------------------------------------------------------------------------------------
# Salt and pepper
# --------------------------------------------------------------------------------------


def sample_salt_pepper_params(rng: np.random.Generator) -> dict:
    """Training distribution: p ~ U(0.02, 0.15)."""
    p = float(rng.uniform(*C.SALT_P_RANGE))
    return {"type": "salt_pepper", "p": p}


def apply_salt_pepper(img01: torch.Tensor, params: dict, rng: np.random.Generator) -> torch.Tensor:
    """Each pixel is corrupted with probability p; the SAME pixels are hit in all 3 channels.
    A corrupted pixel becomes white (1) or black (0) with probability 0.5 each."""
    _, h, w = img01.shape
    hit = rng.random((h, w)) < params["p"]  # which pixels are corrupted
    white = rng.random((h, w)) < 0.5  # salt (white) or pepper (black)
    hit = torch.from_numpy(hit)
    value = torch.from_numpy(white).to(img01.dtype)  # 1.0 for salt, 0.0 for pepper
    # hit/value have shape [H, W]; broadcasting copies them across the 3 channels.
    return torch.where(hit, value, img01)


# --------------------------------------------------------------------------------------
# Gaussian blur
# --------------------------------------------------------------------------------------


def sample_blur_params(rng: np.random.Generator) -> dict:
    """Training distribution: kernel in {3,5,7} uniform, sigma ~ U(0.5, 2.5)."""
    kernel = int(rng.choice(C.BLUR_KERNELS))
    sigma = float(rng.uniform(*C.BLUR_SIGMA_RANGE))
    return {"type": "gaussian_blur", "kernel": kernel, "sigma": sigma}


def _gaussian_kernel_1d(kernel: int, sigma: float) -> torch.Tensor:
    """Normalised 1D Gaussian weights, e.g. kernel 3 -> positions -1, 0, 1."""
    x = torch.arange(kernel, dtype=torch.float32) - (kernel - 1) / 2
    k = torch.exp(-(x**2) / (2 * sigma**2))
    return k / k.sum()


def apply_blur(img01: torch.Tensor, params: dict, rng: np.random.Generator = None) -> torch.Tensor:
    """Separable Gaussian blur with reflect padding. Deterministic (rng is unused)."""
    kernel, sigma = params["kernel"], params["sigma"]
    k1d = _gaussian_kernel_1d(kernel, sigma)
    pad = kernel // 2
    x = img01.unsqueeze(0)  # [1, 3, H, W]
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    # Depthwise convolution: the same kernel is applied to each channel separately.
    kh = k1d.view(1, 1, kernel, 1).repeat(3, 1, 1, 1)  # vertical pass
    kw = k1d.view(1, 1, 1, kernel).repeat(3, 1, 1, 1)  # horizontal pass
    x = F.conv2d(x, kh, groups=3)
    x = F.conv2d(x, kw, groups=3)
    return x.squeeze(0).clamp(0.0, 1.0)


# --------------------------------------------------------------------------------------
# Occlusion (black rectangles)
# --------------------------------------------------------------------------------------


def union_coverage(rects: list, size: int = C.IMG_SIZE) -> float:
    """Fraction of the image covered by the UNION of rectangles [x0, y0, x1, y1]
    (x1, y1 exclusive). Overlapping areas are counted once."""
    mask = np.zeros((size, size), dtype=bool)
    for x0, y0, x1, y1 in rects:
        mask[y0:y1, x0:x1] = True
    return float(mask.mean())


def _rects_overlap(rects: list) -> bool:
    """True if any two rectangles share at least one pixel."""
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            a, b = rects[i], rects[j]
            if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                return True
    return False


def _draw_rects(rng, n: int, total_area: float, equal_shares: bool, size: int) -> list:
    """Draw n random rectangles whose areas add up to about total_area (a fraction of
    the image). Aspect ratio is random between 1:2 and 2:1, position is uniform."""
    if equal_shares:
        shares = np.full(n, 1.0 / n)
    else:
        shares = rng.dirichlet(np.ones(n))  # random split of the area, sums to 1
    rects = []
    for share in shares:
        area = share * total_area * size * size  # in pixels
        aspect = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))  # w / h
        w = int(round(np.sqrt(area * aspect)))
        h = int(round(area / max(w, 1)))
        w = int(np.clip(w, 1, size))
        h = int(np.clip(h, 1, size))
        x0 = int(rng.integers(0, size - w + 1))
        y0 = int(rng.integers(0, size - h + 1))
        rects.append([x0, y0, x0 + w, y0 + h])
    return rects


def _sample_rects(rng, n, target, lo, hi, non_overlapping, equal_shares, size=C.IMG_SIZE,
                  max_tries=10000):
    """Rejection sampling: keep drawing until the achieved union coverage is in [lo, hi]
    (and, if requested, the rectangles do not overlap)."""
    for _ in range(max_tries):
        rects = _draw_rects(rng, n, target, equal_shares, size)
        if non_overlapping and _rects_overlap(rects):
            continue
        cov = union_coverage(rects, size)
        if lo <= cov <= hi:
            return rects, cov
    raise RuntimeError(f"occlusion rejection sampling failed (n={n}, target={target})")


def sample_occlusion_params(rng: np.random.Generator) -> dict:
    """Training distribution: n in {1,2,3}, target coverage ~ U(0.10, 0.35); rectangles
    may overlap; accepted only if the achieved UNION coverage is inside (0.10, 0.35)."""
    n = int(rng.choice(C.OCC_N_RECTS))
    target = float(rng.uniform(*C.OCC_COVERAGE_RANGE))
    lo, hi = C.OCC_COVERAGE_RANGE
    rects, cov = _sample_rects(rng, n, target, lo, hi, non_overlapping=False, equal_shares=False)
    return {"type": "occlusion", "n_rects": n, "target_coverage": target,
            "rects": rects, "achieved_coverage": cov}


def apply_occlusion(img01: torch.Tensor, params: dict, rng: np.random.Generator = None) -> torch.Tensor:
    """Paint the stored rectangles black (value 0). Deterministic (rng is unused)."""
    out = img01.clone()
    for x0, y0, x1, y1 in params["rects"]:
        out[:, y0:y1, x0:x1] = 0.0
    return out


# --------------------------------------------------------------------------------------
# Fixed test severities (CONTRACTS 3.3 right-hand column)
# --------------------------------------------------------------------------------------


def test_params(cond_id: int, severity: str, rng: np.random.Generator) -> dict:
    """Params for a fixed test severity. Salt and blur are fully fixed. For occlusion the
    number of rectangles and the coverage are fixed (+-1 pp), the locations come from rng
    and the rectangles never overlap."""
    s = C.SEVERITY_NAMES.index(severity)
    if cond_id == 1:
        return {"type": "salt_pepper", "p": C.TEST_SALT_P[s]}
    if cond_id == 2:
        kernel, sigma = C.TEST_BLUR[s]
        return {"type": "gaussian_blur", "kernel": kernel, "sigma": sigma}
    if cond_id == 3:
        n, target = C.TEST_OCC[s]
        tol = C.TEST_OCC_TOLERANCE
        rects, cov = _sample_rects(rng, n, target, target - tol, target + tol,
                                   non_overlapping=True, equal_shares=True)
        return {"type": "occlusion", "n_rects": n, "target_coverage": target,
                "rects": rects, "achieved_coverage": cov}
    raise ValueError(f"cond_id {cond_id} has no severities")


test_params.__test__ = False  # stop pytest from collecting this as a test function

# --------------------------------------------------------------------------------------
# Dispatchers
# --------------------------------------------------------------------------------------

_SAMPLERS = {1: sample_salt_pepper_params, 2: sample_blur_params, 3: sample_occlusion_params}
_APPLIERS = {"salt_pepper": apply_salt_pepper, "gaussian_blur": apply_blur,
             "occlusion": apply_occlusion}


def sample_params(cond_id: int, rng: np.random.Generator) -> dict:
    """Sample params from the TRAINING distribution of a condition (0 = clean)."""
    if cond_id == 0:
        return {"type": "clean"}
    return _SAMPLERS[cond_id](rng)


def apply(img01: torch.Tensor, params: dict, rng: np.random.Generator) -> torch.Tensor:
    """Apply the corruption described by params["type"]. Always returns a new tensor."""
    if params["type"] == "clean":
        return img01.clone()
    return _APPLIERS[params["type"]](img01, params, rng)


# --------------------------------------------------------------------------------------
# Severity label (tertiles of the training range)
# --------------------------------------------------------------------------------------


def _tertile(value: float, lo: float, hi: float) -> str:
    """Split [lo, hi] into three equal parts and name the part that value falls in."""
    third = (hi - lo) / 3
    if value < lo + third:
        return "low"
    if value < lo + 2 * third:
        return "medium"
    return "high"


def severity_label(cond_id: int, params: dict):
    """Severity of a train/val sample: tertile of salt p, blur sigma or occlusion
    coverage over the training range. Returns None for clean."""
    if cond_id == 1:
        return _tertile(params["p"], *C.SALT_P_RANGE)
    if cond_id == 2:
        return _tertile(params["sigma"], *C.BLUR_SIGMA_RANGE)
    if cond_id == 3:
        return _tertile(params["achieved_coverage"], *C.OCC_COVERAGE_RANGE)
    return None
