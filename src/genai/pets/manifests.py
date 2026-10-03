"""Deterministic val/test corruption manifests (JSONL + sha256). Implements CONTRACTS section 3.4.

Row layout (one JSON object per line):
    {"image_id": str, "split": "val"|"test", "cond_id": 0..3, "cond_name": str,
     "severity": "low"|"medium"|"high"|None,
     "params": {...},              # exactly what corruptions.apply() needs
     "rects": [[x0,y0,x1,y1],...], # occlusion only, else []
     "achieved_coverage": float|None,  # occlusion only
     "seed": int, "manifest_version": 1}

render(row, clean_img) rebuilds the corrupted tensor from a row, identically every time.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from genai.common import constants as C
from genai.pets import corruptions as corr

MANIFEST_VERSION = 1


def row_seed(*parts) -> int:
    """Stable 63-bit seed from (42, "val", image_id, cond, ...). Uses sha256, NOT python's
    hash() (which changes between runs)."""
    text = "|".join(str(p) for p in (C.SEED,) + parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") >> 1


def _make_row(image_id, split, cond_id, severity, params, seed) -> dict:
    is_occ = params["type"] == "occlusion"
    return {
        "image_id": image_id,
        "split": split,
        "cond_id": cond_id,
        "cond_name": C.CLASS_NAMES[cond_id],
        "severity": severity,
        "params": params,
        "rects": params["rects"] if is_occ else [],
        "achieved_coverage": params["achieved_coverage"] if is_occ else None,
        "seed": seed,
        "manifest_version": MANIFEST_VERSION,
    }


def make_val_rows(split: dict) -> list:
    """4 rows per val image (clean + 3 types), params from the TRAINING distributions."""
    rows = []
    for image_id in split["val"]:
        for cond_id in range(C.NUM_CLASSES):
            seed = row_seed("val", image_id, cond_id)
            params = corr.sample_params(cond_id, np.random.default_rng(seed))
            severity = corr.severity_label(cond_id, params)
            rows.append(_make_row(image_id, "val", cond_id, severity, params, seed))
    return rows


def make_test_rows(split: dict) -> list:
    """10 rows per test image: clean + 3 types x 3 fixed severities."""
    rows = []
    for image_id in split["test"]["ids"]:
        seed = row_seed("test", image_id, 0)
        rows.append(_make_row(image_id, "test", 0, None, {"type": "clean"}, seed))
        for cond_id in (1, 2, 3):
            for severity in C.SEVERITY_NAMES:
                seed = row_seed("test", image_id, cond_id, severity)
                params = corr.test_params(cond_id, severity, np.random.default_rng(seed))
                rows.append(_make_row(image_id, "test", cond_id, severity, params, seed))
    return rows


def _sha_path(path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".sha256")


def write_manifest(rows: list, path) -> str:
    """Write rows as JSONL plus <path>.sha256 (hash of the file bytes). Returns the hash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r) + "\n" for r in rows).encode("utf-8")
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    _sha_path(path).write_text(digest + "\n", encoding="utf-8")
    return digest


def load_manifest(path) -> list:
    """Read a JSONL manifest after checking its sha256 file. Raises ValueError on mismatch."""
    path = Path(path)
    data = path.read_bytes()
    expected = _sha_path(path).read_text(encoding="utf-8").strip()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"manifest sha256 mismatch: {path}")
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line]


def build_val_manifest(split: dict, out_path) -> str:
    """Build and write the val manifest; returns its sha256."""
    return write_manifest(make_val_rows(split), out_path)


def build_test_manifest(split: dict, out_path) -> str:
    """Build and write the test manifest; returns its sha256."""
    return write_manifest(make_test_rows(split), out_path)


def render(row: dict, clean_img):
    """Regenerate the corrupted tensor for a row from the clean [3,128,128] tensor.
    The row seed drives the only random part (salt/pepper pixel choice)."""
    return corr.apply(clean_img, row["params"], np.random.default_rng(row["seed"]))
