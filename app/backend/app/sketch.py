"""Helpers of POST /api/sketch (Task 4, style-conditioned face -> sketch generator).

Same split as hard.py: this module holds the pure pieces (sample folder, inference, PNG),
and the route itself lives in main.py, which passes in what it owns (the session).
Model contract (CONTRACTS 3.9): inputs `photo` f32 [N,3,128,128] in [-1,1] and `style`
int64 [N] (0..2); output `sketch` f32 [N,1,128,128] in [-1,1].
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from PIL import Image

NUM_STYLES = 3


def check_style(style: int) -> int:
    """The form field is 1..3 (what the UI shows); returns the model's style id 0..2. 422 otherwise."""
    if style not in range(1, NUM_STYLES + 1):
        raise HTTPException(422, f"style must be 1, 2 or 3 (got {style}).")
    return style - 1


# --------------------------------------------------------------------------------------
# Sample photos: display-only test faces in their own folder (not the pets samples).
# Same rules as /api/samples: only ids found in the folder are accepted, which also blocks
# path tricks like "../x". A missing or empty folder simply gives an empty list.
# --------------------------------------------------------------------------------------
def sample_ids(samples_dir: Path) -> list:
    return sorted(p.stem for p in samples_dir.glob("*.png"))


def sample_path(samples_dir: Path, sample_id: str) -> Path:
    if sample_id not in sample_ids(samples_dir):
        raise HTTPException(404, f"Unknown sketch sample '{sample_id}'.")
    return samples_dir / f"{sample_id}.png"


def gray_png_bytes(sketch01: np.ndarray) -> bytes:
    """float [1,H,W] in [0,1] -> 8-bit grayscale PNG bytes (mode L). to_png_bytes is RGB only."""
    arr = np.clip(np.rint(sketch01[0] * 255.0), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "L").save(buf, format="PNG")
    return buf.getvalue()


def run_sketch(img01: np.ndarray, style_id: int, session):
    """img01 float32 [3,128,128] in [0,1] -> (grayscale sketch PNG bytes, inference ms)."""
    photo = (img01 * 2.0 - 1.0).astype(np.float32)[None]  # [1,3,128,128] in [-1,1]
    style = np.array([style_id], dtype=np.int64)  # [1]
    t0 = time.perf_counter()
    out = session.run(["sketch"], {"photo": photo, "style": style})[0][0]  # [1,128,128] in [-1,1]
    ms = round((time.perf_counter() - t0) * 1000, 2)
    sketch01 = np.clip((out + 1.0) / 2.0, 0.0, 1.0)
    return gray_png_bytes(sketch01), ms
