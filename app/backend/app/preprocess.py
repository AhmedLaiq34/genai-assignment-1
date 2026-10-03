"""Image preprocessing shared by every endpoint.

It MUST match how the training cache was made (scripts/prepare_pets.py: load_128):
EXIF rotation fixed -> RGB -> bicubic resize straight to 128x128 (no crop) -> float32 in [0, 1].
Keeping it in ONE function means upload, samples and training can not drift apart.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

SIZE = 128  # same as genai.common.constants.IMG_SIZE
ALLOWED_FORMATS = {"PNG", "JPEG", "MPO", "WEBP"}  # MPO is a JPEG variant some cameras write
Image.MAX_IMAGE_PIXELS = 50_000_000  # refuse decompression bombs (PIL raises an error above 2x this)


class BadImage(ValueError):
    """The upload is not a decodable PNG/JPEG/WebP."""


def decode_upload(data: bytes) -> Image.Image:
    """Really decode the bytes (not just trust the file name / content type)."""
    try:
        img = Image.open(io.BytesIO(data))
        if img.format not in ALLOWED_FORMATS:
            raise BadImage(f"Unsupported image format {img.format}; use PNG, JPEG or WebP.")
        img.load()  # forces the full decode so corrupt files fail here
    except BadImage:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as e:
        raise BadImage(f"Could not decode the file as an image ({e.__class__.__name__}).") from e
    return img


def to_128_uint8(img: Image.Image) -> np.ndarray:
    """PIL image -> uint8 [128,128,3] exactly like the training cache."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = img.resize((SIZE, SIZE), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def preprocess_bytes(data: bytes) -> np.ndarray:
    """Upload bytes -> float32 [3,128,128] in [0,1] (channels first, as the models expect)."""
    arr = to_128_uint8(decode_upload(data))
    return np.ascontiguousarray(arr.transpose(2, 0, 1).astype(np.float32) / 255.0)


def to_png_bytes(img01: np.ndarray) -> bytes:
    """float [3,H,W] in [0,1] -> PNG bytes."""
    arr = np.clip(np.rint(img01 * 255.0), 0, 255).astype(np.uint8).transpose(1, 2, 0)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()
