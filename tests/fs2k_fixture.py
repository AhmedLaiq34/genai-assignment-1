"""Mini-FS2K test fixture (no real data needed). Import with:  from fs2k_fixture import ...

Helpers (names are stable, other test files may rely on them):

    make_mini_fs2k(root, n_per_style=14, n_test_per_style=4, seed=0) -> Path
        Writes a tiny fake FS2K download directly into `root` (same layout and naming as the real one):
            root/anno_train.json, root/anno_test.json     (lists of dicts with "image_name" and "style")
            root/photo/photo{1,2,3}/imageNNNN.jpg
            root/sketch/sketch{1,2,3}/sketchNNNN.{jpg,png}   (sketch2 files are .png, the others .jpg)
        Photos are small random smooth RGB pictures of different non-square sizes; each sketch is
        the GRAYSCALE version of its photo (so sketch = gray(photo) can be checked after transforms).
        Train has n_per_style entries per style (style 0/1/2 -> 3*n_per_style entries), test has
        n_test_per_style per style. Returns `root`.

    build_mini_fs2k_data(tmp_path, n_per_style=14, n_test_per_style=4, seed=0) -> dict
        make_mini_fs2k into tmp_path/raw/FS2K, then resolve pairs, make the split and build the caches
        under tmp_path/data (the layout genai.fs2k.dataset expects). Returns
            {"raw_root": Path (the FS2K folder), "data_root": Path (pass this to FS2KDataset),
             "split": dict, "split_file": Path, "cache_root": Path (holds fs2k128 and fs2k143)}

    gray(photo) -> sketch   (torch, [3,H,W] -> [1,H,W]) the grayscale rule used by the fixture, for
        checking paired transforms on tensors.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from genai.fs2k import pairs as pr
from genai.fs2k import prepare as prep
from genai.fs2k import split as sp

SIZES = [(96, 112), (120, 100), (104, 104)]  # (height, width) per photo folder, non-square on purpose


def gray(photo):
    """Same weights PIL uses for RGB -> L: 0.299 R + 0.587 G + 0.114 B. Works on a torch [3,H,W] tensor."""
    return (0.299 * photo[0] + 0.587 * photo[1] + 0.114 * photo[2])[None]


def _smooth_photo(rng: np.random.Generator, height: int, width: int) -> Image.Image:
    small = rng.random((6, 6, 3))  # low-resolution noise, enlarged -> smooth picture
    img = Image.fromarray((small * 255).astype(np.uint8)).resize((width, height), Image.BICUBIC)
    return img


def _write_entries(root: Path, entries_spec: list, rng: np.random.Generator) -> list:
    """entries_spec: list of (folder, number, style). Writes the image files, returns anno dicts."""
    annos = []
    for folder, number, style in entries_spec:
        height, width = SIZES[folder - 1]
        photo = _smooth_photo(rng, height, width)
        sketch = photo.convert("L")
        photo_path = root / "photo" / f"photo{folder}" / f"image{number:04d}.jpg"
        ext = ".png" if folder == 2 else ".jpg"  # like the real download
        sketch_path = root / "sketch" / f"sketch{folder}" / f"sketch{number:04d}{ext}"
        photo_path.parent.mkdir(parents=True, exist_ok=True)
        sketch_path.parent.mkdir(parents=True, exist_ok=True)
        photo.save(photo_path, quality=95)
        sketch.save(sketch_path, quality=95)  # quality only matters for JPEG
        annos.append({"image_name": f"photo{folder}/image{number:04d}", "hair": 0, "style": style})
    return annos


def make_mini_fs2k(root, n_per_style: int = 14, n_test_per_style: int = 4, seed: int = 0) -> Path:
    root = Path(root)
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    number = 0
    for anno_name, per_style in (("anno_train.json", n_per_style), ("anno_test.json", n_test_per_style)):
        spec = []
        for style in (0, 1, 2):
            for k in range(per_style):
                number += 1  # image numbers are unique across train and test
                spec.append((1 + (number % 3), number, style))  # spread over the three photo folders
        (root / anno_name).write_text(json.dumps(_write_entries(root, spec, rng)), encoding="utf-8")
    return root


def build_mini_fs2k_data(tmp_path, n_per_style: int = 14, n_test_per_style: int = 4, seed: int = 0) -> dict:
    tmp_path = Path(tmp_path)
    raw_root = make_mini_fs2k(tmp_path / "raw" / "FS2K", n_per_style, n_test_per_style, seed)
    train_pairs = pr.resolve_pairs(raw_root, "anno_train.json")
    test_pairs = pr.resolve_pairs(raw_root, "anno_test.json")
    split = sp.make_split(train_pairs, test_pairs, seed=42, val_fraction=0.15)
    data_root = tmp_path / "data"
    split_file = data_root / "splits" / "fs2k_split.json"
    sp.save_split(split, split_file)
    cache_root = data_root / "cache"
    prep.build_cache(raw_root, split, cache_root)
    return {"raw_root": raw_root, "data_root": data_root, "split": split, "split_file": split_file, "cache_root": cache_root}
