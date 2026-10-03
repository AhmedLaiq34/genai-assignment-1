"""Official trainval 80/20 split (seed 42) and locked test IDs. Implements CONTRACTS section 3.4.

Split dict layout (what pets_split.json stores):
    {"seed": 42,
     "train": [ids...], "val": [ids...],
     "test": {"locked": true, "ids": [ids...]},
     "sha256": {"train": hex, "val": hex, "test": hex}}
Works from plain lists of image IDs (no dataset download needed).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from genai.common import constants as C


def ids_sha256(ids: list) -> str:
    """sha256 of an ID list (IDs joined by newlines, in the stored order)."""
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def make_split(trainval_ids: list, test_ids: list, seed: int = C.SEED) -> dict:
    """Sort the trainval IDs, shuffle with default_rng(seed), first floor(0.8 n) -> train."""
    ids = sorted(str(i) for i in trainval_ids)
    perm = np.random.default_rng(seed).permutation(len(ids))
    shuffled = [ids[i] for i in perm]
    n_train = len(ids) * 4 // 5  # integer maths for floor(0.8 n), avoids float rounding
    train, val = shuffled[:n_train], shuffled[n_train:]
    test = sorted(str(i) for i in test_ids)
    return {
        "seed": seed,
        "train": train,
        "val": val,
        "test": {"locked": True, "ids": test},
        "sha256": {"train": ids_sha256(train), "val": ids_sha256(val), "test": ids_sha256(test)},
    }


def verify_split(split: dict) -> None:
    """Recompute the hashes and raise ValueError if they do not match the stored ones."""
    parts = {"train": split["train"], "val": split["val"], "test": split["test"]["ids"]}
    for name, ids in parts.items():
        if ids_sha256(ids) != split["sha256"][name]:
            raise ValueError(f"split sha256 mismatch for '{name}' ids")
    if set(split["train"]) & set(split["val"]) or set(split["train"]) & set(parts["test"]) \
            or set(split["val"]) & set(parts["test"]):
        raise ValueError("split sets overlap")


def split_sha256(split: dict) -> str:
    """One hash for the whole split (hash of the three list hashes); stored in checkpoints."""
    h = split["sha256"]
    return hashlib.sha256((h["train"] + h["val"] + h["test"]).encode("utf-8")).hexdigest()


def write_split(split: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=1), encoding="utf-8")


def read_split(path) -> dict:
    """Read the JSON file and verify its hashes (fails loudly if it was edited)."""
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    verify_split(split)
    return split
