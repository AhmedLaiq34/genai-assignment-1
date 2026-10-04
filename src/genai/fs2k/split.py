"""FS2K stratified 15% validation split (seed 42). Implements CONTRACTS 3.7.

The official training portion (anno_train.json) is split into train/val with
sklearn train_test_split(test_size=0.15, stratify=style, random_state=42) on the SORTED pair ids.
The official test portion (anno_test.json) is kept as the locked test set.

Split dict layout (what data/splits/fs2k_split.json stores):
    {"seed": 42, "val_fraction": 0.15,
     "train": [ids], "val": [ids], "test": [ids],            (each list sorted)
     "test_locked": true,
     "style_encoding": {"0": "Style 1", "1": "Style 2", "2": "Style 3"},
     "style_counts": {"train": {"0": n, ...}, "val": {...}, "test": {...}},
     "pairs": {id: {"photo": rel_path, "sketch": rel_path, "style": int}},
     "sha256": {"train": hex, "val": hex, "test": hex}}
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

STYLE_ENCODING = {"0": "Style 1", "1": "Style 2", "2": "Style 3"}  # UI label for each stored id


def ids_sha256(ids: list) -> str:
    """sha256 of an id list (ids joined by newlines, in the stored order)."""
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def style_counts(ids: list, pairs: dict) -> dict:
    """{"0": n0, "1": n1, "2": n2} for a list of ids (JSON keys are strings)."""
    counts = Counter(pairs[i]["style"] for i in ids)
    return {str(k): counts.get(k, 0) for k in (0, 1, 2)}


def make_split(train_pairs: list, test_pairs: list, seed: int = 42, val_fraction: float = 0.15) -> dict:
    """Build the split dict from resolved pairs (see pairs.resolve_pairs)."""
    pairs = {}
    for p in list(train_pairs) + list(test_pairs):
        if p["id"] in pairs:
            raise ValueError(f"duplicate pair id {p['id']} (train and test must not share ids)")
        pairs[p["id"]] = {"photo": p["photo"], "sketch": p["sketch"], "style": int(p["style"])}
    train_ids_all = sorted(p["id"] for p in train_pairs)  # sort first so the result does not depend on file order
    styles = [pairs[i]["style"] for i in train_ids_all]
    train, val = train_test_split(train_ids_all, test_size=val_fraction, stratify=styles, random_state=seed)
    train, val = sorted(train), sorted(val)
    test = sorted(p["id"] for p in test_pairs)
    split = {
        "seed": seed,
        "val_fraction": val_fraction,
        "train": train,
        "val": val,
        "test": test,
        "test_locked": True,
        "style_encoding": dict(STYLE_ENCODING),
        "style_counts": {name: style_counts(ids, pairs) for name, ids in (("train", train), ("val", val), ("test", test))},
        "pairs": {i: pairs[i] for i in sorted(pairs)},
        "sha256": {"train": ids_sha256(train), "val": ids_sha256(val), "test": ids_sha256(test)},
    }
    return split


def verify_split(split: dict) -> None:
    """Recompute the hashes and check the three sets are disjoint. Raises ValueError on a problem."""
    for name in ("train", "val", "test"):
        if ids_sha256(split[name]) != split["sha256"][name]:
            raise ValueError(f"split sha256 mismatch for '{name}' ids")
    train, val, test = set(split["train"]), set(split["val"]), set(split["test"])
    if train & val or train & test or val & test:
        raise ValueError("split sets overlap")


def save_split(split: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=1), encoding="utf-8")


def load_split(path) -> dict:
    """Read the JSON file and verify its sha256 values (fails loudly if it was edited)."""
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    verify_split(split)
    return split
