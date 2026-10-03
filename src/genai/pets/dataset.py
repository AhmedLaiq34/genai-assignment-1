"""Pets datasets (cached 128x128 uint8, runtime corruption, manifest replay).
Implements CONTRACTS 3.1, 3.4, 3.5.

Both datasets return the same tuple:
    (corrupted, clean, cond_id, severity_id)
    corrupted, clean : float32 tensors [3,128,128] in [0,1]
    cond_id          : 0 clean, 1 salt_pepper, 2 gaussian_blur, 3 occlusion
    severity_id      : 0/1/2 = low/medium/high (tertile for train/val, fixed level for test), -1 for clean

PetsTrainDataset   : fresh random corruption on every __getitem__ (train split only).
PetsManifestDataset: fixed corruption rebuilt from a manifest row (val or test), deterministic.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from genai.common import constants as C
from genai.pets import corruptions as corr
from genai.pets import manifests as mf
from genai.pets import samplers
from genai.pets import split as sp

SEVERITY_TO_ID = {None: -1, "low": 0, "medium": 1, "high": 2}
TEST_LOG_PATH = Path("artifacts/test_access.log")  # overridable (tests use a tmp path)


# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------


def _find_dir(root: Path, *parts: str) -> Path:
    """Find root/<parts> directly, or inside root/data, or inside one sub-folder of root
    (Kaggle mounts a dataset as /kaggle/input/<dataset-name>/...)."""
    candidates = [root.joinpath(*parts), root.joinpath("data", *parts)]
    if root.is_dir():
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            candidates += [sub.joinpath(*parts), sub.joinpath("data", *parts)]
    for c in candidates:
        if c.is_dir():
            return c
    return candidates[0]  # not found: return the expected place so the error message is clear


def resolve_data_paths(device_profile_or_cfg="local") -> dict:
    """Return {"root", "cache", "splits", "manifests"} (Paths) for a device profile.

    Accepts a profile name ("local"/"kaggle"/"colab" -> configs/devices/<name>.yaml), a dict
    that has "data_root", or a plain path. A relative data_root (local: "data") is relative
    to the repository root. Optional key "dataset_dir" is appended to data_root (Kaggle).
    """
    from genai.common.paths import ROOT

    arg = device_profile_or_cfg
    if isinstance(arg, dict):
        data_root, extra = arg["data_root"], arg.get("dataset_dir", "")
    elif isinstance(arg, str) and arg in ("local", "kaggle", "colab"):
        import yaml
        prof = yaml.safe_load((ROOT / "configs" / "devices" / f"{arg}.yaml").read_text())
        data_root, extra = prof["data_root"], prof.get("dataset_dir", "")
    else:
        data_root, extra = str(arg), ""
    root = Path(data_root)
    if not root.is_absolute():
        root = ROOT / root
    root = root / extra
    return {
        "root": root,
        "cache": _find_dir(root, "cache", "pets128"),
        "splits": _find_dir(root, "splits"),
        "manifests": _find_dir(root, "manifests"),
    }


def _as_paths(data_root) -> dict:
    return data_root if isinstance(data_root, dict) and "cache" in data_root else resolve_data_paths(data_root)


# --------------------------------------------------------------------------------------
# Cache: uint8 images held in RAM
# --------------------------------------------------------------------------------------


class _Cache:
    """One cached partition ("trainval" or "test"): uint8 [N,128,128,3] + id list.

    Images load lazily and are NOT pickled, so each DataLoader worker (Windows spawns new
    processes) reads the .npy itself instead of receiving a copy through a pipe.
    """

    def __init__(self, cache_dir: Path, part: str):
        self.images_path = Path(cache_dir) / f"{part}_images.npy"
        ids_path = Path(cache_dir) / f"{part}_ids.json"
        if not self.images_path.exists() or not ids_path.exists():
            raise FileNotFoundError(f"cache missing in {cache_dir} (run scripts/prepare_pets.py)")
        self.ids = json.loads(ids_path.read_text())
        self.index = {i: k for k, i in enumerate(self.ids)}
        self._images = None

    @property
    def images(self) -> np.ndarray:
        if self._images is None:
            self._images = np.load(self.images_path)
        return self._images

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_images"] = None
        return state

    def get01(self, image_id: str) -> torch.Tensor:
        """Clean image as float32 [3,128,128] in [0,1]."""
        arr = self.images[self.index[image_id]]  # HWC uint8
        return torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)


# --------------------------------------------------------------------------------------
# Training dataset
# --------------------------------------------------------------------------------------


class PetsTrainDataset(Dataset):
    """Train images with a NEW random corruption every time an item is loaded.

    policy: "iid_uniform" | "fixed:k" decide the class per item; "balanced_batch" needs a
    BalancedBatchSampler, which passes (index, cond_id) tuples as the item key.
    seed: optional extra salt for the per-item RNG (the RNG still differs between loads).
    """

    def __init__(self, split: str = "train", policy: str = "iid_uniform", data_root="local", seed: int = C.SEED):
        if split not in ("train", "val"):
            raise ValueError("PetsTrainDataset only serves 'train' (or 'val' for debugging); "
                             "the test split goes through PetsManifestDataset(final_test=True)")
        samplers.parse_policy(policy)  # fail early on a bad policy string
        self.policy, self.seed = policy, seed
        paths = _as_paths(data_root)
        self.split = sp.read_split(Path(paths["splits"]) / "pets_split.json")  # verifies hashes
        self.cache = _Cache(paths["cache"], "trainval")
        self.ids = list(self.split[split])
        missing = [i for i in self.ids if i not in self.cache.index]
        if missing:
            raise ValueError(f"{len(missing)} split ids are not in the cache, e.g. {missing[:3]}")
        self._counter = 0  # counts loads in THIS process (each worker has its own copy)

    def __len__(self) -> int:
        return len(self.ids)

    def _rng(self) -> np.random.Generator:
        """A fresh generator for every load. Mixed from: the torch seed of this process
        (differs per DataLoader worker and per epoch), the user seed and a load counter.
        No shared global state, so workers never repeat each other's corruptions."""
        self._counter += 1
        return np.random.default_rng([torch.initial_seed() % (2**32), self.seed, self._counter])

    def __getitem__(self, item):
        if isinstance(item, tuple):  # (index, cond_id) from BalancedBatchSampler
            index, cond_id = item
        else:
            index, cond_id = item, None
        rng = self._rng()
        if cond_id is None:
            cond_id = samplers.draw_condition(self.policy, rng)  # raises for balanced_batch
        clean = self.cache.get01(self.ids[index])
        params = corr.sample_params(cond_id, rng)
        corrupted = corr.apply(clean, params, rng)
        sev = SEVERITY_TO_ID[corr.severity_label(cond_id, params)]
        return corrupted, clean, cond_id, sev


# --------------------------------------------------------------------------------------
# Manifest dataset (val / test)
# --------------------------------------------------------------------------------------


def log_test_access(path, note: str = "") -> None:
    """Append one line to the test access log (CONTRACTS 3.4)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{stamp} test split opened {note}\n")


class PetsManifestDataset(Dataset):
    """Deterministic val/test data: row i of the manifest -> same tensors on every load.

    manifest_path: pets_val_manifest.jsonl or pets_test_manifest.jsonl (sha256 verified).
    The test split is refused unless final_test=True; each allowed access is logged.
    """

    def __init__(self, manifest_path, split: str = "val", final_test: bool = False,
                 data_root="local", test_log_path=None):
        if split not in ("val", "test"):
            raise ValueError("split must be 'val' or 'test'")
        if split == "test":
            if not final_test:
                raise PermissionError("test split is locked: pass final_test=True (final evaluation only)")
            log_test_access(test_log_path or TEST_LOG_PATH, f"manifest={manifest_path}")
        paths = _as_paths(data_root)
        self.split = sp.read_split(Path(paths["splits"]) / "pets_split.json")  # verifies hashes
        self.rows = mf.load_manifest(manifest_path)  # verifies the manifest sha256
        bad = [r for r in self.rows if r["split"] != split]
        if bad:
            raise ValueError(f"manifest has {len(bad)} rows whose split is not '{split}'")
        allowed = set(self.split["val"] if split == "val" else self.split["test"]["ids"])
        if {r["image_id"] for r in self.rows} - allowed:
            raise ValueError("manifest contains image ids that are not in the split")
        self.cache = _Cache(paths["cache"], "trainval" if split == "val" else "test")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        clean = self.cache.get01(row["image_id"])
        corrupted = mf.render(row, clean)  # seeded by the row, so identical every time
        return corrupted, clean, row["cond_id"], SEVERITY_TO_ID[row["severity"]]
