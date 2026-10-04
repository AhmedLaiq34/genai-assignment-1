"""FS2K paired dataset with identical paired transforms. Implements CONTRACTS 3.7.

Item returned by FS2KDataset:
    (photo, sketch, style, pair_id)
    photo  : float32 [3,128,128] in [-1,1]
    sketch : float32 [1,128,128] in [-1,1]
    style  : int64 scalar tensor, 0/1/2
    pair_id: str, e.g. "photo1_image0110"

Data layout under the data root (made by genai.fs2k.prepare):
    splits/fs2k_split.json
    cache/fs2k128/{photo,sketch}/<pair_id>.png   every pair resized straight to 128
    cache/fs2k143/{photo,sketch}/<pair_id>.png   official training pairs resized to 143 (for the random crop)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from genai.fs2k import split as sp
from genai.pets.dataset import log_test_access  # same log line format as the pets test split

TEST_LOG_PATH = Path("artifacts/test_access.log")  # overridable (tests use a tmp path)
SPLIT_NAME = "fs2k_split.json"


# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------


def profile_data_root(arg) -> Path:
    """A device-profile name ("local"/"kaggle"/"colab"), a dict with "data_root", or a folder -> a Path.

    A profile reads configs/devices/<name>.yaml: data_root (+ optional dataset_dir). A relative
    data_root (local: "data") is relative to the repository root."""
    from genai.common.paths import ROOT

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
    return root / extra


def resolve_fs2k_paths(data_root) -> dict:
    """Return {"split_file", "cache128", "cache143"} (Paths) for a profile name or a folder.

    The split file is searched with rglob, because Kaggle nests a dataset a few folders deep
    (/kaggle/input/<dataset>/...). The two cache folders sit next to it: <base>/cache/fs2k128 where
    <base> is the folder that contains "splits/". Nothing is guessed: if the split file is not
    found a FileNotFoundError says where we looked."""
    root = profile_data_root(data_root)
    hits = sorted(root.rglob(f"splits/{SPLIT_NAME}")) if root.is_dir() else []
    if not hits:
        raise FileNotFoundError(f"splits/{SPLIT_NAME} not found under {root} (run scripts/prepare_fs2k.py)")
    split_file = hits[0]
    base = split_file.parent.parent
    return {"split_file": split_file, "cache128": base / "cache" / "fs2k128", "cache143": base / "cache" / "fs2k143"}


# --------------------------------------------------------------------------------------
# Paired augmentation
# --------------------------------------------------------------------------------------


def paired_augment(photo: torch.Tensor, sketch: torch.Tensor, crop: int = 128):
    """Same random crop and same horizontal flip for photo [3,H,W] and sketch [1,H,W] (H = W = 143).

    The two images are stacked into ONE 4-channel tensor, so a single crop position and a single
    flip decision are applied to both; they can not get different draws."""
    both = torch.cat([photo, sketch], dim=0)  # [4,H,W]
    _, h, w = both.shape
    top = int(torch.randint(0, h - crop + 1, (1,)))
    left = int(torch.randint(0, w - crop + 1, (1,)))
    both = both[:, top:top + crop, left:left + crop]
    if torch.rand(1).item() < 0.5:  # horizontal flip with p = 0.5
        both = both.flip(-1)
    return both[:3], both[3:]


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------


def _read_png(path: Path, channels: int) -> torch.Tensor:
    """8-bit PNG -> float32 [C,H,W] in [-1,1]."""
    arr = np.asarray(Image.open(path), dtype=np.uint8)
    if channels == 1:
        arr = arr[:, :, None]  # H W -> H W 1
    tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1).float()
    return tensor / 127.5 - 1.0  # 0..255 -> -1..1


class FS2KDataset(Dataset):
    """FS2K photo-sketch pairs of one split ("train", "val" or "test").

    augment=True only has an effect on the train split: it reads the 143 cache and applies
    paired_augment (random 128 crop + flip). Every other case reads the 128 cache untouched.
    The test split is refused unless final_test=True; each allowed access is logged.
    subset: keep only the first N ids (sorted order), for smoke runs and tests."""

    def __init__(self, split: str, data_root, augment: bool = False, final_test: bool = False,
                 subset: int | None = None, test_log_path=None):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be 'train', 'val' or 'test'")
        if split == "test":
            if not final_test:
                raise PermissionError("test split is locked: pass final_test=True (final evaluation only)")
            log_test_access(test_log_path or TEST_LOG_PATH, "fs2k")
        paths = resolve_fs2k_paths(data_root)
        self.split = sp.load_split(paths["split_file"])  # verifies the sha256 values
        self.ids = list(self.split[split])
        if subset is not None:
            self.ids = self.ids[:subset]
        self.pairs = self.split["pairs"]
        self.augment = bool(augment) and split == "train"
        self.cache_dir = paths["cache143"] if self.augment else paths["cache128"]
        missing = [i for i in self.ids if not (self.cache_dir / "photo" / f"{i}.png").exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} cached images missing in {self.cache_dir}, e.g. {missing[:3]}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        pair_id = self.ids[index]
        photo = _read_png(self.cache_dir / "photo" / f"{pair_id}.png", 3)
        sketch = _read_png(self.cache_dir / "sketch" / f"{pair_id}.png", 1)
        if self.augment:
            photo, sketch = paired_augment(photo, sketch, 128)
        style = torch.tensor(self.pairs[pair_id]["style"], dtype=torch.int64)
        return photo, sketch, style, pair_id
