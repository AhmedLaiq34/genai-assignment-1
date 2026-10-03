"""Checkpoint save/load/promote and sha256 helpers. Implements CONTRACTS 3.8."""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from genai.common import paths

# Keys every checkpoint must contain (CONTRACTS 3.8).
CHECKPOINT_KEYS = (
    "model", "optimizer", "scheduler", "scaler", "epoch", "global_step",
    "best_metric", "config", "seed", "split_sha256", "manifest_sha256",
    "git_commit", "torch_version", "created_utc",
)


def sha256_file(path) -> str:
    """Hex sha256 of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(cwd=None):
    """Current git commit hash, or None if git is missing / no commits yet."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd or paths.ROOT,
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        return None


def build_checkpoint(model, optimizer=None, scheduler=None, scaler=None, epoch=0,
                     global_step=0, best_metric=None, config=None, seed=None,
                     split_sha256=None, manifest_sha256=None) -> dict:
    """Assemble the contract checkpoint dict (fills git/torch/time automatically).

    Pass the model/optimizer/scheduler/scaler objects; we store their state_dicts.
    """
    def sd(obj):
        return obj.state_dict() if obj is not None else None

    return {
        "model": sd(model), "optimizer": sd(optimizer), "scheduler": sd(scheduler),
        "scaler": sd(scaler), "epoch": epoch, "global_step": global_step,
        "best_metric": best_metric, "config": config, "seed": seed,
        "split_sha256": split_sha256, "manifest_sha256": manifest_sha256,
        "git_commit": git_commit(), "torch_version": torch.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def save_checkpoint(path, state: dict) -> None:
    """Atomic save: write a temp file in the same folder, then os.replace.

    A crash mid-write leaves the old checkpoint intact and no temp file behind
    (same folder => same filesystem => the rename is atomic).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())  # make sure bytes hit the disk before the rename
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def load_checkpoint(path, map_location="cpu") -> dict:
    """Load a checkpoint dict. weights_only=False because it holds config and
    RNG state (numpy arrays); only load files you created yourself."""
    return torch.load(path, map_location=map_location, weights_only=False)


# ---------------------------------------------------------------- RNG state
def capture_rng_state() -> dict:
    """Snapshot python, numpy, torch CPU and CUDA RNG states (store in a checkpoint)."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    """Put the RNGs back exactly as captured, so a resumed run continues identically."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


# ------------------------------------------------------------------ promote
def _write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def promote(ckpt, name: str, root=None) -> Path:
    """Copy a finished checkpoint to <models>/checkpoints/<name>.pt and record it
    (name, sha256, size, source) in <models>/MANIFEST.json (created if missing).

    root: optional folder acting as the `models` directory (tests pass a tmp
    dir). Default is the real models/ folder (paths.MODELS_CKPT's parent).
    Re-promoting the same name replaces its manifest entry.
    """
    ckpt = Path(ckpt)
    models_dir = Path(root) if root is not None else paths.MODELS_CKPT.parent
    dest = models_dir / "checkpoints" / f"{name}.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)

    tmp = dest.with_name(dest.name + ".tmp")  # copy then rename => never half-written
    shutil.copyfile(ckpt, tmp)
    os.replace(tmp, dest)

    manifest_path = models_dir / "MANIFEST.json"
    entries = []
    if manifest_path.exists():
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [e for e in entries if e.get("name") != name]
    entries.append({
        "name": name,
        "sha256": sha256_file(dest),
        "size": dest.stat().st_size,
        "source": str(ckpt),
    })
    _write_json_atomic(manifest_path, entries)
    return dest
