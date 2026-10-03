"""Seeding helpers: seed_everything, worker_init_fn. Implements CONTRACTS 3.11."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed python, numpy, torch (CPU) and every CUDA device with one integer.

    Note: cuDNN benchmark mode is NOT touched here. It may stay on for
    training (faster, tiny non-determinism); evaluation stays deterministic
    because the manifests are fixed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA in recent torch, but be explicit:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    """Pass as DataLoader(worker_init_fn=...).

    PyTorch already gives every worker its own torch seed
    (base_seed + worker_id) where base_seed is drawn from the main process RNG.
    We read that value with torch.initial_seed() and use it for python/numpy,
    so all three libraries are different per worker yet reproducible for a
    given seed_everything(seed).
    """
    worker_seed = torch.initial_seed() % (2**32)  # numpy needs a 32-bit seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
