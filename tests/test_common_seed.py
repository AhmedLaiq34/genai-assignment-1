"""Tests for genai.common.seed."""
import random

import numpy as np
import torch

from genai.common.seed import seed_everything, worker_init_fn


def _draw():
    return random.random(), np.random.rand(), torch.rand(1).item()


def test_seed_everything_reproducible():
    seed_everything(7)
    a = _draw()
    seed_everything(7)
    b = _draw()
    seed_everything(8)
    c = _draw()
    assert a == b and a != c


def test_worker_init_fn_seeds_python_and_numpy_from_torch():
    torch.manual_seed(123)
    worker_init_fn(0)
    a = (random.random(), np.random.rand())
    torch.manual_seed(123)
    worker_init_fn(0)
    b = (random.random(), np.random.rand())
    torch.manual_seed(124)
    worker_init_fn(0)
    c = (random.random(), np.random.rand())
    assert a == b and a != c


class _NumpyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, i):
        return np.random.rand()


def _run_loader():
    seed_everything(5)
    g = torch.Generator().manual_seed(5)
    dl = torch.utils.data.DataLoader(_NumpyDataset(), batch_size=1, num_workers=2,
                                     worker_init_fn=worker_init_fn, generator=g)
    return [float(x) for x in dl]


def test_dataloader_workers_differ_but_reproduce():
    r1, r2 = _run_loader(), _run_loader()
    assert r1 == r2
    assert len(set(r1)) == 4  # workers do not all repeat the same numpy stream
