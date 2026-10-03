"""Timing helpers (CUDA-synchronised). Feeds timing_ms (CONTRACTS 3.10) and docs/BENCHMARKS.md."""
from __future__ import annotations

import time

import torch


def _sync(device=None) -> None:
    """Wait for queued GPU work so the clock measures real compute time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


class Timer:
    """Context manager: `with Timer() as t: ...` then read t.seconds / t.ms.

    Synchronises CUDA before starting and before stopping (no-op on CPU).
    """

    def __init__(self, device=None):
        self.device = device
        self.seconds = 0.0

    def __enter__(self):
        _sync(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        _sync(self.device)
        self.seconds = time.perf_counter() - self._start
        return False  # never swallow exceptions

    @property
    def ms(self) -> float:
        return self.seconds * 1000.0


def time_block(fn, *args, **kwargs):
    """Call fn(*args, **kwargs); return (result, elapsed_ms)."""
    with Timer() as t:
        result = fn(*args, **kwargs)
    return result, t.ms
