"""Corruption-assignment policies: iid_uniform, balanced_batch, fixed:k. Implements CONTRACTS 3.5.

A "policy" string decides WHICH corruption class (0 clean, 1 salt, 2 blur, 3 occlusion) each
training sample gets:

    iid_uniform     every sample draws its class uniformly from the 4 classes (Task 1)
    balanced_batch  every batch holds exactly B/4 samples of each class (Task 2 classifier, Task 3)
    fixed:k         every sample gets class k (Task 2 specialist k)

Design (how Tasks 2/3 use it)
-----------------------------
iid_uniform and fixed:k are per-sample decisions, so the dataset takes them itself with
`draw_condition(policy, rng)`.  balanced_batch is a per-BATCH rule, so a sampler has to pick
the classes before the batch is built.  We therefore use a *batch sampler* that yields lists
of (image_index, cond_id) tuples, and `PetsTrainDataset.__getitem__` accepts such a tuple:

    ds = PetsTrainDataset("train", policy="balanced_batch", data_root=...)
    bs = make_batch_sampler("balanced_batch", len(ds), batch_size=64, seed=42)
    loader = DataLoader(ds, batch_sampler=bs, num_workers=..., worker_init_fn=worker_init_fn)

For the other two policies the same call returns an ordinary shuffled batch sampler of plain
integer indices, so every task builds its loader the same way.
"""
from __future__ import annotations

import numpy as np

from genai.common import constants as C


def parse_policy(policy: str):
    """Return (name, k). k is only set for 'fixed:k'. Raises ValueError for anything else."""
    if policy in ("iid_uniform", "balanced_batch"):
        return policy, None
    if policy.startswith("fixed:"):
        try:
            k = int(policy.split(":", 1)[1])
        except ValueError:
            raise ValueError(f"bad policy '{policy}'") from None
        if not 0 <= k < C.NUM_CLASSES:
            raise ValueError(f"fixed:k needs 0 <= k < {C.NUM_CLASSES}, got {k}")
        return "fixed", k
    raise ValueError(f"unknown policy '{policy}' (iid_uniform | balanced_batch | fixed:k)")


def draw_condition(policy: str, rng: np.random.Generator) -> int:
    """Per-sample class for iid_uniform and fixed:k (balanced_batch cannot be per-sample)."""
    name, k = parse_policy(policy)
    if name == "iid_uniform":
        return int(rng.integers(C.NUM_CLASSES))
    if name == "fixed":
        return k
    raise ValueError("balanced_batch assigns classes per batch: use a BalancedBatchSampler")


class BalancedBatchSampler:
    """Batch sampler: each batch = B/4 items of every class, as (index, cond_id) tuples.

    Image indices come from a fresh permutation each epoch (every image appears once per
    epoch per slot). One epoch has n_items // batch_size batches (last partial batch dropped).
    The class labels inside a batch are shuffled so the class order carries no information.
    """

    def __init__(self, n_items: int, batch_size: int, seed: int = C.SEED):
        assert batch_size % C.NUM_CLASSES == 0, f"batch_size must be divisible by {C.NUM_CLASSES}"
        assert n_items >= batch_size, "fewer images than one batch"
        self.n_items, self.batch_size, self.seed = n_items, batch_size, seed
        self.epoch = 0  # incremented every time we are iterated, so epochs differ

    def __len__(self) -> int:
        return self.n_items // self.batch_size

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        self.epoch += 1
        order = rng.permutation(self.n_items)
        per_class = self.batch_size // C.NUM_CLASSES
        for b in range(len(self)):
            idx = order[b * self.batch_size:(b + 1) * self.batch_size]
            conds = rng.permutation(np.repeat(np.arange(C.NUM_CLASSES), per_class))
            yield [(int(i), int(c)) for i, c in zip(idx, conds)]


class PlainBatchSampler:
    """Shuffled batches of plain integer indices (for iid_uniform and fixed:k)."""

    def __init__(self, n_items: int, batch_size: int, seed: int = C.SEED, drop_last: bool = True):
        self.n_items, self.batch_size, self.seed, self.drop_last = n_items, batch_size, seed, drop_last
        self.epoch = 0

    def __len__(self) -> int:
        full, rest = divmod(self.n_items, self.batch_size)
        return full + (0 if self.drop_last or rest == 0 else 1)

    def __iter__(self):
        order = np.random.default_rng([self.seed, self.epoch]).permutation(self.n_items)
        self.epoch += 1
        for b in range(len(self)):
            yield [int(i) for i in order[b * self.batch_size:(b + 1) * self.batch_size]]


def make_batch_sampler(policy: str, n_items: int, batch_size: int, seed: int = C.SEED):
    """Pass the result to DataLoader(batch_sampler=...)."""
    name, _ = parse_policy(policy)
    if name == "balanced_batch":
        return BalancedBatchSampler(n_items, batch_size, seed)
    return PlainBatchSampler(n_items, batch_size, seed)


# Kept for the scaffold's original name.
make_sampler = make_batch_sampler
