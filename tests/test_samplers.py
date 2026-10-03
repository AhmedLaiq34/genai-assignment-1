"""Sampler tests (CONTRACTS 3.5). Pure python/numpy, no dataset needed."""
from collections import Counter

import numpy as np
import pytest

from genai.pets import samplers as sm


def test_balanced_batches_have_exact_class_counts():
    bs = sm.BalancedBatchSampler(n_items=1000, batch_size=32, seed=1)
    n = 0
    while n < 100:  # 100 batches, spanning several epochs
        for batch in bs:
            counts = Counter(c for _, c in batch)
            assert counts == {0: 8, 1: 8, 2: 8, 3: 8}
            n += 1
            if n == 100:
                break


def test_balanced_requires_divisible_batch():
    with pytest.raises(AssertionError):
        sm.BalancedBatchSampler(100, 30)


def test_balanced_epochs_differ_and_no_repeat_inside_epoch():
    bs = sm.BalancedBatchSampler(64, 16, seed=3)
    e1 = [i for b in bs for i, _ in b]
    e2 = [i for b in bs for i, _ in b]
    assert len(set(e1)) == len(e1) and e1 != e2


def test_fixed_policy_only_class_k():
    rng = np.random.default_rng(0)
    for k in range(4):
        assert {sm.draw_condition(f"fixed:{k}", rng) for _ in range(200)} == {k}


def test_iid_uniform_covers_all_classes_roughly_evenly():
    rng = np.random.default_rng(0)
    counts = Counter(sm.draw_condition("iid_uniform", rng) for _ in range(4000))
    assert set(counts) == {0, 1, 2, 3} and all(800 < v < 1200 for v in counts.values())


def test_bad_policies_rejected():
    for bad in ("fixed:4", "fixed:x", "nonsense"):
        with pytest.raises(ValueError):
            sm.parse_policy(bad)
    with pytest.raises(ValueError):
        sm.draw_condition("balanced_batch", np.random.default_rng(0))


def test_make_batch_sampler_types():
    assert isinstance(sm.make_batch_sampler("balanced_batch", 100, 8), sm.BalancedBatchSampler)
    plain = sm.make_batch_sampler("fixed:2", 100, 8)
    batch = next(iter(plain))
    assert len(batch) == 8 and all(isinstance(i, int) for i in batch)
