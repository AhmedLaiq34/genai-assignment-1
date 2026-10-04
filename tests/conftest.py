"""Shared pytest fixture for the Task 2 tests: a tiny SYNTHETIC pets dataset (no real data needed).

`tiny_pets_root` builds, in a temp folder, the same files the real pipeline makes:
data/cache/pets128/{trainval,test}_{images.npy,ids.json}, splits/pets_split.json and the val / test
manifests. 80 trainval images -> 64 train / 16 val images -> 64 val manifest rows (16 per condition).
"""
import json

import numpy as np
import pytest

from genai.pets import manifests as mf
from genai.pets import split as sp

N_TRAINVAL, N_TEST = 80, 8


def _fake_images(n, seed):
    """Smooth random pictures (low-res noise upsampled) so SSIM / blur behave sensibly."""
    rng = np.random.default_rng(seed)
    small = rng.random((n, 8, 8, 3))
    big = np.kron(small, np.ones((1, 16, 16, 1)))
    return (big * 255).astype(np.uint8)


@pytest.fixture(scope="session")
def tiny_pets_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny_pets")
    tv_ids = [f"img_{i:03d}" for i in range(N_TRAINVAL)]
    test_ids = [f"test_{i:03d}" for i in range(N_TEST)]
    cache = root / "cache" / "pets128"
    cache.mkdir(parents=True)
    for part, ids, seed in (("trainval", tv_ids, 1), ("test", test_ids, 2)):
        np.save(cache / f"{part}_images.npy", _fake_images(len(ids), seed))
        (cache / f"{part}_ids.json").write_text(json.dumps(ids))
    split = sp.make_split(tv_ids, test_ids)
    sp.write_split(split, root / "splits" / "pets_split.json")
    mf.build_val_manifest(split, root / "manifests" / "pets_val_manifest.jsonl")
    mf.build_test_manifest(split, root / "manifests" / "pets_test_manifest.jsonl")
    return root
