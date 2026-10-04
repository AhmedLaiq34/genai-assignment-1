"""Dataset tests. Use the REAL cache/splits/manifests produced by scripts/prepare_pets.py."""
from collections import Counter

import pytest
import torch
from torch.utils.data import DataLoader

from genai.common.seed import worker_init_fn
from genai.pets import dataset as ds
from genai.pets import samplers as sm

PATHS = ds.resolve_data_paths("local")
VAL_MANIFEST = PATHS["manifests"] / "pets_val_manifest.jsonl"
TEST_MANIFEST = PATHS["manifests"] / "pets_test_manifest.jsonl"

pytestmark = pytest.mark.skipif(not VAL_MANIFEST.exists(), reason="run scripts/prepare_pets.py first")


def _check_item(item):
    corrupted, clean, cond, sev = item
    for t in (corrupted, clean):
        assert t.shape == (3, 128, 128) and t.dtype == torch.float32
        assert 0.0 <= t.min() and t.max() <= 1.0
    assert cond in (0, 1, 2, 3) and sev in (-1, 0, 1, 2)
    assert (sev == -1) == (cond == 0)


def test_train_item_format_and_two_loads_differ():
    d = ds.PetsTrainDataset("train", "fixed:1", data_root="local")
    a, b = d[5], d[5]
    _check_item(a)
    assert torch.equal(a[1], b[1])  # same clean image
    assert not torch.equal(a[0], b[0])  # different salt-and-pepper noise


def test_train_iid_uses_all_classes():
    d = ds.PetsTrainDataset("train", "iid_uniform", data_root="local")
    assert {d[i][2] for i in range(80)} == {0, 1, 2, 3}


def test_train_fixed_k_only_k():
    d = ds.PetsTrainDataset("train", "fixed:3", data_root="local")
    assert {d[i][2] for i in range(30)} == {3}


def test_train_balanced_needs_sampler_and_dataloader_gets_exact_counts():
    d = ds.PetsTrainDataset("train", "balanced_batch", data_root="local")
    with pytest.raises(ValueError):
        d[0]  # no class given
    bs = sm.make_batch_sampler("balanced_batch", len(d), 16)
    loader = DataLoader(d, batch_sampler=bs, num_workers=2, worker_init_fn=worker_init_fn)
    for n, (x, y, cond, sev) in enumerate(loader):
        assert Counter(cond.tolist()) == {0: 4, 1: 4, 2: 4, 3: 4}
        assert x.shape == (16, 3, 128, 128)
        if n == 2:
            break


def test_manifest_dataset_deterministic_and_worker_safe():
    d = ds.PetsManifestDataset(VAL_MANIFEST, "val", data_root="local")
    _check_item(d[3])
    first = [d[i][0] for i in range(12)]
    assert all(torch.equal(first[i], d[i][0]) for i in range(12))  # two loads identical
    loader = DataLoader(torch.utils.data.Subset(d, range(12)), batch_size=4, num_workers=2)
    via_workers = torch.cat([b[0] for b in loader])
    assert torch.equal(via_workers, torch.stack(first))


def test_test_split_locked_and_logged(tmp_path):
    with pytest.raises(PermissionError):
        ds.PetsManifestDataset(TEST_MANIFEST, "test", data_root="local")
    log = tmp_path / "access.log"
    d = ds.PetsManifestDataset(TEST_MANIFEST, "test", final_test=True, data_root="local", test_log_path=log)
    assert len(log.read_text().splitlines()) == 1
    _check_item(d[0])
    assert len(d) == 10 * len(d.split["test"]["ids"])


def test_tampered_manifest_fails(tmp_path):
    bad = tmp_path / "m.jsonl"
    bad.write_bytes(VAL_MANIFEST.read_bytes() + b"\n")
    (tmp_path / "m.jsonl.sha256").write_text((VAL_MANIFEST.parent / "pets_val_manifest.jsonl.sha256").read_text())
    with pytest.raises(ValueError):
        ds.PetsManifestDataset(bad, "val", data_root="local")


def test_find_dir_finds_a_dataset_nested_deeper_in_a_kaggle_style_input_folder(tmp_path):
    """/kaggle/input/datasets/<owner>/<name>/data/cache/pets128 must be found from /kaggle/input."""
    from genai.pets.dataset import _find_dir
    deep = tmp_path / "datasets" / "someone" / "pets-data" / "data" / "cache" / "pets128"
    deep.mkdir(parents=True)
    (tmp_path / "code-dataset" / "src").mkdir(parents=True)          # another dataset next to it
    assert _find_dir(tmp_path, "cache", "pets128") == deep
    (deep.parent / "splits").mkdir()
    assert _find_dir(tmp_path, "splits") == deep.parent / "splits"
