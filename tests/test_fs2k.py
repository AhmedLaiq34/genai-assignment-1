"""FS2K data tests (CPU only). Fixture tests use a tiny fake download made by tests/fs2k_fixture.py;
the real-data tests are skipped when data/raw/fs2k is missing (or the real split was not made yet)."""
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from fs2k_fixture import build_mini_fs2k_data, gray, make_mini_fs2k
from PIL import Image

from genai.common.paths import ROOT
from genai.fs2k import dataset as ds
from genai.fs2k import pairs as pr
from genai.fs2k import prepare as prep
from genai.fs2k import split as sp

sys.path.insert(0, str(ROOT / "app" / "backend"))  # so `import app.preprocess` finds the backend
from app.preprocess import preprocess_bytes  # noqa: E402

REAL_RAW = ROOT / "data" / "raw" / "fs2k"
REAL_SPLIT = ROOT / "data" / "splits" / "fs2k_split.json"
needs_real = pytest.mark.skipif(not (REAL_RAW / "FS2K" / "anno_train.json").exists(), reason="real FS2K download missing")
needs_real_split = pytest.mark.skipif(not REAL_SPLIT.exists(), reason="run scripts/prepare_fs2k.py first")


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    return build_mini_fs2k_data(tmp_path_factory.mktemp("mini_fs2k"))


# ---------------------------------------------------------------- pairs (fixture)


def test_resolve_pairs_fixture(tmp_path):
    root = make_mini_fs2k(tmp_path / "FS2K", n_per_style=5, n_test_per_style=2)
    train = pr.resolve_pairs(root, "anno_train.json")
    test = pr.resolve_pairs(root, "anno_test.json")
    assert len(train) == 15 and len(test) == 6
    for p in train + test:
        assert (root / p["photo"]).is_file() and (root / p["sketch"]).is_file()
        assert p["id"] == p["photo"].split("/")[1] + "_" + Path(p["photo"]).stem  # photo1_image0001
    # sketch2 is a .png in the fake download, like the real one; the others are .jpg
    assert any(p["sketch"].endswith(".png") for p in train) and any(p["sketch"].endswith(".jpg") for p in train)


def test_resolve_accepts_parent_folder(tmp_path):
    make_mini_fs2k(tmp_path / "FS2K", n_per_style=2, n_test_per_style=1)
    assert len(pr.resolve_pairs(tmp_path, "anno_train.json")) == 6  # parent of the folder with the anno files


def test_missing_sketch_raises_and_lists_all(tmp_path):
    root = make_mini_fs2k(tmp_path / "FS2K", n_per_style=4, n_test_per_style=1)
    gone = sorted((root / "sketch").rglob("sketch*.*"))[:2]  # files only
    for g in gone:
        g.unlink()
    with pytest.raises(FileNotFoundError) as err:
        pr.resolve_pairs(root, "anno_train.json")
    text = str(err.value)
    assert "2 unresolved" in text
    assert all(f"image{g.stem[len('sketch'):]}" in text for g in gone)  # every missing entry is named


def test_style_values(mini):
    assert {p["style"] for p in mini["split"]["pairs"].values()} == {0, 1, 2}
    assert pr.style_of({"style": 2}) == 2
    for bad in (3, -1, "1", 1.0, True, None):
        with pytest.raises(ValueError):
            pr.style_of({"style": bad})
    with pytest.raises(ValueError):
        pr.style_of({"image_name": "photo1/image0001"})  # field missing


def test_pair_id():
    assert pr.pair_id({"image_name": "photo1/image0110"}) == "photo1_image0110"


# ---------------------------------------------------------------- split (fixture)


def test_split_fixture_properties(mini):
    split = mini["split"]
    train, val, test = set(split["train"]), set(split["val"]), set(split["test"])
    assert not (train & val) and not (train & test) and not (val & test)
    assert len(train) + len(val) == 42 and len(test) == 12 and len(val) == 7  # ceil(0.15 * 42)
    assert split["test_locked"] is True and split["seed"] == 42
    assert sp.load_split(mini["split_file"]) == split


def test_split_tamper_detected(mini, tmp_path):
    bad = json.loads(mini["split_file"].read_text())
    bad["val"][0] = bad["train"][0]  # break the val list
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        sp.load_split(tmp_path / "bad.json")


def test_split_order_independent(tmp_path):
    root = make_mini_fs2k(tmp_path / "FS2K")
    train, test = pr.resolve_pairs(root, "anno_train.json"), pr.resolve_pairs(root, "anno_test.json")
    a = sp.make_split(train, test)
    b = sp.make_split(list(reversed(train)), test)
    assert a["sha256"] == b["sha256"]


# ---------------------------------------------------------------- real data


@needs_real
def test_resolve_counts_real():
    train = pr.resolve_pairs(REAL_RAW, "anno_train.json")
    test = pr.resolve_pairs(REAL_RAW, "anno_test.json")
    assert (len(train), len(test)) == (1058, 1046)
    assert {p["style"] for p in train + test} == {0, 1, 2}
    assert len({p["id"] for p in train + test}) == 2104


@pytest.fixture(scope="module")
def real_split():
    train = pr.resolve_pairs(REAL_RAW, "anno_train.json")
    test = pr.resolve_pairs(REAL_RAW, "anno_test.json")
    return sp.make_split(train, test), train, test


@needs_real
def test_split_counts_real(real_split):
    split, train, _ = real_split
    assert (len(split["train"]), len(split["val"]), len(split["test"])) == (899, 159, 1046)
    for style in "012":
        n_style = sum(1 for p in train if str(p["style"]) == style)
        assert abs(split["style_counts"]["val"][style] - 0.15 * n_style) <= 1
        assert split["style_counts"]["train"][style] + split["style_counts"]["val"][style] == n_style


@needs_real
def test_split_deterministic_and_disjoint_real(real_split):
    split, train, test = real_split
    again = sp.make_split(list(reversed(train)), test)
    assert again["sha256"] == split["sha256"]
    sp.verify_split(split)  # hashes match and the three sets are disjoint


@needs_real
def test_load_photo_matches_backend_preprocess_real():
    pair = pr.resolve_pairs(REAL_RAW, "anno_test.json")[0]
    path = REAL_RAW / "FS2K" / pair["photo"]
    ours = prep.load_photo(path, 128).astype(np.float32).transpose(2, 0, 1) / 255.0
    theirs = preprocess_bytes(path.read_bytes())
    assert ours.shape == theirs.shape == (3, 128, 128)
    assert np.abs(ours - theirs).max() <= 1 / 255 + 1e-6


@needs_real_split
def test_real_dataset_ranges_and_shapes():
    for split, aug in (("train", False), ("train", True), ("val", False)):
        d = ds.FS2KDataset(split, "local", augment=aug, subset=6)
        for i in range(len(d)):
            photo, sketch, style, pid = d[i]
            assert photo.shape == (3, 128, 128) and sketch.shape == (1, 128, 128)
            assert photo.dtype == sketch.dtype == torch.float32 and style.dtype == torch.int64
            assert -1 <= photo.min() and photo.max() <= 1 and -1 <= sketch.min() and sketch.max() <= 1
    assert len(ds.FS2KDataset("train", "local")) == 899 and len(ds.FS2KDataset("val", "local")) == 159


# ---------------------------------------------------------------- load_photo / load_sketch (fixture)


def test_load_photo_matches_backend_preprocess_fixture(mini):
    path = mini["raw_root"] / next(iter(mini["split"]["pairs"].values()))["photo"]
    ours = prep.load_photo(path, 128).astype(np.float32).transpose(2, 0, 1) / 255.0
    theirs = preprocess_bytes(path.read_bytes())
    assert np.abs(ours - theirs).max() <= 1 / 255 + 1e-6


def test_load_shapes_and_dtypes(mini):
    info = next(iter(mini["split"]["pairs"].values()))
    photo = prep.load_photo(mini["raw_root"] / info["photo"], 143)
    sketch = prep.load_sketch(mini["raw_root"] / info["sketch"], 128)
    assert photo.shape == (143, 143, 3) and photo.dtype == np.uint8
    assert sketch.shape == (128, 128) and sketch.dtype == np.uint8


def test_cache_contents(mini):
    split, cache = mini["split"], mini["cache_root"]
    n128 = len(list((cache / "fs2k128" / "photo").glob("*.png")))
    n143 = len(list((cache / "fs2k143" / "photo").glob("*.png")))
    assert n128 == len(split["pairs"]) == 54 and n143 == len(split["train"]) + len(split["val"]) == 42
    assert not any((cache / "fs2k143" / "photo" / f"{i}.png").exists() for i in split["test"])
    img = ds._read_png(cache / "fs2k128" / "sketch" / f"{split['train'][0]}.png", 1)
    assert img.shape == (1, 128, 128)


# ---------------------------------------------------------------- dataset (fixture)


def test_paths_resolution_nested(mini, tmp_path):
    nested = tmp_path / "input" / "owner" / "fs2k-data"
    nested.mkdir(parents=True)
    shutil.copytree(mini["data_root"], nested / "data")  # one folder deeper, like Kaggle mounts it
    paths = ds.resolve_fs2k_paths(tmp_path / "input")
    assert paths["split_file"] == nested / "data" / "splits" / "fs2k_split.json"
    assert paths["cache128"] == nested / "data" / "cache" / "fs2k128"
    with pytest.raises(FileNotFoundError):
        ds.resolve_fs2k_paths(tmp_path / "empty")


def test_dataset_ranges_and_shapes(mini):
    for split, aug in (("train", False), ("train", True), ("val", False)):
        d = ds.FS2KDataset(split, mini["data_root"], augment=aug)
        assert len(d) == len(mini["split"][split])
        photo, sketch, style, pid = d[0]
        assert photo.shape == (3, 128, 128) and sketch.shape == (1, 128, 128)
        assert photo.dtype == sketch.dtype == torch.float32
        assert style.dtype == torch.int64 and int(style) in (0, 1, 2) and isinstance(pid, str)
        assert -1 <= photo.min() and photo.max() <= 1 and -1 <= sketch.min() and sketch.max() <= 1
    assert len(ds.FS2KDataset("train", mini["data_root"], augment=False, subset=5)) == 5


def test_augment_uses_143_cache_and_val_ignores_augment(mini):
    assert ds.FS2KDataset("train", mini["data_root"], augment=True).cache_dir.name == "fs2k143"
    assert ds.FS2KDataset("train", mini["data_root"], augment=False).cache_dir.name == "fs2k128"
    assert not ds.FS2KDataset("val", mini["data_root"], augment=True).augment


def test_default_collate_batches(mini):
    loader = torch.utils.data.DataLoader(ds.FS2KDataset("train", mini["data_root"], augment=True), batch_size=4)
    photo, sketch, style, pid = next(iter(loader))
    assert photo.shape == (4, 3, 128, 128) and sketch.shape == (4, 1, 128, 128) and style.dtype == torch.int64


def test_test_split_locked(mini, tmp_path, monkeypatch):
    log = tmp_path / "logs" / "test_access.log"
    monkeypatch.setattr(ds, "TEST_LOG_PATH", log)
    with pytest.raises(PermissionError):
        ds.FS2KDataset("test", mini["data_root"])
    assert not log.exists()  # a refused attempt writes nothing
    d = ds.FS2KDataset("test", mini["data_root"], final_test=True)
    assert len(d) == 12 and d[0][0].shape == (3, 128, 128)
    assert len(log.read_text().strip().splitlines()) == 1  # exactly one line
    assert "test split opened" in log.read_text()


def test_paired_augment_keeps_sketch_equal_gray_of_photo():
    torch.manual_seed(0)
    photo = torch.rand(3, 143, 143) * 2 - 1
    sketch = gray(photo)  # a sketch that is exactly the gray version of the photo
    crops, flips = set(), 0
    for _ in range(50):
        p, s = ds.paired_augment(photo, sketch, 128)
        assert p.shape == (3, 128, 128) and s.shape == (1, 128, 128)
        assert torch.allclose(gray(p), s, atol=1e-6)  # crop and flip hit both images the same way
        for top in range(16):  # find where the crop sits, to confirm the draws really vary
            for left in range(16):
                ref = photo[:, top:top + 128, left:left + 128]
                if torch.equal(p, ref):
                    crops.add((top, left))
                elif torch.equal(p, ref.flip(-1)):
                    crops.add((top, left))
                    flips += 1
    assert len(crops) > 5 and 5 < flips < 45  # position varies; flip happens about half the time


def test_dataset_augment_keeps_gray_relation(mini):
    """On the fixture the sketch is gray(photo) up to JPEG noise; augmentation must not break that."""
    d = ds.FS2KDataset("train", mini["data_root"], augment=True)
    for i in range(10):
        photo, sketch, _, _ = d[i % len(d)]
        assert (gray((photo + 1) / 2) - (sketch + 1) / 2).abs().max() < 0.12


# ---------------------------------------------------------------- audit grid, package, app samples


def test_audit_grid_written(mini, tmp_path):
    out = tmp_path / "fig" / "audit.png"
    prep.save_pair_audit_grid(mini["cache_root"], mini["split"], out, per_style=8)
    assert out.is_file() and out.stat().st_size > 5000


def test_package_excludes_test_images(mini, tmp_path):
    test_ids = set(mini["split"]["test"])
    with zipfile.ZipFile(prep.package(mini["data_root"], tmp_path / "a.zip")) as z:
        names = z.namelist()
    assert "splits/fs2k_split.json" in names and "CONTENTS.sha256" in names
    assert not any(Path(n).stem in test_ids for n in names)
    assert sum(n.startswith("cache/fs2k128/photo/") for n in names) == 42
    assert sum(n.startswith("cache/fs2k143/sketch/") for n in names) == 42
    with zipfile.ZipFile(prep.package(mini["data_root"], tmp_path / "b.zip", include_test=True)) as z:
        names = z.namelist()
    assert sum(n.startswith("cache/fs2k128/photo/") for n in names) == 54  # test photos now included
    assert not any("fs2k143" in n and Path(n).stem in test_ids for n in names)  # but never in the 143 cache


def test_export_app_samples_deterministic(mini, tmp_path):
    chosen = prep.export_app_samples(mini["cache_root"], mini["split"], tmp_path / "samples", n=6)
    assert len(chosen) == 6 and set(chosen) <= set(mini["split"]["test"])
    assert sorted(mini["split"]["pairs"][i]["style"] for i in chosen) == [0, 0, 1, 1, 2, 2]
    assert chosen == prep.export_app_samples(mini["cache_root"], mini["split"], tmp_path / "samples2", n=6)
    files = sorted((tmp_path / "samples").glob("*.png"))
    assert len(files) == 6 and np.asarray(Image.open(files[0])).shape == (128, 128, 3)
