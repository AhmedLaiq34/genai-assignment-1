import json

import numpy as np
import pytest
import torch

from genai.pets import manifests as mf
from genai.pets import split as sp


@pytest.fixture(scope="module")
def split():
    return sp.make_split([f"tv_{i}" for i in range(50)], [f"te_{i}" for i in range(6)])


def _img():
    return torch.from_numpy(np.random.default_rng(3).random((3, 128, 128), dtype=np.float32))


def test_row_counts(split):
    val = mf.make_val_rows(split)
    test = mf.make_test_rows(split)
    assert len(val) == 4 * len(split["val"])
    assert len(test) == 10 * len(split["test"]["ids"])
    first = [r for r in test if r["image_id"] == "te_0"]
    assert [(r["cond_id"], r["severity"]) for r in first] == (
        [(0, None)] + [(c, s) for c in (1, 2, 3) for s in ("low", "medium", "high")])


def test_row_schema(split):
    keys = {"image_id", "split", "cond_id", "cond_name", "severity", "params", "rects",
            "achieved_coverage", "seed", "manifest_version"}
    for r in mf.make_val_rows(split) + mf.make_test_rows(split):
        assert set(r) == keys
        json.dumps(r)  # JSON serialisable
        if r["cond_id"] == 3:
            assert r["rects"] and r["achieved_coverage"] is not None
        else:
            assert r["rects"] == [] and r["achieved_coverage"] is None


def test_rows_deterministic_and_seeds_differ(split):
    assert mf.make_val_rows(split) == mf.make_val_rows(split)
    assert mf.row_seed("val", "img", 1) == mf.row_seed("val", "img", 1)
    assert mf.row_seed("val", "img", 1) != mf.row_seed("val", "img", 2)


def test_byte_identical_regeneration(split):
    img = _img()
    for row in mf.make_val_rows(split)[:40] + mf.make_test_rows(split)[:30]:
        a, b = mf.render(row, img), mf.render(row, img)
        assert a.numpy().tobytes() == b.numpy().tobytes()


def test_render_after_file_roundtrip(split, tmp_path):
    path = tmp_path / "test.jsonl"
    mf.build_test_manifest(split, path)
    rows = mf.load_manifest(path)
    img = _img()
    for old, new in zip(mf.make_test_rows(split), rows):
        assert mf.render(old, img).numpy().tobytes() == mf.render(new, img).numpy().tobytes()


def test_val_params_in_training_ranges(split):
    for r in mf.make_val_rows(split):
        p = r["params"]
        if r["cond_id"] == 1:
            assert 0.02 <= p["p"] <= 0.15
        if r["cond_id"] == 2:
            assert 0.5 <= p["sigma"] <= 2.5
        if r["cond_id"] == 3:
            assert 0.10 <= r["achieved_coverage"] <= 0.35
        assert (r["severity"] is None) == (r["cond_id"] == 0)


def test_sha_file_and_tamper_detection(split, tmp_path):
    path = tmp_path / "val.jsonl"
    digest = mf.build_val_manifest(split, path)
    assert (tmp_path / "val.jsonl.sha256").read_text().strip() == digest
    assert len(mf.load_manifest(path)) == 4 * len(split["val"])
    path.write_bytes(path.read_bytes().replace(b'"val"', b'"vaX"', 1))
    with pytest.raises(ValueError):
        mf.load_manifest(path)
