"""Tests for genai.common.checkpoint (round-trip, RNG state, atomicity, promote)."""
import json
import os
import random

import numpy as np
import pytest
import torch

from genai.common import checkpoint as ck


def _state():
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.Adam(model.parameters())
    return ck.build_checkpoint(model, opt, epoch=3, global_step=30, best_metric=0.1,
                               config={"lr": 1e-3}, seed=42), model


def test_round_trip_has_all_keys(tmp_path):
    state, model = _state()
    state["rng"] = ck.capture_rng_state()
    p = tmp_path / "ckpt_last.pt"
    ck.save_checkpoint(p, state)
    back = ck.load_checkpoint(p)
    for k in ck.CHECKPOINT_KEYS:
        assert k in back
    assert back["epoch"] == 3 and back["config"] == {"lr": 1e-3}
    assert torch.equal(back["model"]["weight"], model.weight)


def test_rng_restore_gives_same_numbers(tmp_path):
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    p = tmp_path / "rng.pt"
    ck.save_checkpoint(p, {"rng": ck.capture_rng_state()})
    first = (random.random(), np.random.rand(3), torch.rand(3))
    # scramble the generators, then restore from the loaded file
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    ck.restore_rng_state(ck.load_checkpoint(p)["rng"])
    second = (random.random(), np.random.rand(3), torch.rand(3))
    assert first[0] == second[0]
    assert np.array_equal(first[1], second[1])
    assert torch.equal(first[2], second[2])


def test_atomic_save_leaves_no_temp_files(tmp_path):
    ck.save_checkpoint(tmp_path / "a.pt", {"x": 1})
    assert [f.name for f in tmp_path.iterdir()] == ["a.pt"]


def test_failed_save_keeps_old_file_and_no_temp(tmp_path):
    p = tmp_path / "a.pt"
    ck.save_checkpoint(p, {"x": 1})

    class Bomb:  # pickling this raises in the middle of the write
        def __reduce__(self):
            raise RuntimeError("boom")

    with pytest.raises(Exception):
        ck.save_checkpoint(p, {"x": 2, "bomb": Bomb()})
    assert ck.load_checkpoint(p)["x"] == 1                   # old checkpoint intact
    assert [f.name for f in tmp_path.iterdir()] == ["a.pt"]  # no temp left behind


def test_sha256_known_value(tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"abc")
    assert ck.sha256_file(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_git_commit_tolerates_missing_repo(tmp_path):
    out = ck.git_commit(cwd=tmp_path)  # not a repo -> None (or a hash if inside one)
    assert out is None or len(out) == 40


def test_promote_writes_only_to_given_root(tmp_path):
    src = tmp_path / "src.pt"
    ck.save_checkpoint(src, {"x": 1})
    models = tmp_path / "models"
    dest = ck.promote(src, "t1_universal_ae", root=models)
    assert dest == models / "checkpoints" / "t1_universal_ae.pt"
    manifest = json.loads((models / "MANIFEST.json").read_text())
    assert manifest[0]["name"] == "t1_universal_ae"
    assert manifest[0]["sha256"] == ck.sha256_file(dest)
    assert manifest[0]["size"] == os.path.getsize(dest)
    ck.promote(src, "t2_classifier", root=models)
    ck.promote(src, "t1_universal_ae", root=models)  # re-promote replaces, no duplicate
    names = [e["name"] for e in json.loads((models / "MANIFEST.json").read_text())]
    assert sorted(names) == ["t1_universal_ae", "t2_classifier"]
