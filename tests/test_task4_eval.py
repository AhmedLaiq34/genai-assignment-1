"""Task 4 evaluation on the mini-FS2K fixture with a random-weight generator (CPU, no real data).

Fixture: 14 train pairs per style -> after the 15% split 7 val pairs, 12 test pairs.
The official test log is redirected to a tmp path: artifacts/test_access.log is never written.
"""
import json

import pandas as pd
import pytest
from fs2k_fixture import build_mini_fs2k_data
from PIL import Image
from test_onnx_t4 import make_fixture_checkpoint

from genai.fs2k import dataset as fs2k_dataset
from genai.tasks.task4.evaluate import run_evaluation

FILES = {"per_image.csv", "by_style.csv", "summary.json", "results_grid.png", "failures_grid.png",
         "style_variations.png"}


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    return build_mini_fs2k_data(tmp_path_factory.mktemp("fs2k"))


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    return make_fixture_checkpoint(tmp_path_factory.mktemp("ckpt") / "ckpt_best.pt")


@pytest.fixture()
def test_log(tmp_path, monkeypatch):
    """Redirect the test-split access log to a tmp file (the real one must never be written)."""
    log = tmp_path / "logs" / "test_access.log"
    monkeypatch.setattr(fs2k_dataset, "TEST_LOG_PATH", log)
    return log


def make_cfg(mini, tmp_path):
    return {"data_root": str(mini["data_root"]), "output_root": str(tmp_path / "out")}


def test_val_evaluation_writes_all_files(mini, ckpt, tmp_path, test_log):
    out = run_evaluation(make_cfg(mini, tmp_path), ckpt)
    assert out.parent == tmp_path / "out" / "eval" / "task4" and out.name.endswith("_val")
    assert {p.name for p in out.iterdir()} == FILES
    for png in ("results_grid.png", "failures_grid.png", "style_variations.png"):
        img = Image.open(out / png)
        assert img.width > 128 and img.height > 128
    assert not test_log.exists()                            # evaluating val never opens the test split


def test_grid_sizes(mini, ckpt, tmp_path, test_log):
    out = run_evaluation(make_cfg(mini, tmp_path), ckpt)
    # a tile is 128 px plus 2 px padding (and 2 px at the outer edge)
    assert Image.open(out / "results_grid.png").size == (9 * 130 + 2, 8 * 130 + 2)       # 8 rows x 3 styles x 3 tiles
    assert Image.open(out / "style_variations.png").size == (4 * 130 + 2, 7 * 130 + 2)   # only 7 val photos exist
    assert Image.open(out / "failures_grid.png").size == (3 * 130 + 2, 7 * 130 + 2)


def test_numbers_are_consistent(mini, ckpt, tmp_path, test_log):
    out = run_evaluation(make_cfg(mini, tmp_path), ckpt)
    per = pd.read_csv(out / "per_image.csv")
    by = pd.read_csv(out / "by_style.csv")
    summary = json.loads((out / "summary.json").read_text())
    n_val = len(mini["split"]["val"])

    assert list(per.columns) == ["pair_id", "style", "l1", "ssim", "psnr"]
    assert len(per) == n_val and sorted(per["pair_id"]) == sorted(mini["split"]["val"])
    # every image was scored with its own style id (the one in the split file)
    assert all(per["style"] == [mini["split"]["pairs"][p]["style"] for p in per["pair_id"]])
    assert per["l1"].between(0, 1).all() and per["ssim"].between(-1, 1).all() and (per["psnr"] > 0).all()

    assert by["count"].sum() == n_val and list(by["style"]) == [0, 1, 2]
    for _, row in by.iterrows():
        sub = per[per["style"] == row["style"]]
        assert row["count"] == len(sub)
        assert row["l1"] == pytest.approx(sub["l1"].mean()) and row["psnr"] == pytest.approx(sub["psnr"].mean())

    assert summary["split"] == "val" and summary["count"] == n_val and summary["checkpoint"] == str(ckpt)
    for k in ("l1", "ssim", "psnr"):
        assert summary["overall"][k] == pytest.approx(per[k].mean())
        assert summary["by_style"]["1"][k] == pytest.approx(by.loc[by["style"] == 1, k].item())
    assert sum(v["count"] for v in summary["by_style"].values()) == n_val


def test_final_test_uses_the_test_split_and_logs_one_access(mini, ckpt, tmp_path, test_log):
    out = run_evaluation(make_cfg(mini, tmp_path), ckpt, final_test=True)
    assert out.name.endswith("_test") and {p.name for p in out.iterdir()} == FILES
    per = pd.read_csv(out / "per_image.csv")
    assert sorted(per["pair_id"]) == sorted(mini["split"]["test"]) and len(per) == 12
    assert json.loads((out / "summary.json").read_text())["split"] == "test"
    lines = test_log.read_text().strip().splitlines()
    assert len(lines) == 1 and "test split opened" in lines[0]


def test_val_runs_never_touch_the_log(mini, ckpt, tmp_path, test_log):
    run_evaluation(make_cfg(mini, tmp_path), ckpt)
    run_evaluation(make_cfg(mini, tmp_path), ckpt, final_test=False)
    assert not test_log.exists()


def test_eval_batch_size_does_not_change_the_numbers(mini, ckpt, tmp_path, test_log):
    cfg = make_cfg(mini, tmp_path)
    a = pd.read_csv(run_evaluation(cfg, ckpt) / "per_image.csv")
    b = pd.read_csv(run_evaluation({**cfg, "eval": {"batch_size": 2}}, ckpt) / "per_image.csv")
    assert a["pair_id"].tolist() == b["pair_id"].tolist()
    assert a["l1"].tolist() == pytest.approx(b["l1"].tolist(), abs=1e-5)
