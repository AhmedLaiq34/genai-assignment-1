"""Task 1 train / resume / evaluate / tune on a tiny SYNTHETIC dataset (no real data needed)."""
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

from genai.common.checkpoint import load_checkpoint
from genai.pets import dataset as pets_dataset
from genai.pets import manifests as mf
from genai.pets import split as sp
from genai.tasks.task1 import config as t1cfg
from genai.tasks.task1 import train as t1train
from genai.tasks.task1.evaluate import run_evaluation
from genai.tasks.task1.train import run_training
from genai.tasks.task1.tune import dry_run_overrides, run_study

N_TRAINVAL, N_TEST = 40, 8      # 32 train / 8 val images -> 4 steps per epoch at batch 8


def _fake_images(n, seed):
    """Smooth random pictures (low-res noise upsampled) so SSIM / blur behave sensibly."""
    rng = np.random.default_rng(seed)
    small = rng.random((n, 8, 8, 3))
    big = np.kron(small, np.ones((1, 16, 16, 1)))
    return (big * 255).astype(np.uint8)


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
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


def make_cfg(data_root, tmp_path, **train):
    cfg = t1cfg.load_config(None, "local")
    cfg.update(device="test", data_root=str(data_root), output_root=str(tmp_path / "out"),
               persist_root=str(tmp_path / "out"), num_workers=0)
    # the real config uses the conv latent (D40); these tiny tests pin the small dense model
    cfg["model"].update(latent="dense", depth=4, base_channels=8, bottleneck_dim=32, dropout=0.1)
    cfg["train"].update(batch_size=8, epochs=2, sample_every_epochs=1, **train)
    cfg["run"]["smoke"] = True
    return cfg


def read_rows(run_dir):
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]


def test_train_writes_all_files(data_root, tmp_path):
    cfg = make_cfg(data_root, tmp_path)
    run_dir = run_training(cfg)
    assert run_dir.name.endswith("_test_t1_smoke")                     # run id format + smoke suffix
    for name in ("config.yaml", "ckpt_last.pt", "ckpt_best.pt", "metrics.jsonl"):
        assert (run_dir / name).exists()
    assert len(list((run_dir / "samples").glob("*.png"))) == 2
    rows = read_rows(run_dir)
    assert [r["epoch"] for r in rows] == [1, 2] and rows[-1]["global_step"] == 8
    assert {"clean", "salt_pepper", "gaussian_blur", "occlusion"} <= set(rows[0]["per_condition"])
    ck = load_checkpoint(run_dir / "ckpt_last.pt")
    for key in ("model", "optimizer", "scheduler", "scaler", "epoch", "global_step", "best_metric", "config",
                "seed", "split_sha256", "manifest_sha256", "rng"):
        assert key in ck
    assert ck["epoch"] == 2 and ck["global_step"] == 8 and ck["split_sha256"] and ck["manifest_sha256"]


def test_on_checkpoint_called_and_best_tracks_j(data_root, tmp_path):
    seen = []
    run_dir = run_training(make_cfg(data_root, tmp_path), on_checkpoint=seen.append)
    assert run_dir / "ckpt_last.pt" in seen and run_dir / "ckpt_best.pt" in seen
    best = load_checkpoint(run_dir / "ckpt_best.pt")["best_metric"]
    assert best == pytest.approx(min(r["val_J"] for r in read_rows(run_dir)))


def test_resume_continues_steps_and_restores_rng(data_root, tmp_path, monkeypatch):
    cfg = make_cfg(data_root, tmp_path, pause_after_steps=6)          # stop in the middle of epoch 2
    run_dir = run_training(cfg)
    paused = load_checkpoint(run_dir / "ckpt_last.pt")
    assert paused["global_step"] == 6 and paused["epoch"] == 1 and paused["step_in_epoch"] == 2
    assert len(read_rows(run_dir)) == 1                                # only epoch 1 was validated so far

    restored = []
    real = t1train.restore_rng_state
    monkeypatch.setattr(t1train, "restore_rng_state", lambda s: (restored.append(set(s)), real(s)))
    cfg["train"]["pause_after_steps"] = None
    run_dir2 = run_training(cfg, resume=str(run_dir / "ckpt_last.pt"))
    assert run_dir2 == run_dir                                         # same run continues
    assert restored == [{"python", "numpy", "torch", "cuda"}]          # all RNG states restored
    final = load_checkpoint(run_dir / "ckpt_last.pt")
    assert final["global_step"] == 8 and final["epoch"] == 2 and final["step_in_epoch"] == 0
    assert [r["epoch"] for r in read_rows(run_dir)] == [1, 2]


def test_resume_auto_finds_latest_and_rejects_changed_config(data_root, tmp_path):
    cfg = make_cfg(data_root, tmp_path, pause_after_steps=2)
    run_dir = run_training(cfg)
    assert t1cfg.find_latest_checkpoint(cfg) == run_dir / "ckpt_last.pt"
    bad = json.loads(json.dumps(cfg))
    bad["train"]["lr"] = 0.5
    with pytest.raises(ValueError, match="lr"):
        run_training(bad, resume="auto")
    cfg["train"]["pause_after_steps"] = None
    assert run_training(cfg, resume="auto") == run_dir


def test_tbd_epochs_need_max_steps(data_root, tmp_path):
    cfg = make_cfg(data_root, tmp_path)
    cfg["train"]["epochs"] = "TBD_AFTER_BENCHMARK"
    with pytest.raises(ValueError, match="TBD"):
        run_training(cfg)
    cfg["train"]["max_steps"] = 3                                      # a step cap is enough
    run_dir = run_training(cfg)
    assert load_checkpoint(run_dir / "ckpt_last.pt")["global_step"] == 3


def test_evaluate_val_outputs_and_test_stays_locked(data_root, tmp_path, monkeypatch):
    log = tmp_path / "test_access.log"
    monkeypatch.setattr(pets_dataset, "TEST_LOG_PATH", log)
    cfg = make_cfg(data_root, tmp_path)
    run_dir = run_training(cfg)
    out = run_evaluation(cfg, str(run_dir / "ckpt_best.pt"))            # default: val
    assert out.name.endswith("_val") and not log.exists()               # test never opened
    df = pd.read_csv(out / "per_image.csv")
    assert len(df) == 32 and {"MAE", "SSIM", "PSNR"} <= set(df.columns)
    table = pd.read_csv(out / "table_cond_severity.csv")
    assert {"clean", "salt_pepper", "gaussian_blur", "occlusion", "overall"} <= set(table["cond"])
    sel = pd.read_csv(out / "selected.csv")
    assert (sel["group"] == "representative").sum() == 12 and (sel["group"] == "worst").sum() == 4
    assert (out / "representative_12.png").exists() and (out / "worst_4.png").exists()


def test_training_and_tuning_code_never_touch_the_test_split(data_root):
    for name in ("train.py", "tune.py"):
        text = (Path(t1train.__file__).parent / name).read_text(encoding="utf-8")
        assert "final_test=True" not in text and "pets_test_manifest" not in text
        assert 'split="test"' not in text
    with pytest.raises(PermissionError):
        pets_dataset.PetsManifestDataset(data_root / "manifests" / "pets_test_manifest.jsonl", "test", False, data_root)


def test_study_dry_run(data_root, tmp_path):
    cfg = dry_run_overrides(make_cfg(data_root, tmp_path), tmp_path / "dry")
    cfg["tuned_params"] = [dict(p) for p in cfg["tuned_params"]]
    for p in cfg["tuned_params"]:
        if p["name"] == "batch":
            p["choices"] = [4, 8]
        if p["name"] == "bottleneck_dim":
            p["choices"] = [32, 64]
    # the real config now holds the approved budgets (D17); the TBD guard is tested on a TBD copy (D37)
    cfg.update(n_trials="TBD_AFTER_BENCHMARK", epochs_per_trial="TBD_AFTER_BENCHMARK")
    with pytest.raises(ValueError, match="TBD"):                        # budgets stay TBD unless given
        run_study(cfg)
    cfg.update(n_trials=2, epochs_per_trial=1)
    out = run_study(cfg)
    assert cfg["study"] == "t1_universal_v2_dryrun" and str(tmp_path / "dry") in str(out)
    trials = pd.read_csv(out / "trials.csv")
    assert len(trials) == 2 and trials["state"].eq("COMPLETE").all()
    assert {"params_lr", "params_batch", "params_bottleneck_dim", "params_encoder_channels",
            "params_dropout", "params_alpha"} <= set(trials.columns)
    db = tmp_path / "dry" / "optuna" / "t1_universal_v2_dryrun.db"
    assert db.exists()
    run_study(cfg)                                                      # continues the same study: no new trials
    study = optuna.load_study(study_name=cfg["study"], storage=f"sqlite:///{db.as_posix()}")
    assert len(study.trials) == 2


def test_run_id_format():
    rid = t1cfg.make_run_id("local", "t1", smoke=True)
    assert rid.endswith("_local_t1_smoke") and len(rid.split("_")[0]) == 13
