"""Task 4 Optuna study: dry run on the mini-FS2K fixture, restart behaviour, and the study rebuilt from the
Colab console log (no real data, CPU only). Plan: docs/TASK4_PLAN.md B.1, B.2, E.

Nothing here touches the real artifacts/optuna or studies/task4_cgan: everything goes to pytest tmp folders.
"""
import csv
import gc
import json
import logging
from pathlib import Path

import optuna
import pytest
import yaml
from fs2k_fixture import build_mini_fs2k_data

from genai.common.paths import ROOT
from genai.tasks.task4 import config as t4cfg
from genai.tasks.task4 import log_study as ls
from genai.tasks.task4 import tune as t4tune

CONFIG = ROOT / "configs" / "task4_cgan.yaml"
CSV = ROOT / "studies" / "task4_cgan" / "trials_from_console_log.csv"

# The real search space has base_channels 32/64, which is slow on CPU: the tests use tiny networks.
TINY_SPACE = [
    {"name": "lr_g", "type": "float", "low": 1.0e-4, "high": 4.0e-4, "log": True},
    {"name": "lr_d", "type": "float", "low": 1.0e-4, "high": 4.0e-4, "log": True},
    {"name": "batch_size", "type": "categorical", "choices": [4]},
    {"name": "base_channels", "type": "categorical", "choices": [8]},
    {"name": "dropout", "type": "float", "low": 0.0, "high": 0.5, "log": False},
    {"name": "style_dim", "type": "categorical", "choices": [4]},
    {"name": "lambda_l1", "type": "float", "low": 10.0, "high": 200.0, "log": True},
]


@pytest.fixture(scope="module")
def fs2k(tmp_path_factory):
    return build_mini_fs2k_data(tmp_path_factory.mktemp("fs2k"))


def make_cfg(fs2k, tmp_path, n_trials=2):
    """Task 4 study config with tiny trials: 1 epoch, 16 train pairs, tiny networks, tmp folders."""
    cfg = t4cfg.load_config("configs/task4_cgan.yaml", "local")
    cfg = t4tune.dry_run_overrides(cfg, tmp_path)
    cfg.update(device="test", data_root=str(fs2k["data_root"]), num_workers=0,
               n_trials=n_trials, epochs_per_trial=1, trial_train_subset=16, tuned_params=TINY_SPACE)
    cfg["model"].update(base_channels=8, style_dim=4)
    cfg["train"].update(batch_size=4, val_subset=6)
    return cfg


@pytest.fixture
def optuna_logs(caplog):
    """Optuna logs through its own handler (stderr) and does not propagate; let caplog see the lines."""
    optuna.logging.enable_propagation()
    caplog.set_level(logging.INFO, logger="optuna")
    yield caplog
    optuna.logging.disable_propagation()


# ------------------------------------------------------------------------------- the study itself
def test_study_dry_run_and_rerun(fs2k, tmp_path, capsys, optuna_logs):
    cfg = make_cfg(fs2k, tmp_path)
    out_dir = t4tune.run_study(cfg)

    # files: DB, trials.csv, best_params.json (and the persistent copy of the DB)
    db = Path(cfg["storage"])
    assert db.exists() and (tmp_path / "optuna" / db.name).exists()
    assert (out_dir / "trials.csv").exists()
    best = json.loads((out_dir / "best_params.json").read_text())
    assert best["study"] == "task4_cgan_dryrun" and best["objective"] == "val L1 on [0,1] sketches"
    assert sum(best["n_trials"].values()) == 2 and best["n_trials"]["COMPLETE"] == 2
    assert set(best["params"]) == {s["name"] for s in TINY_SPACE}

    # the five-line block of plan B.1, exactly
    study = optuna.load_study(study_name=cfg["study"], storage=f"sqlite:///{db.as_posix()}")
    printed = capsys.readouterr().out.splitlines()
    block = printed[printed.index("=== Task 4 Optuna study complete ==="):][:5]
    assert block == ["=== Task 4 Optuna study complete ===",
                     "Trials run: 2",
                     f"Best trial: #{study.best_trial.number}  best val L1: {study.best_value:.4f}",
                     f"Best params: {study.best_params}",
                     f"Saved to {(out_dir / 'best_params.json').as_posix()}"]
    assert any(line.startswith("[trial 0]") for line in printed)          # our own prefixed lines
    assert "A new study created in RDB with name: task4_cgan_dryrun" in optuna_logs.text
    assert "Trial 0 finished with value" in optuna_logs.text

    # trial user attributes, and the trial run went to runs/task4_trials (never runs/task4)
    attrs = study.trials[0].user_attrs
    assert {"run_id", "best_epoch", "val_ssim_at_best", "seconds"} <= set(attrs)
    assert (tmp_path / "runs" / "task4_trials" / attrs["run_id"] / "metrics.jsonl").exists()
    assert not (tmp_path / "runs" / "task4").exists()

    # rerun with the same budget: the existing study is used and no new trial is run
    optuna_logs.clear()
    t4tune.run_study(cfg)
    assert "Using an existing study with `study_name='task4_cgan_dryrun'` instead of creating a new one." in optuna_logs.text
    assert len(optuna.load_study(study_name=cfg["study"], storage=f"sqlite:///{db.as_posix()}").trials) == 2


def test_running_trial_marked_fail_and_not_counted(fs2k, tmp_path):
    cfg = make_cfg(fs2k, tmp_path, n_trials=1)
    db = Path(cfg["storage"])
    db.parent.mkdir(parents=True)
    storage = f"sqlite:///{db.as_posix()}"
    study = optuna.create_study(study_name=cfg["study"], storage=storage, direction="minimize")
    study.ask()                                    # a trial left RUNNING by a "killed" process
    assert study.trials[0].state == optuna.trial.TrialState.RUNNING

    t4tune.run_study(cfg)
    trials = optuna.load_study(study_name=cfg["study"], storage=storage).trials
    assert [t.state.name for t in trials] == ["FAIL", "COMPLETE"]   # the FAIL did not use up the budget of 1


def test_make_trial_config(fs2k, tmp_path):
    cfg = make_cfg(fs2k, tmp_path)
    values = {"lr_g": 1e-4, "lr_d": 2e-4, "batch_size": 8, "base_channels": 32, "dropout": 0.1,
              "style_dim": 16, "lambda_l1": 50.0}
    tcfg = t4tune.make_trial_config(cfg, values, 3)
    assert tcfg["run"]["trial"] is True and tcfg["run_id"].endswith("task4_cgan_dryrun-trial3_smoke")
    assert (tcfg["train"]["lr_g"], tcfg["train"]["lr_d"], tcfg["train"]["batch_size"], tcfg["train"]["lambda_l1"]) \
        == (1e-4, 2e-4, 8, 50.0)
    assert (tcfg["model"]["base_channels"], tcfg["model"]["style_dim"], tcfg["model"]["dropout"]) == (32, 16, 0.1)
    assert tcfg["train"]["epochs"] == 1 and tcfg["train"]["train_subset"] == 16
    assert tcfg["train"]["sample_every_epochs"] == 0 and tcfg["train"]["snapshot_every_epochs"] == 0
    assert tcfg["train"]["checkpoint_every_minutes"] > 1e6
    assert "trial" not in cfg["run"]               # the base config is not modified


def test_tbd_budget_is_refused(fs2k, tmp_path):
    cfg = make_cfg(fs2k, tmp_path)
    cfg["n_trials"] = "TBD_AFTER_BENCHMARK"
    with pytest.raises(ValueError, match="TBD_AFTER_BENCHMARK"):
        t4tune.run_study(cfg)


def test_dry_run_overrides_keep_real_files_safe(tmp_path):
    cfg = t4cfg.load_config("configs/task4_cgan.yaml", "local")
    dry = t4tune.dry_run_overrides(cfg, tmp_path)
    assert dry["study"] == "task4_cgan_dryrun" and dry["run"]["smoke"] is True
    for key in ("storage", "study_dir", "output_root", "persist_root"):
        assert str(tmp_path) in str(dry[key])
    assert cfg["storage"] == "artifacts/optuna/task4_cgan.db"      # the input config is unchanged


# ------------------------------------------------------------------ study rebuilt from the console log
def build(tmp_path, **kw):
    return ls.build_log_study(CSV, tmp_path / "task4_cgan.db", CONFIG, **kw)


def test_build_log_study_counts_and_best(tmp_path):
    study = build(tmp_path)
    states = [t.state.name for t in study.trials]
    assert len(states) == 26
    assert (states.count("COMPLETE"), states.count("PRUNED"), states.count("FAIL")) == (14, 6, 6)
    assert [t.number for t in study.trials] == list(range(26))
    assert all(t.state.name == "FAIL" and "filename bug" in t.user_attrs["note"] for t in study.trials[:6])
    assert study.best_trial.number == 25 and study.best_value == 0.0928155106318572
    row25 = next(r for r in csv.DictReader(open(CSV, encoding="utf-8")) if r["trial"] == "25")
    assert study.best_params == {"lr_g": float(row25["lr_g"]), "lr_d": float(row25["lr_d"]),
                                 "batch_size": 8, "base_channels": 64, "dropout": float(row25["dropout"]),
                                 "style_dim": 32, "lambda_l1": float(row25["lambda_l1"])}
    assert study.user_attrs["source"] == "rebuilt from the Colab console log; original DB lost"
    assert "original Colab ranges are unknown" in study.user_attrs["search_ranges"]
    assert study.trials[25].user_attrs["original_trial_number"] == 25
    # it is stored: a fresh load of the file gives the same study
    reloaded = optuna.load_study(study_name="task4_cgan", storage=f"sqlite:///{(tmp_path / 'task4_cgan.db').as_posix()}")
    assert len(reloaded.trials) == 26 and reloaded.best_value == 0.0928155106318572


def test_build_log_study_refuses_existing_db_unless_overwrite(tmp_path):
    study = build(tmp_path)
    with pytest.raises(FileExistsError):
        build(tmp_path)
    assert len(optuna.load_study(study_name="task4_cgan",
                                 storage=f"sqlite:///{(tmp_path / 'task4_cgan.db').as_posix()}").trials) == 26
    del study                                      # release the SQLite file (Windows cannot delete an open file)
    gc.collect()
    assert len(build(tmp_path, overwrite=True).trials) == 26      # rebuilt, not appended (no 52)


def test_log_study_main_writes_exports(tmp_path, capsys):
    ls.main(["--out-db", str(tmp_path / "study" / "task4_cgan.db")])
    out = capsys.readouterr().out
    assert "(study rebuilt from the Colab console log, not produced by tune.py)" in out
    assert "Trials run: 26" in out and "Best trial: #25  best val L1: 0.0928" in out
    folder = tmp_path / "study"
    assert (folder / "trials.csv").exists()
    best = json.loads((folder / "best_params.json").read_text())
    assert best["n_trials"] == {"COMPLETE": 14, "PRUNED": 6, "FAIL": 6}
    assert best["source"] == "rebuilt from the Colab console log; original DB lost"


def test_csv_values_lie_inside_the_tuned_ranges():
    space = {s["name"]: s for s in yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["tuned_params"]}
    rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8")) if r["state"] == "COMPLETE"]
    assert len(rows) == 14
    for row in rows:
        for name, spec in space.items():
            value = ls.parse_value(spec, row[name])
            if spec["type"] == "categorical":
                assert value in spec["choices"], (row["trial"], name, value)
            else:
                assert spec["low"] <= value <= spec["high"], (row["trial"], name, value)
