"""Task 3 Optuna study (t3_moe) on the tiny SYNTHETIC dataset of tests/conftest.py and the RANDOM fixture source
checkpoints of tests/t3_fixtures.py. CPU only (run with CUDA_VISIBLE_DEVICES=-1).

Covers: the 2-trial dry run (DB, trials.csv, user attributes, trial 0 = PDF start, every trial reloads the Task 2
weights, trial runs kept apart, no checkpoints left), collapse pruning, median pruning, persistence + timeout, the TBD
guard, search-space checks, make_trial_config, write_final_config and the structure of configs/task3_moe.yaml.
Every test pins its own tiny budget; only test_real_config_* read the real YAML (structure, never its budgets).
"""
import json
from pathlib import Path

import optuna
import pandas as pd
import pytest
import yaml

from genai.models.moe import SoftMoE, parameter_hash
from genai.tasks.task1.config import load_config
from genai.tasks.task1.tune import suggest_params
from genai.tasks.task3 import train as tr
from genai.tasks.task3 import tune
from genai.tasks.task3.tune import (PARAM_TARGETS, PDF_START, dry_run_overrides, make_trial_config, run_study,
                                    write_final_config)
from genai.tasks.task3.train import run_training
from t3_fixtures import make_t3_sources

REAL_THRESHOLDS = {"min_mean_weight": 0.02, "max_foreign_weight": 0.9}
TUNED = [
    {"name": "joint_lr", "type": "float", "low": 1e-5, "high": 2e-4, "log": True},
    {"name": "tau", "type": "float", "low": 0.5, "high": 5.0, "log": True},
    {"name": "lambda_c", "type": "float", "low": 0.01, "high": 1.0, "log": True},
    {"name": "lambda_b", "type": "float", "low": 0.001, "high": 0.1, "log": True},
    {"name": "recon_l1_share", "type": "float", "low": 0.3, "high": 0.95, "log": False},
]
GOOD_FLAGS = {"collapsed": False, "reasons": [], "mean_w": [0.25] * 4, "mean_w_by_class": [[0.25] * 4] * 4}


def make_cfg(root, tmp_path, paths, sha):
    """A full Task 3 study config shrunk to a tiny CPU study: 32 train images (batch 8 -> 4 steps per epoch),
    16 validation rows, 2 trials of 1 warm-up + 1 joint epoch. Folders are those of a dry run under tmp_path."""
    cfg = {
        "task": "t3", "component": "soft_moe", "seed": 42, "device": "test", "device_profile": "local",
        "data_root": str(root), "output_root": str(tmp_path / "out"), "persist_root": str(tmp_path / "out"),
        "num_workers": 0,
        "run": {"desc": "t3moe", "smoke": False, "project": "genai-a1"},
        "sources": {"dir": str(Path(paths["classifier"]).parent), "expected_sha256": sha},
        "model": {"tau": 1.0},
        "train": {"batch_size": 8, "warmup_epochs": 1, "joint_epochs": 1, "warmup_lr": 2e-4, "joint_lr": 5e-5,
                  "weight_decay": 1e-4, "lambda_1": 0.8, "lambda_s": 0.2, "lambda_c": 0.1, "lambda_b": 0.01,
                  "amp": False, "scheduler": "cosine", "grad_clip": None, "train_subset": None, "val_subset": 16,
                  "val_every_epochs": 1, "val_batch_size": 16, "val_workers": 0, "epoch0_validation": True,
                  "checkpoint_every_minutes": 10, "log_every_steps": 5, "sample_every_epochs": 1,
                  "max_steps": None, "pause_after_steps": None},
        "collapse": {"min_mean_weight": 0.0, "max_foreign_weight": 1.0},       # never flags (unless a test says so)
        "study": "t3_moe", "n_trials": 12, "timeout_minutes": 16, "epochs_per_trial": {"warmup": 1, "joint": 3},
        "trial_val_subset": 736, "trial_train_subset": 32,
        "sampler": {"name": "TPE", "seed": 42}, "pruner": {"name": "Median", "n_startup_trials": 4, "n_warmup_steps": 2},
        "enqueue_pdf_start": True, "storage": "artifacts/optuna/t3_moe.db", "study_dir": "studies/t3_moe",
        "tuned_params": [dict(p) for p in TUNED],
    }
    cfg = dry_run_overrides(cfg, tmp_path / "dry")          # distinct study name, tmp folders, smoke, 2 trials x (1 + 1)
    cfg["trial_train_subset"] = 32
    return cfg


@pytest.fixture(scope="module")
def t3_sources(tmp_path_factory):
    return make_t3_sources(tmp_path_factory.mktemp("t3src_tune"))


def load_study(cfg):
    db = Path(cfg["storage"])
    return optuna.load_study(study_name=cfg["study"], storage=f"sqlite:///{db.as_posix()}")


@pytest.fixture(scope="module")
def dry_study(tiny_pets_root, tmp_path_factory, t3_sources):
    """One real 2-trial dry study (tiny models, 1 + 1 epochs) in a tmp dir; counts every model load."""
    paths, sha = t3_sources
    tmp = tmp_path_factory.mktemp("t3study")
    cfg = make_cfg(tiny_pets_root, tmp, paths, sha)
    real_load = SoftMoE.load_from_task2
    reference = real_load(paths, 1.0, expected_sha256=sha)
    ref_hashes = {"gate": parameter_hash(reference.gate), "experts": parameter_hash(reference.experts)}
    loads, saved = [], []

    def counting(paths_, tau, expected_sha256=None):
        model = real_load(paths_, tau, expected_sha256=expected_sha256)
        loads.append({"gate": parameter_hash(model.gate), "experts": parameter_hash(model.experts), "tau": tau})
        return model

    mp = pytest.MonkeyPatch()
    mp.setattr(SoftMoE, "load_from_task2", counting)
    try:
        out = run_study(cfg, on_checkpoint=saved.append)
    finally:
        mp.undo()
    return {"cfg": cfg, "out": out, "tmp": tmp, "saved": saved, "loads": loads, "ref": ref_hashes,
            "db": Path(cfg["storage"]), "paths": paths, "sha": sha, "root": tiny_pets_root}


# ------------------------------------------------------------------------------------ the dry run
def test_dry_run_writes_db_csv_user_attrs_and_run_info(dry_study):
    cfg, out, db = dry_study["cfg"], dry_study["out"], dry_study["db"]
    assert cfg["study"] == "t3_moe_dryrun" and str(dry_study["tmp"] / "dry") in str(out)
    assert db.exists() and dry_study["saved"] == [db, db]                      # persistence hook called once per trial
    trials = pd.read_csv(out / "trials.csv")
    assert len(trials) == 2 and trials["state"].eq("COMPLETE").all()
    assert {"params_joint_lr", "params_tau", "params_lambda_c", "params_lambda_b", "params_recon_l1_share",
            "user_attrs_run_id"} <= set(trials.columns)
    assert trials["value"].between(0, 1).all()                                 # fixed J values

    study = load_study(cfg)
    assert study.direction == optuna.study.StudyDirection.MINIMIZE and len(study.trials) == 2
    for t in study.trials:
        assert t.user_attrs["run_id"].endswith("_smoke") and "t3_moe_dryrun-trial" in t.user_attrs["run_id"]
        assert len(t.user_attrs["val_J_per_epoch"]) == 2                       # warm-up epoch + joint epoch, one counter
        assert t.user_attrs["val_J_per_epoch"][1] == pytest.approx(t.value)    # the one joint epoch is the best
        by_class = t.user_attrs["mean_w_by_class"]
        assert len(by_class) == 4 and all(sum(row) == pytest.approx(1.0, abs=1e-4) for row in by_class)
        assert "pruned_reason" not in t.user_attrs                             # nothing was pruned

    info = json.loads((out / "study_run.json").read_text())
    assert info["complete"] == 2 and info["pruned"] == 0 and info["fail"] == 0 and info["timeout_hit"] is False
    assert info["n_trials_requested"] == 2 and info["epochs_per_trial"] == {"warmup": 1, "joint": 1}


def test_trial_zero_is_the_pdf_start_and_later_trials_are_sampled(dry_study):
    trials = load_study(dry_study["cfg"]).trials
    assert trials[0].params == pytest.approx(PDF_START)
    assert trials[1].params != trials[0].params
    for name, spec in {s["name"]: s for s in TUNED}.items():
        assert spec["low"] <= trials[1].params[name] <= spec["high"]


def test_every_trial_reloads_the_task_2_weights(dry_study):
    loads, ref = dry_study["loads"], dry_study["ref"]
    assert len(loads) == 2                                                     # load_from_task2 once per trial
    for load in loads:                                                         # each trial started from the ORIGINAL weights
        assert load["gate"] == ref["gate"] and load["experts"] == ref["experts"]
    trials = load_study(dry_study["cfg"]).trials
    assert [load["tau"] for load in loads] == pytest.approx([t.params["tau"] for t in trials])


def test_trial_runs_are_isolated_cheap_and_without_epoch_0(dry_study):
    tmp = dry_study["tmp"] / "dry"
    folders = sorted((tmp / "trials" / "runs" / "task3").iterdir())
    assert len(folders) == 2 and all(d.name.endswith("_smoke") for d in folders)
    for d in folders:
        assert not list(d.glob("ckpt_*.pt")) and (d / "metrics.jsonl").exists() and (d / "config.yaml").exists()
        rows = [json.loads(line) for line in (d / "metrics.jsonl").read_text().splitlines()]
        assert [r["epoch"] for r in rows] == [1, 2] and [r["stage"] for r in rows] == ["warmup", "joint"]
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        assert cfg["train"]["val_subset"] == 16 and cfg["train"]["epoch0_validation"] is False
    assert not (tmp / "runs").exists()                                         # nothing in the folders resume="auto" reads


def test_continuing_the_study_adds_no_trial(dry_study):
    run_study(dry_study["cfg"])                                                # n_trials = 2 are already answered
    assert len(load_study(dry_study["cfg"]).trials) == 2


# ---------------------------------------------------------------------------------- pruning
def test_forced_collapse_prunes_the_trial_with_reason_collapse(tiny_pets_root, tmp_path, t3_sources, monkeypatch):
    paths, sha = t3_sources
    cfg = make_cfg(tiny_pets_root, tmp_path, paths, sha)
    cfg["n_trials"] = 1
    cfg["collapse"] = dict(REAL_THRESHOLDS)
    real = tr.validate

    def collapsing(model, loader, device, collapse_cfg=None):
        res = real(model, loader, device, collapse_cfg)
        res["mean_w_by_class"][1][0] = 0.95                                    # salt rows sent to the identity branch
        res["flags"] = tr.collapse_flags(res["mean_w"], res["mean_w_by_class"], REAL_THRESHOLDS)
        return res

    monkeypatch.setattr(tr, "validate", collapsing)
    out = run_study(cfg)
    study = load_study(cfg)
    trial = study.trials[0]
    assert trial.state == optuna.trial.TrialState.PRUNED
    assert trial.user_attrs["pruned_reason"] == "collapse" and trial.user_attrs["collapse_reasons"]
    assert len(trial.user_attrs["val_J_per_epoch"]) == 1                       # stopped at the FIRST validation (the warm-up epoch)
    run_dir = next((tmp_path / "dry" / "trials" / "runs" / "task3").iterdir())
    assert len((run_dir / "metrics.jsonl").read_text().splitlines()) == 1
    assert json.loads((out / "study_run.json").read_text())["pruned_reasons"] == {"collapse": 1}


def test_median_pruner_prunes_a_bad_trial_with_reason_median_and_persists_the_db(tiny_pets_root, tmp_path,
                                                                                 t3_sources, monkeypatch):
    """Fake the training (no CPU time): trial 0 is good, trial 1 is bad and the MedianPruner stops it."""
    paths, sha = t3_sources
    cfg = make_cfg(tiny_pets_root, tmp_path, paths, sha)
    cfg.update(pruner={"name": "Median", "n_startup_trials": 1, "n_warmup_steps": 0}, persist_root=str(tmp_path / "persist"))
    cfg["epochs_per_trial"] = {"warmup": 1, "joint": 2}
    seen = []

    def fake_train(tcfg, resume=None, on_checkpoint=None, report_fn=None, sources=None):
        seen.append({"warmup": tcfg["train"]["warmup_epochs"], "joint": tcfg["train"]["joint_epochs"],
                     "sources": sorted(sources), "output_root": tcfg["output_root"]})
        folder = tmp_path / f"fake{len(seen)}"
        folder.mkdir()
        j = 0.1 if len(seen) == 1 else 0.9                                     # trial 0 good, trial 1 bad
        for step in (1, 2, 3):
            report_fn(step, j, GOOD_FLAGS)                                     # the bad trial is pruned at step 1
        return folder, j

    monkeypatch.setattr(tune, "_train", fake_train)
    saved = []
    out = run_study(cfg, on_checkpoint=saved.append)
    trials = pd.read_csv(out / "trials.csv")
    assert list(trials["state"]) == ["COMPLETE", "PRUNED"]
    study = load_study(cfg)
    assert study.trials[1].user_attrs["pruned_reason"] == "median"
    assert study.trials[1].user_attrs["val_J_per_epoch"] == [0.9]
    assert study.trials[0].user_attrs["val_J_per_epoch"] == [0.1, 0.1, 0.1]
    assert all(s["warmup"] == 1 and s["joint"] == 2 for s in seen)
    assert seen[0]["sources"] == ["blur", "classifier", "occlusion", "salt", "t1"]      # the verified sources are passed on
    assert all(Path(s["output_root"]).name == "trials" for s in seen)
    persisted = tmp_path / "persist" / "optuna" / Path(cfg["storage"]).name                # DB copy after every trial
    assert persisted.exists() and persisted.stat().st_size > 0 and len(saved) == 2
    assert json.loads((out / "study_run.json").read_text())["pruned_reasons"] == {"median": 1}


def test_timeout_ends_the_study_before_n_trials_and_is_reported(tiny_pets_root, tmp_path, t3_sources, monkeypatch):
    import time
    paths, sha = t3_sources
    cfg = make_cfg(tiny_pets_root, tmp_path, paths, sha)
    cfg.update(n_trials=5, timeout_minutes=0.001)                              # 0.06 seconds

    def slow_train(tcfg, resume=None, on_checkpoint=None, report_fn=None, sources=None):
        time.sleep(0.2)
        folder = tmp_path / f"slow{len(list(tmp_path.glob('slow*')))}"
        folder.mkdir()
        return folder, 0.2

    monkeypatch.setattr(tune, "_train", slow_train)
    out = run_study(cfg)
    info = json.loads((out / "study_run.json").read_text())
    assert 1 <= info["answered"] < 5 and info["timeout_hit"] is True and info["n_trials_requested"] == 5


# ------------------------------------------------------------------------------------ guards
def test_tbd_guard_and_source_check_stop_the_study_before_trial_0(tiny_pets_root, tmp_path, t3_sources):
    paths, sha = t3_sources
    for key, value in (("n_trials", "TBD_AFTER_BENCHMARK"), ("timeout_minutes", "TBD_AFTER_BENCHMARK"),
                       ("epochs_per_trial", "TBD_AFTER_BENCHMARK"),
                       ("epochs_per_trial", {"warmup": "TBD_AFTER_BENCHMARK", "joint": 1})):
        cfg = make_cfg(tiny_pets_root, tmp_path, paths, sha)
        cfg[key] = value
        with pytest.raises(ValueError, match="TBD"):
            run_study(cfg)
    cfg = make_cfg(tiny_pets_root, tmp_path, paths, sha)
    cfg["sources"]["expected_sha256"] = dict(sha, blur="1" * 64)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        run_study(cfg)
    assert not Path(cfg["storage"]).exists()                                   # no study database was created


def test_search_space_checks():
    tune._check_search_space(TUNED)
    with pytest.raises(ValueError, match="unknown tuned parameter"):
        tune._check_search_space([{"name": "batch", "type": "float", "low": 1, "high": 2}])
    with pytest.raises(ValueError, match="low < high"):
        tune._check_search_space([{"name": "tau", "type": "float", "low": 2.0, "high": 1.0}])
    with pytest.raises(ValueError, match="positive"):
        tune._check_search_space([{"name": "tau", "type": "float", "low": 0.0, "high": 1.0}])
    with pytest.raises(ValueError, match="inside \\(0, 1\\)"):
        tune._check_search_space([{"name": "recon_l1_share", "type": "float", "low": 0.3, "high": 1.0}])


# ------------------------------------------------------------------------------ trial / dry-run config
def test_make_trial_config_maps_values_and_budget(tiny_pets_root, tmp_path, t3_sources):
    paths, sha = t3_sources
    cfg = make_cfg(tiny_pets_root, tmp_path, paths, sha)
    cfg.update(epochs_per_trial={"warmup": 1, "joint": 3}, trial_val_subset=8, trial_train_subset=24)
    before = json.dumps(cfg, sort_keys=True)
    values = {"joint_lr": 1e-4, "tau": 3.0, "lambda_c": 0.2, "lambda_b": 0.05, "recon_l1_share": 0.6}
    tcfg = make_trial_config(cfg, values, 7)
    t = tcfg["train"]
    assert t["joint_lr"] == 1e-4 and tcfg["model"]["tau"] == 3.0 and t["lambda_c"] == 0.2 and t["lambda_b"] == 0.05
    assert t["lambda_1"] == pytest.approx(0.6) and t["lambda_s"] == pytest.approx(0.4)        # r and 1 - r
    assert (t["warmup_epochs"], t["joint_epochs"]) == (1, 3) and t["val_subset"] == 8 and t["train_subset"] == 24
    assert t["val_every_epochs"] == 1 and t["epoch0_validation"] is False and t["max_steps"] is None
    assert "-trial7" in tcfg["run_id"] and tcfg["run_id"].endswith("_smoke") and tcfg["run"]["desc"].endswith("-trial7")
    assert Path(tcfg["output_root"]).name == "trials" and tcfg["persist_root"] == tcfg["output_root"]
    assert json.dumps(cfg, sort_keys=True) == before                                          # base config untouched
    assert set(PARAM_TARGETS) == {s["name"] for s in TUNED}

    cfg["epochs_per_trial"] = 4                                                               # scripts/tune.py --epochs 4
    assert (make_trial_config(cfg, values, 0)["train"]["warmup_epochs"], make_trial_config(cfg, values, 0)["train"]["joint_epochs"]) == (1, 4)
    cfg["trial_val_subset"] = 10
    with pytest.raises(ValueError, match="multiple of 4"):
        make_trial_config(cfg, values, 0)
    cfg["trial_val_subset"] = None                                                            # falls back to train.val_subset
    assert make_trial_config(cfg, values, 0)["train"]["val_subset"] == 16


def test_dry_run_overrides_isolate_the_study(tmp_path):
    base = {"study": "t3_moe", "storage": "artifacts/optuna/t3_moe.db", "study_dir": "studies/t3_moe", "output_root": "artifacts",
            "persist_root": "artifacts", "run": {"smoke": False, "desc": "t3moe"}, "n_trials": 12, "timeout_minutes": 16,
            "epochs_per_trial": {"warmup": 1, "joint": 3}, "trial_val_subset": 736}
    copy = json.loads(json.dumps(base))
    dry = dry_run_overrides(base, tmp_path)
    assert base == copy                                                                       # the real config is not changed
    assert dry["study"] == "t3_moe_dryrun" and dry["run"]["smoke"] is True
    for key in ("storage", "study_dir", "output_root", "persist_root"):
        assert str(tmp_path) in str(dry[key])
    assert dry["n_trials"] == 2 and dry["epochs_per_trial"] == {"warmup": 1, "joint": 1} and dry["trial_val_subset"] is None


# ----------------------------------------------------------------------------------- final config
def test_write_final_config_is_loadable_and_trainable(dry_study, tmp_path):
    cfg = dry_study["cfg"]
    study = load_study(cfg)
    out = write_final_config(cfg, study, tmp_path / "final.yaml")
    best = study.best_trial
    final = load_config(out)                                                                  # loads like any config
    assert final["train"]["joint_lr"] == pytest.approx(best.params["joint_lr"])
    assert final["model"]["tau"] == pytest.approx(best.params["tau"])
    assert final["train"]["lambda_c"] == pytest.approx(best.params["lambda_c"])
    assert final["train"]["lambda_b"] == pytest.approx(best.params["lambda_b"])
    assert final["train"]["lambda_1"] == pytest.approx(best.params["recon_l1_share"])
    assert final["train"]["lambda_s"] == pytest.approx(1.0 - best.params["recon_l1_share"])
    assert (final["train"]["warmup_epochs"], final["train"]["joint_epochs"]) == (1, 1)         # final budget = the base config's
    assert final["component"] == "soft_moe" and final["train"]["batch_size"] == 8
    src = final["final_config_source"]
    assert src["study"] == cfg["study"] and src["best_trial"] == best.number and src["best_value"] == pytest.approx(best.value)
    assert src["trial_0"]["params"] == pytest.approx(PDF_START) and src["trial_0"]["state"] == "COMPLETE"
    assert len(src["best_val_J_per_epoch"]) == 2
    text = yaml.safe_load(Path(out).read_text())
    assert not {"data_root", "output_root", "persist_root", "device", "num_workers"} & set(text)   # device values merged again

    # usable as --config <final> (device keys come from the profile; here overridden to the tmp folders)
    overrides = {"data_root": str(dry_study["root"]), "output_root": str(tmp_path / "out"), "persist_root": str(tmp_path / "out"),
                 "num_workers": 0, "device": "test", "train.epoch0_validation": False}
    run_dir = run_training(load_config(out, "local", overrides))
    assert (run_dir / "ckpt_best.pt").exists()
    ck = tr.load_checkpoint(run_dir / "ckpt_best.pt")
    assert ck["config"]["model"]["tau"] == pytest.approx(best.params["tau"])


def test_write_final_config_needs_a_completed_trial(tmp_path):
    study = optuna.create_study(study_name="empty", storage=f"sqlite:///{(tmp_path / 'e.db').as_posix()}")
    with pytest.raises(ValueError, match="no completed trial"):
        write_final_config({"train": {}}, study, tmp_path / "x.yaml")
    assert not (tmp_path / "x.yaml").exists()


# ------------------------------------------------------------------------------------ the real YAML
def test_real_config_has_every_key_and_valid_ranges():
    cfg = load_config("configs/task3_moe.yaml", "local")
    for key in ("task", "component", "seed", "data_config", "run", "sources", "model", "train", "collapse", "study", "n_trials",
                "timeout_minutes", "epochs_per_trial", "trial_val_subset", "trial_train_subset", "sampler", "pruner",
                "enqueue_pdf_start", "storage", "study_dir", "budget", "eval", "tuned_params"):
        assert key in cfg, key
    assert cfg["task"] == "t3" and cfg["component"] == "soft_moe" and cfg["study"] == "t3_moe"
    for key in ("batch_size", "warmup_epochs", "joint_epochs", "warmup_lr", "joint_lr", "weight_decay", "lambda_1", "lambda_s",
                "lambda_c", "lambda_b", "amp", "scheduler", "grad_clip", "train_subset", "val_subset", "val_every_epochs",
                "val_batch_size", "checkpoint_every_minutes", "log_every_steps", "sample_every_epochs", "max_steps",
                "pause_after_steps"):
        assert key in cfg["train"], key
    assert cfg["train"]["batch_size"] % 4 == 0 and cfg["train"]["lambda_1"] + cfg["train"]["lambda_s"] == pytest.approx(1.0)
    assert cfg["sources"]["dir"] is None and cfg["model"]["tau"] > 0 and set(cfg["collapse"]) == {"min_mean_weight", "max_foreign_weight"}
    assert set(cfg["budget"]) == {"max_minutes"} and set(cfg["eval"]) == {"max_rows", "batch_size", "n_examples"}

    tune._check_search_space(cfg["tuned_params"])                                             # the YAML is valid
    spaces = {s["name"]: s for s in cfg["tuned_params"]}
    assert set(spaces) == set(PARAM_TARGETS) == set(PDF_START)
    for name, value in PDF_START.items():
        assert spaces[name]["low"] <= value <= spaces[name]["high"], name                      # trial 0 lies inside every range
    study = optuna.create_study()
    study.enqueue_trial(PDF_START)
    assert suggest_params(study.ask(), cfg["tuned_params"]) == pytest.approx(PDF_START)
    cfg["output_root"] = "artifacts"
    cfg["run"]["smoke"] = False
    tcfg = make_trial_config(cfg, dict(PDF_START), 0)                                          # works on the real structure
    assert tcfg["train"]["lambda_1"] == pytest.approx(0.8) and tcfg["train"]["lambda_s"] == pytest.approx(0.2)
    assert tcfg["train"]["val_subset"] == cfg["trial_val_subset"]
    assert isinstance(cfg["epochs_per_trial"], dict) and set(cfg["epochs_per_trial"]) == {"warmup", "joint"}


def test_real_config_budget_values_carry_the_approval_comment():
    lines = Path("configs/task3_moe.yaml").read_text(encoding="utf-8").splitlines()
    for key in ("warmup_epochs", "joint_epochs", "n_trials", "timeout_minutes", "epochs_per_trial", "trial_val_subset",
                "pruner", "max_minutes"):
        line = next(line for line in lines if line.lstrip().startswith(key + ":"))
        # D99: first approval; D100: the values were reduced for the 20-25 minute cap, the comment says so
        assert ("RECOMMENDATION, approved by the student on 2026-10-05" in line
                or "REDUCED by the student's 20-25 minute cap, 2026-10-05 (D100" in line), key


# ---------------------------------------------------------------------------------- locked test set
def test_study_code_never_touches_the_test_split():
    text = Path(tune.__file__).read_text(encoding="utf-8")
    assert "final_test=True" not in text and "pets_test_manifest" not in text
    assert 'split="test"' not in text and "split='test'" not in text
