"""Task 2 specialists on the tiny SYNTHETIC dataset of tests/conftest.py (CPU, no real data).

Covers: the fixed:k training loader, validation filtered to the specialist's corruption, three runs in
three folders, checkpoint config, the dispatcher error, resume, the shared study (dry run, user
attributes, continuation, pruning logic), write_final_config and the locked test split.
"""
import json
from pathlib import Path

import optuna
import pandas as pd
import pytest
import torch
import yaml

from genai.common.checkpoint import load_checkpoint
from genai.models.autoencoder import UniversalAE
from genai.pets import dataset as pets_dataset
from genai.pets.dataset import resolve_data_paths
from genai.tasks.task1.config import is_tbd, load_config
from genai.tasks.task1.tune import dry_run_overrides
from genai.tasks.task2 import SPECIALIST_COND_ID, runs
from genai.tasks.task2 import specialist as spec
from genai.tasks.task2.train import run_training
from genai.tasks.task2.tune import run_study

TRIAL_FOLDER = spec.TRIAL_ROOT
COND_NAME = {"salt": "salt_pepper", "blur": "gaussian_blur", "occlusion": "occlusion"}


def make_cfg(root, tmp_path, corruption="salt", **train):
    """Specialist config shrunk to a tiny, fast model: 32 train images, batch 8 -> 4 steps per epoch."""
    cfg = load_config(spec.BASE_CONFIG, "local")
    cfg.update(device="test", data_root=str(root), output_root=str(tmp_path / "out"),
               persist_root=str(tmp_path / "out"), num_workers=0)
    # the real config uses the conv latent (D42); most tiny tests pin the small dense model, like Task 1's tests
    cfg["model"].update(latent="dense", depth=4, base_channels=4, bottleneck_dim=16, dropout=0.1)
    cfg["train"].update(batch_size=8, epochs=1, train_subset=32, sample_every_epochs=1, **train)
    cfg["run"]["smoke"] = True
    cfg["specialist"]["corruption"] = corruption
    return cfg


def read_rows(run_dir):
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]


# ------------------------------------------------------------------------------------------ data
@pytest.mark.parametrize("corruption", sorted(SPECIALIST_COND_ID))
def test_training_loader_yields_only_its_own_class(tiny_pets_root, tmp_path, corruption):
    cfg = make_cfg(tiny_pets_root, tmp_path, corruption)
    k = SPECIALIST_COND_ID[corruption]
    _ds, _sampler, loader = spec.build_train_loader(cfg, resolve_data_paths(cfg), 8, k)
    for n, (corrupted, clean, cond, _sev) in enumerate(loader):
        assert (cond == k).all() and corrupted.shape == (8, 3, 128, 128)
        assert not torch.equal(corrupted, clean)                  # the image really is corrupted
        if n == 2:
            break
    assert n == 2


@pytest.mark.parametrize("corruption", sorted(SPECIALIST_COND_ID))
def test_validation_uses_only_rows_of_its_own_condition(tiny_pets_root, tmp_path, corruption):
    cfg = make_cfg(tiny_pets_root, tmp_path, corruption)
    k = SPECIALIST_COND_ID[corruption]
    ds, rows, loader, sha = spec.build_val_loader(cfg, resolve_data_paths(cfg), k)
    assert len(rows) == 16 and sha and all(ds.rows[i]["cond_id"] == k for i in rows)
    assert all((cond == k).all() for _c, _t, cond, _s in loader)
    cfg["train"]["val_subset"] = 5                                # filter first, THEN cut
    _ds, rows5, loader5, _sha = spec.build_val_loader(cfg, resolve_data_paths(cfg), k)
    assert len(rows5) == 5 and sum(len(c) for _a, _b, c, _s in loader5) == 5
    assert all(ds.rows[i]["cond_id"] == k for i in rows5)


# ----------------------------------------------------------------------------------- training
def test_three_runs_three_folders_and_checkpoint_config(tiny_pets_root, tmp_path, monkeypatch):
    log = tmp_path / "test_access.log"
    monkeypatch.setattr(pets_dataset, "TEST_LOG_PATH", log)
    run_dirs = {}
    for corruption in SPECIALIST_COND_ID:
        cfg = make_cfg(tiny_pets_root, tmp_path, corruption)
        run_dirs[corruption] = spec.train_specialist(cfg)
    assert not log.exists()                                       # the test split was never opened
    assert len({d.parent for d in run_dirs.values()}) == 3        # three separate folders
    for corruption, run_dir in run_dirs.items():
        assert run_dir.parent.name == f"task2_specialist_{corruption}"
        assert run_dir.name.endswith(f"_test_t2spec_{corruption}_smoke")
        for name in ("config.yaml", "ckpt_last.pt", "ckpt_best.pt", "metrics.jsonl"):
            assert (run_dir / name).exists()
        assert len(list((run_dir / "samples").glob("*.png"))) == 1
        # metrics: only this corruption's rows were validated (16 rows of one condition)
        row = read_rows(run_dir)[0]
        assert list(row["per_condition"]) == [COND_NAME[corruption]] and row["val_rows"] == 16
        assert row["global_step"] == 4 and all(k.startswith(COND_NAME[corruption]) for k in row["by_severity"])
        # checkpoint: config says which specialist this is and rebuilds the model
        ck = load_checkpoint(run_dir / "ckpt_best.pt")
        c = ck["config"]
        assert c["component"] == "specialist" and c["run_id"] == run_dir.name
        assert c["specialist"] == {"corruption": corruption, "cond_id": SPECIALIST_COND_ID[corruption]}
        assert ck["best_metric"] == pytest.approx(row["val_J"])
        model = UniversalAE.from_config(c["model"])
        model.load_state_dict(ck["model"])
        assert model.hparams["base_channels"] == 4 and model.bottleneck_dim == 16
    # independent parameters: the three specialists are different models
    weights = [load_checkpoint(d / "ckpt_best.pt")["model"]["to_latent.2.weight"] for d in run_dirs.values()]
    assert not torch.equal(weights[0], weights[1]) and not torch.equal(weights[1], weights[2])


@pytest.mark.parametrize("bad", [None, "pepper", "classifier"])
def test_missing_or_invalid_corruption_is_a_clear_error(tiny_pets_root, tmp_path, bad):
    cfg = make_cfg(tiny_pets_root, tmp_path, bad)
    with pytest.raises(ValueError, match="specialist.corruption"):
        run_training(cfg)                                         # through the lead's dispatcher
    with pytest.raises(ValueError, match="specialist.corruption"):
        spec.train_specialist(cfg)
    assert not (tmp_path / "out").exists()                        # nothing was created


def test_resume_continues_global_step_and_rejects_other_corruption(tiny_pets_root, tmp_path):
    cfg = make_cfg(tiny_pets_root, tmp_path, "blur", pause_after_steps=3)
    cfg["train"]["epochs"] = 2                                    # 8 steps in total; pause inside epoch 1
    run_dir = run_training(cfg)
    paused = load_checkpoint(run_dir / "ckpt_last.pt")
    assert paused["global_step"] == 3 and paused["epoch"] == 0 and paused["step_in_epoch"] == 3
    assert not (run_dir / "metrics.jsonl").exists()               # nothing validated yet

    wrong = json.loads(json.dumps(cfg))                           # the same run, but as another specialist
    wrong["specialist"]["corruption"] = "salt"
    with pytest.raises(ValueError, match="belongs to"):
        run_training(wrong, resume=str(run_dir / "ckpt_last.pt"))

    cfg["train"]["pause_after_steps"] = None
    assert runs.find_latest_checkpoint(cfg) == run_dir / "ckpt_last.pt"
    assert run_training(cfg, resume="auto") == run_dir            # same run continues
    final = load_checkpoint(run_dir / "ckpt_last.pt")
    assert final["global_step"] == 8 and final["epoch"] == 2 and final["step_in_epoch"] == 0
    assert [r["epoch"] for r in read_rows(run_dir)] == [1, 2]


def test_tbd_epochs_need_max_steps(tiny_pets_root, tmp_path):
    cfg = make_cfg(tiny_pets_root, tmp_path, "occlusion")
    cfg["train"]["epochs"] = "TBD_AFTER_BENCHMARK"
    with pytest.raises(ValueError, match="TBD"):
        spec.train_specialist(cfg)
    cfg["train"]["max_steps"] = 2                                 # a step cap is enough
    run_dir = spec.train_specialist(cfg)
    assert load_checkpoint(run_dir / "ckpt_last.pt")["global_step"] == 2


# ------------------------------------------------------------------------------------- study
def small_space(cfg):
    """Tiny model sizes and batches so that a trial takes a few seconds on the CPU."""
    cfg["tuned_params"] = [dict(p) for p in cfg["tuned_params"]]
    for p in cfg["tuned_params"]:
        p["choices"] = {"bottleneck": [16, 32], "channels": [4, 8], "batch": [4, 8]}.get(p["name"])
        if p["choices"] is None:
            del p["choices"]
    return cfg


@pytest.fixture(scope="module")
def dry_study(tiny_pets_root, tmp_path_factory):
    """One 2-trial, 1-epoch shared study (3 trainings per trial) in a tmp dir. Returns a dict."""
    tmp = tmp_path_factory.mktemp("t2study")
    base = make_cfg(tiny_pets_root, tmp, corruption=None)
    cfg = small_space(dry_run_overrides(base, tmp / "dry"))
    cfg.update(trial_train_subset=16, trial_val_subset=6)
    cfg.update(n_trials="TBD_AFTER_BENCHMARK", epochs_per_trial="TBD_AFTER_BENCHMARK")   # the guard (the YAML holds real budgets, D44)
    with pytest.raises(ValueError, match="TBD"):                  # budgets stay TBD unless given
        run_study(cfg)
    cfg.update(n_trials=2, epochs_per_trial=1)
    saved = []
    out = run_study(cfg, on_checkpoint=saved.append)
    return {"cfg": cfg, "out": out, "tmp": tmp, "saved": saved,
            "db": tmp / "dry" / "optuna" / "t2_specialist_shared_dryrun.db"}


def test_study_dry_run_writes_db_csv_and_per_corruption_attributes(dry_study):
    cfg, out, db = dry_study["cfg"], dry_study["out"], dry_study["db"]
    assert cfg["study"] == "t2_specialist_shared_dryrun" and str(dry_study["tmp"] / "dry") in str(out)
    assert db.exists() and dry_study["saved"] == [db, db]         # persistence hook called once per trial
    trials = pd.read_csv(out / "trials.csv")
    assert len(trials) == 2 and trials["state"].eq("COMPLETE").all()
    assert {"params_lr", "params_bottleneck", "params_channels", "params_batch",
            "params_l1_ssim_weight"} <= set(trials.columns)
    for c in ("salt", "blur", "occlusion"):
        assert trials[f"user_attrs_J_{c}"].notna().all()
    mean_j = trials[["user_attrs_J_salt", "user_attrs_J_blur", "user_attrs_J_occlusion"]].mean(axis=1)
    assert (trials["value"] - mean_j).abs().max() < 1e-9          # trial value = mean of the three J


def test_study_trial_runs_are_isolated_cheap_and_use_the_trial_val_subset(dry_study):
    tmp = dry_study["tmp"] / "dry"
    for corruption in SPECIALIST_COND_ID:
        folder = tmp / TRIAL_FOLDER / "runs" / f"task2_specialist_{corruption}"
        run_dirs = sorted(folder.iterdir())
        assert len(run_dirs) == 2                                 # one run per trial
        assert [f"-trial{n}-{corruption}" in d.name for n, d in enumerate(run_dirs)] == [True, True]
        for d in run_dirs:
            assert not list(d.glob("ckpt_*.pt")) and (d / "metrics.jsonl").exists()
            assert read_rows(d)[0]["val_rows"] == 6               # trial_val_subset wins over train.val_subset
    assert not (tmp / "runs").exists()                            # nothing in the real run folders



def test_study_continuation_adds_no_trials(dry_study):
    run_study(dry_study["cfg"])                                   # n_trials=2 already done
    db = dry_study["db"]
    study = optuna.load_study(study_name=dry_study["cfg"]["study"], storage=f"sqlite:///{db.as_posix()}")
    assert len(study.trials) == 2


def test_pruning_after_a_corruption_uses_the_running_mean(tiny_pets_root, tmp_path, monkeypatch):
    """Fake the training (no CPU time): trial 0 is good, trial 1 is bad and must be pruned after salt."""
    cfg = small_space(dry_run_overrides(make_cfg(tiny_pets_root, tmp_path, corruption=None), tmp_path / "dry"))
    cfg.update(n_trials=2, epochs_per_trial=1, pruner={"name": "Median", "n_startup_trials": 1, "n_warmup_steps": 0})
    seen = []

    def fake_train(tcfg, resume=None, on_checkpoint=None):
        seen.append(tcfg["specialist"]["corruption"])
        d = tmp_path / f"fake{len(seen)}"
        d.mkdir()
        return d, 0.1 if len(seen) <= 3 else 0.9              # trial 0: three good runs; trial 1: bad

    monkeypatch.setattr(spec, "_train_specialist", fake_train)
    out = run_study(cfg)
    trials = pd.read_csv(out / "trials.csv")
    assert list(trials["state"]) == ["COMPLETE", "PRUNED"]
    assert seen == ["salt", "blur", "occlusion", "salt"]          # the bad trial stopped after its first corruption
    assert trials["user_attrs_J_salt"].notna().all() and trials["user_attrs_J_blur"].isna()[1]
    assert trials["value"][0] == pytest.approx(0.1)


# ------------------------------------------------------------------------------ final config
def test_write_final_config_is_loadable_and_trainable(dry_study, tiny_pets_root, tmp_path):
    cfg, db = dry_study["cfg"], dry_study["db"]
    out = spec.write_final_config(db, tmp_path / "final.yaml", cfg=cfg)
    final = load_config(out)                                      # loads like any config
    study = optuna.load_study(study_name=cfg["study"], storage=f"sqlite:///{db.as_posix()}")
    best = study.best_trial
    assert final["train"]["lr"] == pytest.approx(best.params["lr"])
    assert final["train"]["batch_size"] == best.params["batch"]
    assert final["train"]["alpha"] == pytest.approx(best.params["l1_ssim_weight"])
    assert final["model"]["base_channels"] == best.params["channels"]
    assert final["model"]["bottleneck_dim"] == best.params["bottleneck"]
    assert final["model"]["depth"] == 4 and final["component"] == "specialist"
    assert final["specialist"]["corruption"] is None and is_tbd(final["train"]["epochs"])
    src = final["final_config_source"]
    assert src["study"] == cfg["study"] and src["best_trial"] == best.number
    assert src["best_value"] == pytest.approx(best.value) and set(src["J_per_corruption"]) == {"salt", "blur", "occlusion"}
    assert not {"data_root", "output_root", "persist_root"} & set(yaml.safe_load(Path(out).read_text()))

    # usable as --config <final> --set specialist.corruption=blur (epochs TBD -> a step cap here)
    overrides = {"specialist.corruption": "blur", "train.max_steps": 2, "train.train_subset": 16,
                 "device": "test", "data_root": str(tiny_pets_root), "output_root": str(tmp_path / "out"),
                 "persist_root": str(tmp_path / "out"), "num_workers": 0, "run.smoke": True}
    run_dir = run_training(load_config(out, "local", overrides))
    ck = load_checkpoint(run_dir / "ckpt_last.pt")
    assert ck["global_step"] == 2 and ck["config"]["specialist"]["cond_id"] == 2
    assert ck["config"]["model"]["base_channels"] == best.params["channels"]


def test_write_final_config_command_line_and_errors(dry_study, tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(dry_study["cfg"]), encoding="utf-8")
    out = tmp_path / "cli_final.yaml"
    spec.main(["write-final-config", "--config", str(base), "--study-db", str(dry_study["db"]), "--out", str(out)])
    assert "final_config_source" in load_config(out)
    with pytest.raises(FileNotFoundError):                         # no silent empty database
        spec.write_final_config(tmp_path / "nope.db", tmp_path / "x.yaml", cfg=dry_study["cfg"])
    assert not (tmp_path / "nope.db").exists()


# ----------------------------------------------------------------------------- locked test set
def test_specialist_code_never_touches_the_test_split(tiny_pets_root):
    text = Path(spec.__file__).read_text(encoding="utf-8")
    assert "final_test=True" not in text and "pets_test_manifest" not in text and 'split="test"' not in text
    with pytest.raises(PermissionError):
        pets_dataset.PetsManifestDataset(tiny_pets_root / "manifests" / "pets_test_manifest.jsonl", "test", False,
                                         tiny_pets_root)


# ------------------------------------------------------------------------- conv latent (D42)
def test_default_config_uses_the_conv_latent_and_valid_bottleneck_choices():
    cfg = load_config(spec.BASE_CONFIG, "local")
    assert cfg["model"]["latent"] == "conv" and cfg["model"]["depth"] == 3
    cells = (128 // 2 ** cfg["model"]["depth"]) ** 2                     # 16 x 16 latent cells = 256
    assert cfg["model"]["bottleneck_dim"] % cells == 0
    choices = next(p["choices"] for p in cfg["tuned_params"] if p["name"] == "bottleneck")
    assert choices and all(c % cells == 0 for c in choices)             # the old dense choices (64..512) would fail
    for c in choices:                                                    # every choice really builds
        UniversalAE.from_config(dict(cfg["model"], base_channels=4, bottleneck_dim=c))


def test_conv_latent_specialist_trains_and_checkpoint_rebuilds(tiny_pets_root, tmp_path):
    cfg = make_cfg(tiny_pets_root, tmp_path, "blur")
    cfg["model"].update(latent="conv", depth=3, base_channels=4, bottleneck_dim=256)
    run_dir = spec.train_specialist(cfg)
    ck = load_checkpoint(run_dir / "ckpt_best.pt")
    assert ck["config"]["model"]["latent"] == "conv"
    model = UniversalAE.from_config(ck["config"]["model"])
    model.load_state_dict(ck["model"])
    assert model.latent == "conv" and tuple(model.encode(torch.zeros(1, 3, 128, 128)).shape) == (1, 1, 16, 16)


# ------------------------------------------------------------- study robustness fixes (D42)
def test_open_study_seed_offset_running_trials_fail_and_budget_counts_answers(tmp_path, monkeypatch):
    """The three Task 1 study fixes, shared by the classifier and specialist studies (runs.open_study)."""
    seeds = []
    real_sampler = optuna.samplers.TPESampler
    monkeypatch.setattr(optuna.samplers, "TPESampler", lambda seed=None: (seeds.append(seed), real_sampler(seed=seed))[1])
    cfg = {"study": "t2_robust", "sampler": {"seed": 42}, "pruner": {"n_startup_trials": 1, "n_warmup_steps": 0}}
    storage = f"sqlite:///{(tmp_path / 's.db').as_posix()}"

    study = runs.open_study(cfg, "minimize", storage)
    assert [x for x in seeds if x is not None] == [42]            # fresh study: plain seed
    study.ask()                                                   # trial 0 stays RUNNING ("killed process")
    ok = study.ask(); study.tell(ok, 0.5)                         # trial 1 COMPLETE
    bad = study.ask(); study.tell(bad, state=optuna.trial.TrialState.FAIL)       # trial 2 FAIL
    pruned = study.ask(); study.tell(pruned, state=optuna.trial.TrialState.PRUNED)  # trial 3 PRUNED

    study = runs.open_study(cfg, "minimize", storage)             # restart
    assert [x for x in seeds if x is not None] == [42, 42 + 4]   # seed + number of existing trials (load_study makes a default sampler with seed None)
    states = [t.state for t in study.trials]
    assert optuna.trial.TrialState.RUNNING not in states and states[0] == optuna.trial.TrialState.FAIL
    assert runs.n_answered(study) == 2                            # only COMPLETE and PRUNED count; FAILs are retried
