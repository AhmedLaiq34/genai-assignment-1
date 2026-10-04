"""Task 3 Optuna study `t3_moe`. Implements CONTRACTS 3.6 (fixed objective J) and 3.13; design in plan B15, B17, B18, D4.

Entry point:  run_study(cfg, on_checkpoint=None) -> study directory (Path)

* Tuned (PDF page 6): joint learning rate, temperature tau, lambda_c, lambda_b and the reconstruction
  weighting (one share r: lambda_1 = r, lambda_s = 1 - r).
* Every trial trains the soft MoE with genai.tasks.task3.train._train: a short warm-up plus a few joint
  epochs, starting again from the ORIGINAL Task 2 checkpoints (hash-checked). The trial's score is the
  best fixed J = 0.5*L1 + 0.5*(1-SSIM) of its non-collapsed joint-stage validations on the first
  `trial_val_subset` validation rows. J does not depend on r, so trials stay comparable.
* Pruning: MedianPruner on J after each epoch, and ROUTING COLLAPSE: a validation flagged by
  collapse_flags prunes the trial at once (user attribute pruned_reason = "collapse", else "median").
* Trial 0 is the PDF's starting point (PDF_START), enqueued only when the study has no trial yet.
* The budget is `n_trials` answered trials (COMPLETE or PRUNED) or `timeout_minutes`, whichever comes
  first (a trial that has started is finished). Storage: SQLite, load_if_exists, so a restarted session
  continues the same study; after every trial the DB is copied to persist_root and on_checkpoint is called.
* Trial runs live in <output_root>/trials/runs/task3/ (away from the real run folders, so resume="auto" of the
  final training never picks up a trial) and their checkpoints are deleted after each trial.
* This module never reads the test set.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import optuna
import yaml

from genai.tasks.task1.config import deep_copy, is_tbd, make_run_id, resolve, set_dotted
from genai.tasks.task1.tune import export_study, suggest_params
from genai.tasks.task2 import runs
from genai.tasks.task3.sources import find_sources, verify_sources
from genai.tasks.task3.train import _train, expected_hashes

# Where each tuned parameter goes inside the config (names are the ones in tuned_params).
# recon_l1_share is the share r: it sets lambda_1 = r and lambda_s = 1 - r (decision B8).
PARAM_TARGETS = {
    "joint_lr": "train.joint_lr",
    "tau": "model.tau",
    "lambda_c": "train.lambda_c",
    "lambda_b": "train.lambda_b",
    "recon_l1_share": ("train.lambda_1", "train.lambda_s"),
}

# The PDF's starting values (lambda_1 0.8, lambda_s 0.2, lambda_c 0.1, lambda_b 0.01) are enqueued as trial 0.
# The PDF does not give tau and the joint lr: tau 1.0 is the plain softmax, 5e-5 the geometric middle of the lr range.
PDF_START = {"joint_lr": 5e-5, "tau": 1.0, "lambda_c": 0.1, "lambda_b": 0.01, "recon_l1_share": 0.8}

# Device values come from configs/devices/<profile>.yaml and are merged in again when a config is loaded.
DEVICE_KEYS = ("device", "device_profile", "data_root", "dataset_dir", "output_root", "persist_root", "num_workers")


# ------------------------------------------------------------------------------------ budgets
def trial_epochs(cfg: dict) -> tuple:
    """(warm-up epochs, joint epochs) of one trial, from cfg['epochs_per_trial'] = {warmup: 1, joint: 3}.

    A plain number (scripts/tune.py --epochs N) is read as the number of JOINT epochs with 1 warm-up epoch.
    Raises ValueError for TBD_AFTER_BENCHMARK (the budget was never measured).
    """
    spec = cfg["epochs_per_trial"]
    if isinstance(spec, dict):
        warmup, joint = spec["warmup"], spec["joint"]
    else:
        warmup, joint = 1, spec
    if is_tbd(warmup) or is_tbd(joint):
        raise ValueError("epochs_per_trial is TBD_AFTER_BENCHMARK: measure first or pass explicit values")
    return int(warmup), int(joint)


def _check_budget(cfg: dict) -> None:
    """Fail before the first trial if a budget was never set (the TBD guard of Tasks 1 and 2)."""
    for key in ("n_trials", "timeout_minutes", "epochs_per_trial"):      # timeout_minutes: null = no time limit
        if key not in cfg or is_tbd(cfg[key]):
            raise ValueError(f"{key} is TBD_AFTER_BENCHMARK (or missing): measure first "
                             f"(scripts/benchmark.py) or pass explicit values")
    trial_epochs(cfg)                                         # also checks the two numbers inside


def _check_search_space(space: list) -> None:
    """Fail before the first trial if a tuned parameter would break training."""
    for spec in space:
        name = spec["name"]
        if name not in PARAM_TARGETS:
            raise ValueError(f"unknown tuned parameter {name!r}; known: {sorted(PARAM_TARGETS)}")
        if spec["type"] != "float" or not spec["low"] < spec["high"]:
            raise ValueError(f"{name}: expected a float range with low < high, got {spec}")
        if name in ("tau", "joint_lr", "lambda_c", "lambda_b") and spec["low"] <= 0:
            raise ValueError(f"{name}: the lower end of the range must be positive, got {spec['low']}")
        if name == "recon_l1_share" and not (0 < spec["low"] and spec["high"] < 1):
            raise ValueError(f"recon_l1_share must stay inside (0, 1) so that lambda_s = 1 - r > 0, got {spec}")


# ------------------------------------------------------------------------------------ trial config
def apply_values(cfg: dict, values: dict) -> None:
    """Write sampled / best values into a config (in place), following PARAM_TARGETS."""
    for name, value in values.items():
        target = PARAM_TARGETS[name]
        if isinstance(target, tuple):                          # recon_l1_share r -> lambda_1 = r, lambda_s = 1 - r
            set_dotted(cfg, target[0], float(value))
            set_dotted(cfg, target[1], 1.0 - float(value))
        else:
            set_dotted(cfg, target, value)


def make_trial_config(cfg: dict, values: dict, trial_number: int) -> dict:
    """Copy of the base config with the sampled values and the trial's budget inserted (decisions B13, B15)."""
    tcfg = deep_copy(cfg)
    apply_values(tcfg, values)
    warmup, joint = trial_epochs(cfg)
    tc = tcfg["train"]
    tc["warmup_epochs"], tc["joint_epochs"] = warmup, joint
    if cfg.get("trial_train_subset"):
        tc["train_subset"] = cfg["trial_train_subset"]
    # validation rows of a trial: trial_val_subset wins; otherwise train.val_subset (the dry run sets it).
    # The first rows of the val manifest are whole images (4 rows per image), so a multiple of 4 keeps all classes equal.
    if cfg.get("trial_val_subset"):
        tc["val_subset"] = int(cfg["trial_val_subset"])
    if tc.get("val_subset") and int(tc["val_subset"]) % 4 != 0:
        raise ValueError(f"trial_val_subset must be a multiple of 4 (whole images), got {tc['val_subset']}")
    tc["val_every_epochs"] = 1                                 # the pruner and the collapse check need every epoch
    tc["epoch0_validation"] = False                            # epoch 0 is logged in the final training only (B14)
    tc["checkpoint_every_minutes"] = 1e9                       # trial runs only save at epoch ends
    tc["sample_every_epochs"] = 10 ** 6                        # no sample grids during trials (except the last epoch)
    tc["max_steps"] = tc["pause_after_steps"] = None
    desc = f"{cfg['study']}-trial{trial_number}"
    tcfg["run_id"] = make_run_id(cfg["device"], desc, smoke=bool(cfg["run"].get("smoke")))
    tcfg["run"] = dict(cfg["run"], desc=desc)
    # Trial runs go to <output_root>/trials/runs/task3/..., so resume="auto" of the final training (which looks in
    # <output_root>/runs/task3 and <persist_root>/runs/task3) never picks up a trial checkpoint (D27, D29).
    tcfg["output_root"] = tcfg["persist_root"] = str(resolve(cfg["output_root"]) / "trials")
    return tcfg


# ------------------------------------------------------------------------------------ study
def run_study(cfg: dict, on_checkpoint=None) -> Path:
    """Run (or continue) the Task 3 study; returns the folder holding trials.csv, the plots and study_run.json."""
    cfg = deep_copy(cfg)
    _check_budget(cfg)
    _check_search_space(cfg["tuned_params"])
    n_trials = int(cfg["n_trials"])
    timeout_s = None if cfg["timeout_minutes"] is None else float(cfg["timeout_minutes"]) * 60

    # The Task 2 checkpoints are located and hash-checked ONCE here (a wrong file stops the study before trial 0);
    # every trial checks them again and rebuilds the experts and the gate from the originals.
    sources = find_sources(cfg)
    verify_sources(sources, expected_hashes(cfg))

    db_path = resolve(cfg["storage"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # J: lower is better. open_study adds the Task 1 fixes: sampler seed + number of existing trials, and trials
    # left RUNNING by a killed process are marked FAIL (runs.py).
    study = runs.open_study(cfg, "minimize", f"sqlite:///{db_path.as_posix()}")

    # Trial 0 = the PDF's starting values, only for a study that has no trial yet (a restart must not enqueue again).
    if cfg.get("enqueue_pdf_start", True) and len(study.trials) == 0:
        tuned = {spec["name"] for spec in cfg["tuned_params"]}
        study.enqueue_trial({k: v for k, v in PDF_START.items() if k in tuned})

    def objective(trial: optuna.Trial) -> float:
        values = suggest_params(trial, cfg["tuned_params"])
        tcfg = make_trial_config(cfg, values, trial.number)
        trial.set_user_attr("run_id", tcfg["run_id"])
        history = []                                           # val J after every epoch (warm-up epochs first)

        def report(step: int, val_J: float, flags: dict) -> None:
            """Called by _train after each finished epoch's validation."""
            trial.report(val_J, step=step)
            history.append(val_J)
            trial.set_user_attr("val_J_per_epoch", list(history))
            trial.set_user_attr("mean_w_by_class", flags["mean_w_by_class"])      # of the last validation
            if flags["collapsed"]:                             # routing collapse: prune from the first validation on
                trial.set_user_attr("pruned_reason", "collapse")
                trial.set_user_attr("collapse_reasons", flags["reasons"])
                raise optuna.TrialPruned()
            if trial.should_prune():                           # MedianPruner
                trial.set_user_attr("pruned_reason", "median")
                raise optuna.TrialPruned()

        run_dir, best_J = _train(tcfg, report_fn=report, sources=sources)
        if not cfg.get("trial_keep_checkpoints"):              # a trial only needs its J: save disk space
            for ckpt in run_dir.glob("ckpt_*.pt"):
                ckpt.unlink()
        if best_J is None:
            raise ValueError("a trial finished without any eligible joint-stage validation")
        return best_J

    def after_trial(study_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Copy the DB to the persistent folder (Drive on Colab) and notify the caller."""
        persist = resolve(cfg["persist_root"]) / "optuna" / db_path.name
        if persist.resolve() != db_path.resolve():
            persist.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(db_path, persist)
        if on_checkpoint is not None:
            on_checkpoint(db_path)

    # The budget counts trials that gave an answer (COMPLETE or PRUNED); a FAILED trial is retried (D19).
    already = runs.n_answered(study)
    started = time.monotonic()
    study.optimize(objective, n_trials=max(n_trials - already, 0), timeout=timeout_s, callbacks=[after_trial])
    elapsed = time.monotonic() - started

    out_dir = resolve(cfg["study_dir"])
    export_study(study, out_dir)
    write_run_info(study, out_dir, cfg, already, elapsed)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"study {cfg['study']}: {len(done)} complete / {len(study.trials)} total trials; exported to {out_dir}")
    if done:
        print(f"best J {study.best_value:.5f} with {study.best_params}")
    return out_dir


def write_run_info(study: optuna.Study, out_dir: Path, cfg: dict, answered_before: int, elapsed: float) -> Path:
    """study_run.json: counts per state, pruned reasons and whether the timeout ended this session (for the report)."""
    states = [t.state.name for t in study.trials]
    reasons = [t.user_attrs.get("pruned_reason") for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    answered = runs.n_answered(study)
    timeout_s = None if cfg["timeout_minutes"] is None else float(cfg["timeout_minutes"]) * 60
    info = {
        "study": cfg["study"], "n_trials_requested": int(cfg["n_trials"]), "timeout_minutes": cfg["timeout_minutes"],
        "epochs_per_trial": cfg["epochs_per_trial"], "trial_val_subset": cfg.get("trial_val_subset"),
        "trials_total": len(states), "complete": states.count("COMPLETE"), "pruned": states.count("PRUNED"),
        "fail": states.count("FAIL"), "answered": answered, "answered_before_this_session": answered_before,
        "pruned_reasons": {r: reasons.count(r) for r in sorted({str(r) for r in reasons})},
        "timeout_hit": bool(timeout_s is not None and answered < int(cfg["n_trials"]) and elapsed >= timeout_s),
        "elapsed_seconds_this_session": round(elapsed, 1),
    }
    path = out_dir / "study_run.json"
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return path


# ------------------------------------------------------------------------------------ final config
def write_final_config(cfg: dict, study: optuna.Study, out_path) -> Path:
    """Write a FULL training config that uses the best trial of the study.

    cfg      : the base config (its train.warmup_epochs / joint_epochs are the FINAL budget and stay as they are)
    study    : the finished study (best trial = lowest J among COMPLETE trials)
    out_path : where to write the YAML (relative paths are relative to the repository root)
    Device values (paths, workers) are left out: they are merged in again from the device profile.
    Block `final_config_source` records where the values came from, including the PDF-start trial 0.
    """
    final = deep_copy(cfg)
    for key in DEVICE_KEYS:
        final.pop(key, None)
    try:
        best = study.best_trial
    except ValueError:
        raise ValueError(f"study {study.study_name!r} has no completed trial yet") from None

    apply_values(final, best.params)                           # same name -> config map as the study
    first = study.trials[0] if study.trials else None
    final["final_config_source"] = {
        "study": study.study_name,
        "best_trial": best.number,
        "best_value": best.value,
        "best_params": dict(best.params),
        "best_val_J_per_epoch": best.user_attrs.get("val_J_per_epoch"),
        "best_mean_w_by_class": best.user_attrs.get("mean_w_by_class"),
        "trial_0": None if first is None else {"state": first.state.name, "value": first.value,
                                                "params": dict(first.params)},
    }
    out = resolve(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ("# Final Task 3 config written by genai.tasks.task3.tune.write_final_config (best trial of the study).\n"
              "# Train with:  python scripts/train.py --task t3 --config <this file>\n")
    out.write_text(header + yaml.safe_dump(final, sort_keys=False), encoding="utf-8")
    print(f"final config written to {out} (best trial {best.number}, J {best.value:.5f})")
    return out


# ------------------------------------------------------------------------------------ dry run
def dry_run_overrides(cfg: dict, tmp_root: Path) -> dict:
    """Make a config safe for a dry run: distinct study name, tmp storage / run / export folders, smoke runs,
    2 trials of 1 warm-up + 1 joint epoch. The real artifacts/optuna/t3_moe.db and studies/ are never touched.

    trial_val_subset is cleared so that train.val_subset (set by scripts/tune.py --dry-run) decides the validation rows.
    """
    cfg = deep_copy(cfg)
    cfg["study"] = cfg["study"] + "_dryrun"
    cfg["storage"] = str(tmp_root / "optuna" / f"{cfg['study']}.db")
    cfg["study_dir"] = str(tmp_root / "studies" / cfg["study"])
    cfg["output_root"] = str(tmp_root)
    cfg["persist_root"] = str(tmp_root)
    cfg["run"] = dict(cfg["run"], smoke=True)
    cfg["n_trials"] = 2
    cfg["epochs_per_trial"] = {"warmup": 1, "joint": 1}
    cfg["trial_val_subset"] = None
    return cfg
