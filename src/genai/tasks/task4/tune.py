"""Task 4 Optuna study (style-conditioned face-to-sketch cGAN). Plan: docs/TASK4_PLAN.md B.1.

Entry point:  run_study(cfg, on_checkpoint=None) -> study directory (Path)

* Study `task4_cgan` tunes: lr_g, lr_d, batch_size, base_channels, dropout, style_dim, lambda_l1.
* Every trial trains the cGAN with genai.tasks.task4.train._train and returns the BEST validation L1
  of the trial (L1 between G(photo, true style) and the true sketch, both rescaled to [0,1]).
* After every epoch the trial reports its val L1; the MedianPruner may stop it early.
* Storage: SQLite file (cfg["storage"]), load_if_exists=True, so a restarted session continues the
  same study. After every trial the DB is copied to the persistent folder and on_checkpoint(db) is called.
* Budgets come from the config: n_trials stays TBD_AFTER_BENCHMARK until the benchmark has measured it.
  The dry run (dry_run_overrides) never touches the real study file.
* Trial runs are written to <output_root>/runs/task4_trials/<run_id>/ (config key run.trial: true is what
  train.py looks at), so resume="auto" of the final run can never pick up a trial checkpoint.
* This module never reads the test set.

The same file also holds print_study_summary, which log_study.py (the study rebuilt from the Colab
console log) reuses.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import optuna

from genai.tasks.task1.tune import export_study            # imported, not copied: trials.csv + 3 plots
from genai.tasks.task4.config import PARAM_TARGETS, deep_copy, is_tbd, make_run_id, resolve, runs_dir, set_dotted
from genai.tasks.task4.train import _train

COMPLETE = optuna.trial.TrialState.COMPLETE
PRUNED = optuna.trial.TrialState.PRUNED
FAIL = optuna.trial.TrialState.FAIL
RUNNING = optuna.trial.TrialState.RUNNING


def suggest_params(trial: optuna.Trial, space: list) -> dict:
    """Ask Optuna for one value per entry of cfg['tuned_params']."""
    values = {}
    for spec in space:
        name = spec["name"]
        if spec["type"] == "float":
            values[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
        elif spec["type"] == "int":
            values[name] = trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
        elif spec["type"] == "categorical":
            values[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"unknown parameter type {spec['type']!r} for {name}")
    return values


def make_trial_config(cfg: dict, values: dict, trial_number: int) -> dict:
    """Copy of the base config with the sampled values and the trial's epoch budget inserted."""
    tcfg = deep_copy(cfg)
    for name, value in values.items():
        set_dotted(tcfg, PARAM_TARGETS[name], value)
    tcfg["train"]["epochs"] = cfg["epochs_per_trial"]
    if cfg.get("trial_train_subset"):
        tcfg["train"]["train_subset"] = cfg["trial_train_subset"]
    tcfg["train"]["checkpoint_every_minutes"] = 1e9      # trial runs only save at epoch ends
    tcfg["train"]["sample_every_epochs"] = 0             # 0 = no sample grids (train.py draws one at epoch 1 otherwise)
    tcfg["train"]["snapshot_every_epochs"] = 0           # no ckpt_epochNNN.pt files
    tcfg["run_id"] = make_run_id(cfg["device"], f"{cfg['study']}-trial{trial_number}",
                                  smoke=bool(cfg["run"].get("smoke")))
    # run.trial: true makes train.py use runs/task4_trials instead of runs/task4
    tcfg["run"] = dict(cfg["run"], desc=f"{cfg['study']}-trial{trial_number}", trial=True)
    return tcfg


def _trial_run_stats(tcfg: dict) -> dict:
    """best_epoch and val SSIM at the best epoch, read from the trial run's metrics.jsonl.
    Returns {} if the file is missing (for example the trial crashed before the first epoch ended)."""
    path = runs_dir(tcfg, kind="task4_trials") / tcfg["run_id"] / "metrics.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r.get("val/l1") is not None]
    if not rows:
        return {}
    best = min(rows, key=lambda r: r["val/l1"])
    stats = {"best_epoch": best["epoch"]}
    if best.get("val/ssim") is not None:
        stats["val_ssim_at_best"] = best["val/ssim"]
    return stats


def print_study_summary(study: optuna.Study, study_dir: Path, source: str = "tune.py run_study") -> Path:
    """Print the five-line result block and write best_params.json into study_dir (returns its path)."""
    study_dir = Path(study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    count = lambda state: len([t for t in study.trials if t.state == state])     # noqa: E731
    out = study_dir / "best_params.json"
    out.write_text(json.dumps({
        "study": study.study_name,
        "best_trial": study.best_trial.number,
        "best_val_l1": study.best_value,
        "objective": "val L1 on [0,1] sketches",
        "params": study.best_params,
        "n_trials": {"COMPLETE": count(COMPLETE), "PRUNED": count(PRUNED), "FAIL": count(FAIL)},
        "source": source,
    }, indent=2) + "\n", encoding="utf-8")
    print("=== Task 4 Optuna study complete ===")
    print(f"Trials run: {len(study.trials)}")
    print(f"Best trial: #{study.best_trial.number}  best val L1: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"Saved to {out.as_posix()}")
    return out


def run_study(cfg: dict, on_checkpoint=None) -> Path:
    """Run (or continue) the Task 4 study; returns the folder holding trials.csv, plots, best_params.json."""
    cfg = deep_copy(cfg)
    n_trials, epochs = cfg["n_trials"], cfg["epochs_per_trial"]
    if is_tbd(n_trials) or is_tbd(epochs):
        raise ValueError("n_trials / epochs_per_trial are TBD_AFTER_BENCHMARK: measure first "
                         "(scripts/benchmark.py) or pass explicit values (--n-trials / --epochs)")

    # Optuna's own lines ("Using an existing study...", "Trial N finished...", "Trial N pruned.") must show.
    optuna.logging.set_verbosity(optuna.logging.INFO)

    db_path = resolve(cfg["storage"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{db_path.as_posix()}"
    ps = cfg["pruner"]

    # How many trials the DB already holds (0 for a new study). Used for the sampler seed below.
    try:
        n_existing = len(optuna.load_study(study_name=cfg["study"], storage=storage).trials)
    except KeyError:                                          # study does not exist yet
        n_existing = 0

    study = optuna.create_study(
        study_name=cfg["study"],
        storage=storage,
        load_if_exists=True,                                  # restart-safe
        direction="minimize",                                 # val L1: lower is better
        # seed + number of existing trials: a fresh study is reproducible (seed 42), and a RESUMED
        # study does not re-draw the same first random parameters it already tried (as in Task 1, D19).
        sampler=optuna.samplers.TPESampler(seed=int(cfg["sampler"]["seed"]) + n_existing),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=ps["n_startup_trials"],
                                           n_warmup_steps=ps["n_warmup_steps"]),
    )

    # Only one process ever writes this study, so a trial still marked RUNNING at start-up belongs to a
    # process that was killed or crashed. Mark it FAIL so it is not left dangling.
    for t in study.trials:
        if t.state == RUNNING:
            study.tell(t.number, state=FAIL)
            print(f"[trial {t.number}] was left RUNNING by an interrupted process -> marked FAIL")

    def objective(trial: optuna.Trial) -> float:
        values = suggest_params(trial, cfg["tuned_params"])
        tcfg = make_trial_config(cfg, values, trial.number)
        trial.set_user_attr("run_id", tcfg["run_id"])
        print(f"[trial {trial.number}] {epochs} epochs with {values}")

        def report(epoch: int, val_l1: float) -> None:       # called after each finished epoch
            trial.report(val_l1, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        start = time.monotonic()
        try:
            _run_dir, best_l1 = _train(tcfg, report_fn=report)
        finally:                                              # also for pruned / crashed trials
            trial.set_user_attr("seconds", round(time.monotonic() - start, 1))
            for key, value in _trial_run_stats(tcfg).items():
                trial.set_user_attr(key, value)
        print(f"[trial {trial.number}] best val L1 {best_l1:.4f}")
        return best_l1

    def after_trial(study_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Copy the DB to the persistent folder (Drive on Colab / Kaggle output) and notify the caller."""
        persist = resolve(cfg["persist_root"]) / "optuna" / db_path.name
        if persist.resolve() != db_path.resolve():
            persist.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(db_path, persist)
        if on_checkpoint is not None:
            on_checkpoint(db_path)

    # The budget counts trials that produced an answer (COMPLETE or PRUNED). A FAILED trial (crash,
    # network error) is not an answer about its parameters, so it does not use up the budget and the
    # study runs another trial instead (D19). Failed trials stay in the DB and are reported.
    already = len([t for t in study.trials if t.state in (COMPLETE, PRUNED)])
    study.optimize(objective, n_trials=max(int(n_trials) - already, 0), callbacks=[after_trial])

    out_dir = resolve(cfg["study_dir"])
    if any(t.state == COMPLETE for t in study.trials):
        print_study_summary(study, out_dir)
        export_study(study, out_dir)                          # trials.csv + 3 plots (plots need >= 2 complete trials)
    else:
        print("no trial completed: nothing to summarise")
    return out_dir


def dry_run_overrides(cfg: dict, tmp_root: Path) -> dict:
    """Make a config safe for a dry run: distinct study name, tmp storage / run / export folders,
    smoke run ids. The real artifacts/optuna/task4_cgan.db and studies/ are never touched.
    (Tiny data and the trial count are set by the caller: trial_train_subset, train.val_subset, n_trials.)"""
    cfg = deep_copy(cfg)
    cfg["study"] = cfg["study"] + "_dryrun"
    cfg["storage"] = str(tmp_root / "optuna" / f"{cfg['study']}.db")
    cfg["study_dir"] = str(tmp_root / "studies" / cfg["study"])
    cfg["output_root"] = str(tmp_root)
    cfg["persist_root"] = str(tmp_root)
    cfg["run"] = dict(cfg["run"], smoke=True)
    return cfg
