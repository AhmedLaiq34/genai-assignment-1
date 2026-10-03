"""Task 1 Optuna study. Implements CONTRACTS 3.6 (fixed objective J) and 3.13.

Entry point:  run_study(cfg, on_checkpoint=None) -> study directory (Path)

* Study `t1_universal` tunes: lr, batch size, bottleneck dim, encoder channels, dropout, alpha.
* Every trial trains the model with genai.tasks.task1.train and returns the BEST validation
  J (= 0.5*L1 + 0.5*(1-SSIM) on the val manifest). J does not depend on alpha, so trials with
  different alpha are comparable.
* MedianPruner stops unpromising trials after an epoch (only useful when a trial has >1 epoch).
* Storage: SQLite file (cfg["storage"]), load_if_exists=True, so a restarted session continues
  the same study. After every trial the DB is copied to the persistent folder and
  on_checkpoint(db_path) is called.
* Budgets come from the config: n_trials and epochs_per_trial stay TBD_AFTER_BENCHMARK until
  measured. They can be overridden for tests / dry runs through cfg["dry_run"] or the CLI.
* This module never reads the test set.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import optuna

from genai.tasks.task1.config import deep_copy, is_tbd, make_run_id, resolve, set_dotted
from genai.tasks.task1.train import _train

# Where each tuned parameter goes inside the config (the names are the ones in tuned_params).
PARAM_TARGETS = {
    "lr": "train.lr",
    "batch": "train.batch_size",
    "bottleneck_dim": "model.bottleneck_dim",
    "encoder_channels": "model.base_channels",
    "dropout": "model.dropout",
    "alpha": "train.alpha",
}


def suggest_params(trial: optuna.Trial, space: list) -> dict:
    """Ask Optuna for one value per entry of cfg['tuned_params']."""
    values = {}
    for spec in space:
        name = spec["name"]
        if spec["type"] == "float":
            values[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
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
    tcfg["train"]["sample_every_epochs"] = 10 ** 6       # no sample grids during trials
    tcfg["run_id"] = make_run_id(cfg["device"], f"{cfg['study']}-trial{trial_number}",
                                  smoke=bool(cfg["run"].get("smoke")))
    tcfg["run"] = dict(cfg["run"], desc=f"{cfg['study']}-trial{trial_number}")
    return tcfg


def _storage_path(cfg: dict) -> Path:
    return resolve(cfg["storage"])


def export_study(study: optuna.Study, out_dir: Path) -> None:
    """trials.csv plus three plots (history, parameter importance, parallel coordinates)."""
    import matplotlib
    matplotlib.use("Agg")                       # no window; we only save files
    import matplotlib.pyplot as plt
    from optuna.visualization import matplotlib as ovm

    out_dir.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(out_dir / "trials.csv", index=False)
    plots = {"optimization_history.png": ovm.plot_optimization_history,
             "param_importances.png": ovm.plot_param_importances,
             "parallel_coordinate.png": ovm.plot_parallel_coordinate}
    for filename, plot_fn in plots.items():
        try:
            ax = plot_fn(study)
            fig = ax.figure if hasattr(ax, "figure") else ax[0].figure
            fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as err:                # e.g. importance needs >= 2 finished trials
            (out_dir / (filename + ".skipped.txt")).write_text(f"not drawn: {err}\n", encoding="utf-8")


def run_study(cfg: dict, on_checkpoint=None) -> Path:
    """Run (or continue) the Task 1 study; returns the folder holding trials.csv and plots."""
    cfg = deep_copy(cfg)
    n_trials, epochs = cfg["n_trials"], cfg["epochs_per_trial"]
    if is_tbd(n_trials) or is_tbd(epochs):
        raise ValueError("n_trials / epochs_per_trial are TBD_AFTER_BENCHMARK: measure first "
                         "(scripts/benchmark.py) or pass explicit values (--n-trials / --epochs)")

    db_path = _storage_path(cfg)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    ps = cfg["pruner"]
    study = optuna.create_study(
        study_name=cfg["study"],
        storage=f"sqlite:///{db_path.as_posix()}",
        load_if_exists=True,                                  # restart-safe
        direction="minimize",                                 # J: lower is better
        sampler=optuna.samplers.TPESampler(seed=cfg["sampler"]["seed"]),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=ps["n_startup_trials"],
                                           n_warmup_steps=ps["n_warmup_steps"]),
    )

    def objective(trial: optuna.Trial) -> float:
        values = suggest_params(trial, cfg["tuned_params"])
        tcfg = make_trial_config(cfg, values, trial.number)
        trial.set_user_attr("run_id", tcfg["run_id"])

        def report(epoch: int, val_J: float) -> None:        # called after each finished epoch
            trial.report(val_J, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        _run_dir, best_J = _train(tcfg, report_fn=report)
        return best_J

    def after_trial(study_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Copy the DB to the persistent folder (Drive on Colab) and notify the caller."""
        persist = resolve(cfg["persist_root"]) / "optuna" / db_path.name
        if persist.resolve() != db_path.resolve():
            persist.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(db_path, persist)
        if on_checkpoint is not None:
            on_checkpoint(db_path)

    already = len([t for t in study.trials if t.state.is_finished()])
    study.optimize(objective, n_trials=max(int(n_trials) - already, 0), callbacks=[after_trial])

    out_dir = resolve(cfg["study_dir"])
    export_study(study, out_dir)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"study {cfg['study']}: {len(done)} complete / {len(study.trials)} total trials; exported to {out_dir}")
    if done:
        print(f"best J {study.best_value:.5f} with {study.best_params}")
    return out_dir


def dry_run_overrides(cfg: dict, tmp_root: Path) -> dict:
    """Make a config safe for a dry run: distinct study name, tmp storage / run / export folders,
    tiny data. The real artifacts/optuna/t1_universal.db and studies/ are never touched."""
    cfg = deep_copy(cfg)
    cfg["study"] = cfg["study"] + "_dryrun"
    cfg["storage"] = str(tmp_root / "optuna" / f"{cfg['study']}.db")
    cfg["study_dir"] = str(tmp_root / "studies" / cfg["study"])
    cfg["output_root"] = str(tmp_root)
    cfg["persist_root"] = str(tmp_root)
    cfg["run"] = dict(cfg["run"], smoke=True)
    return cfg
