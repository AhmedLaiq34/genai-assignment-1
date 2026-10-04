"""Run-directory and config helpers shared by the Task 2 classifier and specialist code.

Task 1's helpers in genai.tasks.task1.config hard-code the folder `runs/task1`; Task 2 has one
folder per component, so this file provides the same three helpers for Task 2:

    artifacts/runs/task2_classifier/<run_id>/
    artifacts/runs/task2_specialist_<corruption>/<run_id>/
"""
from __future__ import annotations

from pathlib import Path

from genai.tasks.task1.config import resolve
from genai.tasks.task2 import SPECIALIST_COND_ID

COMPONENTS = ("classifier", "specialist")
TRACKER_GROUP = "task2"


def get_component(cfg: dict) -> str:
    """cfg['component'] must be 'classifier' or 'specialist' (set in the two Task 2 YAML files)."""
    component = cfg.get("component")
    if component not in COMPONENTS:
        raise ValueError(f"cfg['component'] must be one of {COMPONENTS}, got {component!r}")
    return component


def get_corruption(cfg: dict) -> str:
    """Which specialist this run trains: 'salt', 'blur' or 'occlusion'.

    It is chosen on the command line / in the notebook OVERRIDES:  specialist.corruption=salt
    A missing or invalid value is an error (there is no sensible default: three different
    models are trained from the same config).
    """
    corruption = (cfg.get("specialist") or {}).get("corruption")
    if corruption not in SPECIALIST_COND_ID:
        raise ValueError(
            f"specialist.corruption must be one of {sorted(SPECIALIST_COND_ID)}, got {corruption!r}. "
            f"Pass it as an override, e.g. --set specialist.corruption=salt")
    return corruption


def run_group(cfg: dict) -> str:
    """Folder name under runs/ for this run: task2_classifier or task2_specialist_<corruption>."""
    if get_component(cfg) == "classifier":
        return "task2_classifier"
    return f"task2_specialist_{get_corruption(cfg)}"


def run_desc(cfg: dict) -> str:
    """Short description used in run_id: t2cls or t2spec_<corruption>."""
    return "t2cls" if get_component(cfg) == "classifier" else f"t2spec_{get_corruption(cfg)}"


def runs_dir(cfg: dict, root_key: str = "output_root") -> Path:
    """<output_root>/runs/<run_group>  (root_key='persist_root' gives the persistent copy)."""
    return resolve(cfg[root_key]) / "runs" / run_group(cfg)


def open_study(cfg: dict, direction: str, storage: str):
    """Create (or reopen) the Optuna study of this config, with the Task 1 robustness fixes (D19, D42).

    * sampler seed = cfg seed + number of trials already in the DB: a fresh study is reproducible
      (seed 42) and a RESUMED study does not re-draw the first random parameters it already tried;
    * only one process ever writes a study (CONTRACTS 3.13), so a trial still RUNNING at start-up
      belongs to a process that was killed: it is marked FAIL instead of staying RUNNING forever.
    """
    import optuna

    try:
        n_existing = len(optuna.load_study(study_name=cfg["study"], storage=storage).trials)
    except KeyError:                                          # the study does not exist yet
        n_existing = 0
    ps = cfg["pruner"]
    study = optuna.create_study(
        study_name=cfg["study"],
        storage=storage,
        load_if_exists=True,                                  # restart-safe
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=int(cfg["sampler"]["seed"]) + n_existing),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=ps["n_startup_trials"],
                                           n_warmup_steps=ps["n_warmup_steps"]),
    )
    for t in study.trials:
        if t.state == optuna.trial.TrialState.RUNNING:
            study.tell(t.number, state=optuna.trial.TrialState.FAIL)
            print(f"trial {t.number} was left RUNNING by an interrupted process -> marked FAIL")
    return study


def n_answered(study) -> int:
    """Trials that produced an answer (COMPLETE or PRUNED). A FAILED trial (crash) says nothing about its
    parameters, so it does not use up the budget: the study runs another trial instead."""
    import optuna
    answered = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    return len([t for t in study.trials if t.state in answered])


def find_latest_checkpoint(cfg: dict) -> Path | None:
    """Newest ckpt_last.pt of THIS component (resume='auto'). Same rule as Task 1, own folder."""
    candidates = []
    for key in ("persist_root", "output_root"):
        if key in cfg:
            candidates += list(runs_dir(cfg, key).glob("*/ckpt_last.pt"))
    candidates = [p for p in candidates if not p.parent.name.endswith("_smoke") or cfg.get("run", {}).get("smoke")]
    return max(candidates, key=lambda p: p.stat().st_mtime, default=None)
