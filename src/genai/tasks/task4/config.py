"""Config helpers for Task 4 (train / tune / evaluate / scripts). Plan: docs/TASK4_PLAN.md section D.

The generic helpers are imported from Task 1 (not copied); only the Task 4 specific parts live here.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from genai.common.paths import ROOT
from genai.tasks.task1.config import (  # noqa: F401  (re-exported for the other Task 4 modules)
    deep_copy, is_tbd, load_config, make_run_id, resolve, set_dotted,
)

DEFAULT_CONFIG = "configs/task4_cgan.yaml"
FINAL_CONFIG = "configs/task4_final.yaml"

# Where each tuned parameter (names exactly as in the Colab log) goes inside the config.
PARAM_TARGETS = {
    "lr_g": "train.lr_g",
    "lr_d": "train.lr_d",
    "batch_size": "train.batch_size",
    "base_channels": "model.base_channels",
    "dropout": "model.dropout",
    "style_dim": "model.style_dim",
    "lambda_l1": "train.lambda_l1",
}


def runs_dir(cfg: dict, root_key: str = "output_root", kind: str = "task4") -> Path:
    """<root>/runs/task4 (final and smoke runs) or <root>/runs/task4_trials (Optuna trials).

    The two kinds are kept apart so that resume="auto" can never pick up a trial checkpoint (plan C20).
    """
    if kind not in ("task4", "task4_trials"):
        raise ValueError(f"unknown run kind {kind!r}")
    return resolve(cfg[root_key]) / "runs" / kind


def find_latest_checkpoint(cfg: dict) -> Path | None:
    """Newest ckpt_last.pt of a final/smoke run under persist_root, output_root and cfg['resume_roots'].

    resume_roots is a list of extra folders (for example the previous Kaggle version's output attached
    as an input). Smoke runs (folder name ends with _smoke) are skipped unless run.smoke is true.
    """
    roots = []
    for key in ("persist_root", "output_root"):
        if key in cfg:
            roots.append(runs_dir(cfg, key))
    for extra in cfg.get("resume_roots") or []:
        # an extra root may be the runs folder itself or the folder that contains runs/task4
        extra = resolve(extra)
        roots += [extra, extra / "runs" / "task4"]
    candidates = []
    for root in roots:
        candidates += list(root.glob("*/ckpt_last.pt"))
        candidates += list(root.glob("**/runs/task4/*/ckpt_last.pt"))   # nested Kaggle input folders
    smoke = bool(cfg.get("run", {}).get("smoke"))
    candidates = [p for p in candidates if smoke or not p.parent.name.endswith("_smoke")]
    return max(candidates, key=lambda p: p.stat().st_mtime, default=None)


def write_final_config(best_params_json, base_config: str = DEFAULT_CONFIG,
                       out_path: str = FINAL_CONFIG) -> Path:
    """Copy the best study parameters into a full training config (train.epochs stays TBD).

    best_params_json is the best_params.json written by tune.print_study_summary.
    """
    import json
    best = json.loads(Path(best_params_json).read_text(encoding="utf-8"))["params"]
    base = resolve(base_config)
    cfg = yaml.safe_load(base.read_text(encoding="utf-8"))
    for key in ("study", "storage", "study_dir", "n_trials", "epochs_per_trial", "sampler", "pruner",
                "tuned_params", "objective", "trial_train_subset", "search_space_status"):
        cfg.pop(key, None)                               # study-only keys do not belong in a training config
    for name, value in best.items():
        set_dotted(cfg, PARAM_TARGETS[name], value)
    cfg["train"]["epochs"] = "TBD_AFTER_BENCHMARK"
    cfg["train"]["schedule"] = "linear_decay_half"       # plan C10: the final run decays the lr in the 2nd half
    cfg["source"] = f"best parameters from {Path(best_params_json).as_posix()}"
    out = resolve(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out
