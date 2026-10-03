"""Config loading and small path helpers for Task 1 (used by train / tune / evaluate / scripts)."""
from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path

import yaml

from genai.common.paths import ROOT

DEFAULT_CONFIG = "configs/task1_universal.yaml"


def load_config(config_path=None, device_profile: str = "local", overrides: dict | None = None) -> dict:
    """Read the task YAML, merge configs/devices/<profile>.yaml on top, apply overrides.

    overrides uses dotted keys, e.g. {"train.max_steps": 50, "run.smoke": True}.
    """
    config_path = Path(config_path or DEFAULT_CONFIG)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    device = yaml.safe_load((ROOT / "configs" / "devices" / f"{device_profile}.yaml").read_text(encoding="utf-8"))
    cfg.update(device)               # device values (paths, workers) win over task values
    cfg["device_profile"] = device_profile
    for dotted, value in (overrides or {}).items():
        set_dotted(cfg, dotted, value)
    return cfg


def set_dotted(cfg: dict, dotted: str, value) -> None:
    """cfg['a']['b'] = value for dotted == 'a.b' (creating missing levels)."""
    *parents, last = dotted.split(".")
    node = cfg
    for key in parents:
        node = node.setdefault(key, {})
    node[last] = value


def resolve(path_like, base: Path = ROOT) -> Path:
    """Relative paths are relative to the repository root; absolute paths stay as they are."""
    p = Path(path_like)
    return p if p.is_absolute() else base / p


def is_tbd(value) -> bool:
    """True for the placeholder string TBD_AFTER_BENCHMARK (budget not measured yet)."""
    return isinstance(value, str) and value.startswith("TBD")


def make_run_id(device: str, desc: str, smoke: bool = False) -> str:
    """<YYYYmmdd-HHMM>_<device>_<shortdesc>, with the suffix _smoke for smoke runs."""
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M')}_{device}_{desc}"
    return run_id + "_smoke" if smoke and not run_id.endswith("_smoke") else run_id


def runs_dir(cfg: dict, root_key: str = "output_root") -> Path:
    """<output_root>/runs/task1  (or persist_root for root_key='persist_root')."""
    return resolve(cfg[root_key]) / "runs" / "task1"


def find_latest_checkpoint(cfg: dict) -> Path | None:
    """Newest ckpt_last.pt under the persistent folder (and the local output folder).

    This is what resume="auto" uses: after a cloud session restarts, the previous run's
    checkpoint is found again without the student typing a path.
    """
    candidates = []
    for key in ("persist_root", "output_root"):
        if key in cfg:
            candidates += list(runs_dir(cfg, key).glob("*/ckpt_last.pt"))
    candidates = [p for p in candidates if not p.parent.name.endswith("_smoke") or cfg.get("run", {}).get("smoke")]
    return max(candidates, key=lambda p: p.stat().st_mtime, default=None)


def deep_copy(cfg: dict) -> dict:
    return copy.deepcopy(cfg)
