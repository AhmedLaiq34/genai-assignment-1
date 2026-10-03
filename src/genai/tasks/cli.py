"""Shared helpers for scripts/train.py, tune.py and evaluate.py (they only parse arguments)."""
from __future__ import annotations

import importlib

import yaml

from genai.tasks.task1.config import load_config

# task flag -> (package under genai.tasks, default config file)
TASKS = {
    "t1": ("task1", "configs/task1_universal.yaml"),
    "t2cls": ("task2", "configs/task2_classifier.yaml"),
    "t2spec": ("task2", "configs/task2_specialist.yaml"),
    "t3": ("task3", "configs/task3_moe.yaml"),
    "t4": ("task4", "configs/task4_cgan.yaml"),
}


def parse_overrides(items) -> dict:
    """['train.max_steps=50', 'run.smoke=true'] -> {'train.max_steps': 50, 'run.smoke': True}
    (the value is parsed as YAML, so numbers / true / null work)."""
    out = {}
    for item in items or []:
        key, _, value = item.partition("=")
        out[key.strip()] = yaml.safe_load(value)
    return out


def build_config(task: str, config, device_profile: str, overrides=None) -> dict:
    """Task YAML + configs/devices/<profile>.yaml + command-line overrides."""
    return load_config(config or TASKS[task][1], device_profile, parse_overrides(overrides))


def entry_point(task: str, module: str, function: str):
    """Import genai.tasks.<pkg>.<module>.<function>; raise NotImplementedError if it is missing
    (tasks other than t1 are implemented by other workstreams)."""
    pkg = TASKS[task][0]
    try:
        mod = importlib.import_module(f"genai.tasks.{pkg}.{module}")
        return getattr(mod, function)
    except (ImportError, AttributeError) as err:
        raise NotImplementedError(f"task {task}: {function}() is not implemented yet ({err})") from err
