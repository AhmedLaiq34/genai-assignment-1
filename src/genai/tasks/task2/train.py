"""Task 2 training entry point (plan 10.1): a thin dispatcher on cfg["component"].

    component: classifier  -> genai.tasks.task2.classifier.train_classifier
    component: specialist  -> genai.tasks.task2.specialist.train_specialist
                              (needs the override specialist.corruption=salt|blur|occlusion)

scripts/train.py --task t2cls and --task t2spec both end up here; the YAML file chosen by the
task flag says which component it is.
"""
from __future__ import annotations

from pathlib import Path

from genai.tasks.task2.runs import get_component, get_corruption


def run_training(cfg: dict, resume: str | None = None, on_checkpoint=None) -> Path:
    """Train one Task 2 component. Returns the run directory."""
    if get_component(cfg) == "classifier":
        from genai.tasks.task2.classifier import train_classifier
        return train_classifier(cfg, resume=resume, on_checkpoint=on_checkpoint)
    get_corruption(cfg)                       # fail early with a clear message if it is missing
    from genai.tasks.task2.specialist import train_specialist
    return train_specialist(cfg, resume=resume, on_checkpoint=on_checkpoint)
