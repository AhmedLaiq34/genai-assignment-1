"""Task 2 Optuna entry point (plan 10.1): a thin dispatcher on cfg["component"].

    component: classifier  -> classifier.study_classifier   (study t2_classifier)
    component: specialist  -> specialist.study_specialists  (study t2_specialist_shared)

The specialist study trains all three corruptions in every trial (PDF page 5 allows one shared
search), so it does not use specialist.corruption.
"""
from __future__ import annotations

from pathlib import Path

from genai.tasks.task2.runs import get_component


def run_study(cfg: dict, on_checkpoint=None) -> Path:
    """Run (or continue) the Task 2 study of this config. Returns the study directory."""
    if get_component(cfg) == "classifier":
        from genai.tasks.task2.classifier import study_classifier
        return study_classifier(cfg, on_checkpoint=on_checkpoint)
    from genai.tasks.task2.specialist import study_specialists
    return study_specialists(cfg, on_checkpoint=on_checkpoint)
