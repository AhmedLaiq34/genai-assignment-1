"""Task 2 evaluation entry point (plan 10.1).

Two ways to call run_evaluation(cfg, checkpoint, final_test):

* checkpoint is None or a dict  -> the full hard-routed system (oracle AND predicted routing).
  The four checkpoints come from the dict {"classifier": path, "salt": path, "blur": path,
  "occlusion": path}; when `checkpoint` is None they are read from cfg["eval"]["checkpoints"]
  (set in the YAML or with --set eval.checkpoints.classifier=<path> ...).
  Implemented in genai.tasks.task2.evaluation.evaluate_task2.
* checkpoint is a single path, component classifier -> the classifier's own validation report
  (accuracy, macro P/R/F1, per class, normalised confusion matrix).
  A single specialist is not evaluated alone: use the four-checkpoint mode, whose oracle table
  shows every specialist on its own corruption.

final_test=False (default) uses the val manifest; the test manifest is opened only with
final_test=True (never done during development).
"""
from __future__ import annotations

from pathlib import Path

from genai.tasks.task2.runs import get_component


def _checkpoint_dict(cfg: dict, checkpoint) -> dict:
    """The four checkpoint paths as a dict; raises a clear error if one is missing."""
    paths = dict(checkpoint) if isinstance(checkpoint, dict) else dict((cfg.get("eval") or {}).get("checkpoints") or {})
    missing = [name for name in ("classifier", "salt", "blur", "occlusion") if not paths.get(name)]
    if missing:
        raise ValueError(f"Task 2 evaluation needs four checkpoints; missing: {missing}. "
                         f"Set them in cfg['eval']['checkpoints'] (--set eval.checkpoints.<name>=<path>).")
    return paths


def run_evaluation(cfg: dict, checkpoint=None, final_test: bool = False) -> Path:
    """Evaluate Task 2 (see module docstring). Returns the output directory."""
    if checkpoint is None or isinstance(checkpoint, dict):
        from genai.tasks.task2.evaluation import evaluate_task2
        return evaluate_task2(cfg, _checkpoint_dict(cfg, checkpoint), final_test=final_test)
    if get_component(cfg) == "classifier":
        from genai.tasks.task2.classifier import evaluate_classifier
        return evaluate_classifier(cfg, str(checkpoint), final_test=final_test)
    raise ValueError("A single specialist checkpoint is not evaluated alone: pass all four checkpoints "
                     "(--set eval.checkpoints.<classifier|salt|blur|occlusion>=<path>) instead of --ckpt.")
