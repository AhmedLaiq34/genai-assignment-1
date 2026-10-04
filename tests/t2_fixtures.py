"""Helper for the Task 2 routing / ONNX tests: RANDOMLY INITIALISED fixture checkpoints.

make_fixture_checkpoints(out_dir) writes four checkpoints with exactly the layout real training
produces (genai.common.checkpoint.build_checkpoint + save_checkpoint, config with component, model,
specialist, run.smoke, run_id ending in _smoke):

    classifier.pt  salt.pt  blur.pt  occlusion.pt

The weights are random, so every number computed from them is meaningless. They only prove that the
code paths (loading, routing, evaluation, ONNX export) work. Used by the tests (tmp dirs) and, with
bigger models, for the manual integration run in artifacts/fixtures/task2/.
"""
from __future__ import annotations

from pathlib import Path

import torch

from genai.common.checkpoint import build_checkpoint, save_checkpoint
from genai.models.autoencoder import UniversalAE
from genai.models.classifier import CorruptionClassifier
from genai.tasks.task2 import SPECIALIST_COND_ID

# Small models keep the tests fast on the CPU.
TINY_CLASSIFIER = {"channels": [4, 8, 8, 8], "dropout": 0.1, "num_classes": 4}
TINY_AE = {"in_ch": 3, "base_channels": 4, "depth": 4, "bottleneck_dim": 16, "dropout": 0.0}


def _warm_up(model: torch.nn.Module) -> None:
    """A few training-mode passes so BatchNorm running statistics are not the trivial defaults."""
    model.train()
    with torch.no_grad():
        for _ in range(3):
            model(torch.rand(4, 3, 128, 128))
    model.eval()


def write_checkpoint(path, model, config: dict, step: int = 3) -> Path:
    """Save `model` as a contract checkpoint at `path`."""
    save_checkpoint(path, build_checkpoint(model, config=config, global_step=step))
    return Path(path)


def make_fixture_checkpoints(out_dir, classifier_cfg=None, ae_cfg=None, seed: int = 0) -> dict:
    """Write the four fixture checkpoints into out_dir. Returns {"classifier": path, "salt": ..., ...}."""
    classifier_cfg, ae_cfg = classifier_cfg or TINY_CLASSIFIER, ae_cfg or TINY_AE
    out_dir = Path(out_dir)
    torch.manual_seed(seed)
    paths = {}

    classifier = CorruptionClassifier.from_config(classifier_cfg)
    _warm_up(classifier)
    paths["classifier"] = write_checkpoint(
        out_dir / "classifier.pt", classifier,
        {"component": "classifier", "model": classifier_cfg, "run": {"smoke": True},
         "run_id": "20260101-0000_test_t2cls_smoke"})

    for name, cond_id in SPECIALIST_COND_ID.items():
        specialist = UniversalAE.from_config(ae_cfg)
        _warm_up(specialist)
        paths[name] = write_checkpoint(
            out_dir / f"{name}.pt", specialist,
            {"component": "specialist", "model": ae_cfg,
             "specialist": {"corruption": name, "cond_id": cond_id},
             "run": {"smoke": True}, "run_id": f"20260101-0000_test_t2spec_{name}_smoke"})
    return paths
