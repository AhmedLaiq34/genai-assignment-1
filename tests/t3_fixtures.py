"""Helper for the Task 3 tests: RANDOMLY INITIALISED fixture source checkpoints.

make_t3_sources(tmp) writes, into one folder, the files the Task 3 code looks for (the names are the
values of genai.tasks.task3.SOURCE_FILES):

    t2_classifier.pt  t2_ae_salt.pt  t2_ae_blur.pt  t2_ae_occlusion.pt   (from t2_fixtures, tiny models)
    t1_universal_ae.pt                                                     (a tiny Task 1 autoencoder)

The weights are random, so every number computed from them is meaningless; they only prove that the
code paths (loading, hashing, training, evaluation, ONNX export, app) work. All five checkpoints carry
run.smoke = True, so a model built from them can never be promoted.

Usage in a test:

    paths, sha = make_t3_sources(tmp_path / "sources")
    model = SoftMoE.load_from_task2(paths, tau=1.0, expected_sha256=sha)
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from genai.common.checkpoint import sha256_file
from genai.models.autoencoder import UniversalAE
from genai.tasks.task3 import SOURCE_FILES
from t2_fixtures import TINY_AE, TINY_CLASSIFIER, _warm_up, make_fixture_checkpoints, write_checkpoint


def make_t3_sources(tmp, classifier_cfg=None, ae_cfg=None, seed: int = 0) -> tuple:
    """Write the four Task 2 fixture checkpoints and a tiny Task 1 checkpoint into folder `tmp`.

    Returns (paths, sha):
      paths = {"classifier": Path, "salt": Path, "blur": Path, "occlusion": Path, "t1": Path}
              (the files are named like the real ones, SOURCE_FILES)
      sha   = {"classifier": hex, "salt": hex, "blur": hex, "occlusion": hex}   (sha256 of the four
              Task 2 files; the same layout as TASK2_SHA256, so tests pass expected_sha256=sha.
              The T1 hash is not in it: use sha256_file(paths["t1"]) if a test needs it.)
    """
    tmp = Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    ae_cfg = ae_cfg or TINY_AE

    # t2_fixtures names its files classifier.pt / salt.pt / ...; rename them to the real file names.
    made = make_fixture_checkpoints(tmp, classifier_cfg or TINY_CLASSIFIER, ae_cfg, seed=seed)
    paths = {}
    for name, old in made.items():
        new = tmp / SOURCE_FILES[name]
        os.replace(old, new)
        paths[name] = new

    # A tiny Task 1 autoencoder (optional comparison column in Task 3). Its own seed stream.
    torch.manual_seed(seed + 1000)
    t1_model = UniversalAE.from_config(ae_cfg)
    _warm_up(t1_model)
    paths["t1"] = write_checkpoint(
        tmp / SOURCE_FILES["t1"], t1_model,
        {"model": ae_cfg, "run": {"smoke": True}, "run_id": "20260101-0000_test_t1_smoke"})

    sha = {name: sha256_file(paths[name]) for name in ("classifier", "salt", "blur", "occlusion")}
    return paths, sha
