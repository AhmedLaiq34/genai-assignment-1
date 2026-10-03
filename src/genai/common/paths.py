"""Repo-root-relative path constants. Implements CONTRACTS §2 layout / §3.8."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
SPLITS = DATA / "splits"          # committed
MANIFESTS = DATA / "manifests"    # committed
ARTIFACTS = ROOT / "artifacts"    # ignored
RUNS = ARTIFACTS / "runs"
OPTUNA_WORK = ARTIFACTS / "optuna"
STUDIES = ROOT / "studies"        # committed
MODELS_CKPT = ROOT / "models" / "checkpoints"
MODELS_ONNX = ROOT / "models" / "onnx"
REPORT_FIGS = ROOT / "report" / "figures"
