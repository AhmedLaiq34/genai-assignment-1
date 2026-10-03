"""PyTorch vs ONNX Runtime parity check. Implements CONTRACTS 3.9.

verify_parity(name, ckpt_path, onnx_path, n=16) feeds the same n inputs to the PyTorch model and
to onnxruntime (CPU), reports the max and mean absolute difference, and appends one row to
report/tables/onnx_parity.csv. The inputs are val-manifest images covering all 4 conditions
(n/4 each) unless `inputs` is given (tests pass random tensors).
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from genai.common.checkpoint import sha256_file
from genai.common.constants import ONNX_PARITY_TOL_MAX_ABS
from genai.common.paths import ROOT
from genai.export.onnx_export import load_t1_model

PARITY_CSV = ROOT / "report" / "tables" / "onnx_parity.csv"
FIELDS = ["date_utc", "model", "tag", "n_inputs", "max_abs_diff", "mean_abs_diff", "tolerance",
          "passed", "onnx_sha256", "checkpoint"]


def val_inputs(n: int, data_root="local") -> torch.Tensor:
    """n/4 corrupted val images of each of the 4 conditions, from the real val manifest."""
    from genai.pets.dataset import PetsManifestDataset, resolve_data_paths
    paths = resolve_data_paths(data_root)
    ds = PetsManifestDataset(Path(paths["manifests"]) / "pets_val_manifest.jsonl", "val", False, paths)
    per = max(1, n // 4)
    picks = []
    for cond in range(4):
        picks += [i for i, r in enumerate(ds.rows) if r["cond_id"] == cond][:per]
    return torch.stack([ds[i][0] for i in picks])


def verify_parity(name: str, ckpt_path, onnx_path, n: int = 16, inputs=None, tag: str = "",
                  csv_path=None, data_root="local") -> dict:
    """Compare PyTorch and ONNX Runtime outputs. Returns the result dict (also appended to the CSV)."""
    if name != "t1_universal":
        raise NotImplementedError(f"parity check for '{name}' is not implemented yet")
    model, _ckpt = load_t1_model(ckpt_path)
    x = inputs if inputs is not None else val_inputs(n, data_root)

    with torch.no_grad():
        expected = model(x).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = session.run(["output"], {"input": x.numpy().astype(np.float32)})[0]

    diff = np.abs(expected - got)
    result = {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": name, "tag": tag,
        "n_inputs": int(x.shape[0]), "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
        "tolerance": ONNX_PARITY_TOL_MAX_ABS, "passed": bool(diff.max() <= ONNX_PARITY_TOL_MAX_ABS),
        "onnx_sha256": sha256_file(onnx_path), "checkpoint": str(ckpt_path),
    }
    csv_path = Path(csv_path or PARITY_CSV)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(result)
    return result
