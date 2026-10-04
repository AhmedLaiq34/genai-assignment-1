"""ONNX export and parity check of the complete Task 3 soft mixture of experts. Implements CONTRACTS 3.9.

    name           file (models/onnx/)   input    outputs
    t3_soft_moe    t3_soft_moe.onnx      input    output   (N x 3 x 128 x 128, [0,1]: the mixture x_hat)
                                                  weights  (N x 4, softmax(logits / tau), order = BRANCH_NAMES)

The whole pipeline is ONE graph: gate, softmax with the temperature tau, identity branch, the three
experts and the weighted sum. tau is a Python number inside the model, so it becomes a constant of the
graph. Same settings as the other exports: opset 17, dynamic batch axis (on the input and on BOTH
outputs), the classic exporter (dynamo=False), model in eval() mode.

    SoftMoEExport(model)                       the wrapper that is traced: forward(x) -> (output, weights)
    export_t3(name, ckpt_path, out_path)       graph + <out_path>.meta.json sidecar
    verify_t3_parity(name, ckpt_path, onnx_path)   PyTorch vs ONNX Runtime; two rows in onnx_parity.csv

Smoke / fixture models: a checkpoint from a smoke run (config run.smoke, or a run_id ending in _smoke;
a model built from smoke source checkpoints is marked smoke at training time) gets "smoke": true in its
sidecar and is REFUSED a place under models/onnx/ (write it to artifacts/fixtures/task3/onnx/ instead).

The helpers of task2_export (smoke test, "is this path inside models/onnx", the "data_root is required"
rule) are imported, not copied.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from genai.common.checkpoint import sha256_file
from genai.common.constants import IMG_SIZE, ONNX_OPSET, ONNX_PARITY_TOL_MAX_ABS
from genai.export.onnx_verify import FIELDS, PARITY_CSV
from genai.export.task2_export import _inputs_or_val, _inside_models_onnx, _is_smoke
from genai.models.autoencoder import count_parameters
from genai.models.moe import SoftMoE
from genai.tasks.task3 import ONNX_KEY
from genai.tasks.task3.evaluate import load_soft_moe

WEIGHT_SUM_TOL = 1e-5        # |sum of the four weights - 1| must stay below this (plan D7)


class SoftMoEExport(nn.Module):
    """Wraps a SoftMoE so that the graph has exactly the two outputs of CONTRACTS 3.9.

    SoftMoE.forward returns (x_hat, w, logits); the ONNX file should return only (output, weights):
    the logits are an intermediate value that only the training loss needs.
    """

    def __init__(self, model: SoftMoE):
        super().__init__()
        self.model = model
        self.eval()          # gate in eval mode (BatchNorm running statistics, dropout off); the experts are always eval

    def forward(self, x: torch.Tensor):
        x_hat, w, _logits = self.model(x)
        return x_hat, w


def _check_name(name: str) -> None:
    if name != ONNX_KEY:
        raise ValueError(f"'{name}' is not the Task 3 model; expected '{ONNX_KEY}'")


def export_t3(name: str, ckpt_path, out_path) -> None:
    """Export the Task 3 checkpoint `ckpt_path` to the ONNX file `out_path` (see the module docstring)."""
    _check_name(name)
    model, ckpt = load_soft_moe(ckpt_path)           # also checks that this is a soft_moe checkpoint
    out_path = Path(out_path)
    smoke = _is_smoke(ckpt)
    if smoke and _inside_models_onnx(out_path):
        raise ValueError(f"{ckpt_path} comes from a smoke run: refusing to write it to models/onnx/. "
                         f"Use another folder, e.g. artifacts/fixtures/task3/onnx/ (--out-dir).")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        SoftMoEExport(model), dummy, str(out_path),
        input_names=["input"], output_names=["output", "weights"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}, "weights": {0: "batch"}},   # batch size is free
        opset_version=ONNX_OPSET,
        dynamo=False,        # the classic TorchScript exporter, exactly as in the Task 1 and Task 2 exports
    )

    cfg = ckpt["config"]
    meta = {
        "name": name,
        "onnx_file": out_path.name,
        "onnx_sha256": sha256_file(out_path),
        "source_checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "run_id": cfg.get("run_id", ""),
        "global_step": ckpt["global_step"],
        "smoke": smoke,
        "opset": ONNX_OPSET,
        "inputs": ["input"],
        "outputs": ["output", "weights"],
        "tau": model.tau,                                       # baked into the graph as a constant
        "model_config": cfg["model"],
        "source_checkpoints": cfg.get("source_checkpoints"),    # the Task 2 files this model started from (sha256)
        "parameters": count_parameters(model, trainable_only=False),
    }
    Path(str(out_path) + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"exported {out_path} (smoke={smoke}, tau={model.tau})")


def _run_onnx(onnx_path, x: torch.Tensor) -> tuple:
    """Feed x to the ONNX file with ONNX Runtime (CPU). Returns (output, weights) as numpy arrays."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    output, weights = session.run(["output", "weights"], {"input": x.numpy().astype(np.float32)})
    return output, weights


def _parity_row(model_name: str, tag: str, n: int, diff: np.ndarray, passed: bool, onnx_path, ckpt_path) -> dict:
    """One row of onnx_parity.csv (the columns of onnx_verify.FIELDS)."""
    return {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": model_name, "tag": tag,
        "n_inputs": n, "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
        "tolerance": ONNX_PARITY_TOL_MAX_ABS, "passed": bool(passed),
        "onnx_sha256": sha256_file(onnx_path), "checkpoint": str(ckpt_path),
    }


def verify_t3_parity(name: str, ckpt_path, onnx_path, n: int = 16, inputs=None, tag: str = "",
                     csv_path=None, data_root=None) -> dict:
    """Compare PyTorch and ONNX Runtime on the same inputs, for BOTH outputs.

    The inputs are n val-manifest images (n/4 of each of the 4 conditions) unless `inputs` is given;
    then `data_root` is required (no hidden "local" default, see task2_export._inputs_or_val).

    Checks (all must hold for the weights row to pass):
      * max |output_onnx - output_torch|   <= 1e-4
      * max |weights_onnx - weights_torch| <= 1e-4
      * |sum of the four ONNX weights - 1| <= 1e-5 on every row
      * the dominant branch (argmax of the weights) is the same. A row whose two largest PyTorch weights
        differ by less than the tolerance is a tie, where the argmax may legitimately flip; it is not counted.
    Appends two rows to report/tables/onnx_parity.csv (or csv_path): `name` (the output) and `name:weights`.
    Returns {"passed", "output": <row>, "weights": <row>, "max_weight_sum_error", "n_dominant_agree", ...}.
    """
    _check_name(name)
    model, _ckpt = load_soft_moe(ckpt_path)
    x = _inputs_or_val(n, inputs, data_root)

    with torch.no_grad():
        out_t, w_t, _logits = model(x)
    out_o, w_o = _run_onnx(onnx_path, x)
    out_t, w_t = out_t.numpy(), w_t.numpy()

    out_diff, w_diff = np.abs(out_t - out_o), np.abs(w_t - w_o)
    sum_error = np.abs(w_o.sum(axis=1) - 1.0)                       # one number per input

    top_two = np.sort(w_t, axis=1)[:, -2:]                          # the two largest PyTorch weights per row
    tie = (top_two[:, 1] - top_two[:, 0]) < ONNX_PARITY_TOL_MAX_ABS
    agree = (w_t.argmax(axis=1) == w_o.argmax(axis=1)) | tie
    n_agree = int(agree.sum())

    output_ok = bool(out_diff.max() <= ONNX_PARITY_TOL_MAX_ABS)
    weights_ok = bool(w_diff.max() <= ONNX_PARITY_TOL_MAX_ABS and sum_error.max() <= WEIGHT_SUM_TOL
                      and n_agree == len(agree))
    n_inputs = int(x.shape[0])
    row_out = _parity_row(name, tag, n_inputs, out_diff, output_ok, onnx_path, ckpt_path)
    row_w = _parity_row(f"{name}:weights", tag, n_inputs, w_diff, weights_ok, onnx_path, ckpt_path)

    csv_path = Path(csv_path or PARITY_CSV)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows([row_out, row_w])

    return {"check": "t3_parity", "model": name, "passed": output_ok and weights_ok, "n_inputs": n_inputs,
            "output": row_out, "weights": row_w,
            "max_weight_sum_error": float(sum_error.max()),
            "n_dominant_agree": n_agree, "n_dominant_ties": int(tie.sum()),
            "onnx_sha256": row_out["onnx_sha256"], "checkpoint": str(ckpt_path)}
