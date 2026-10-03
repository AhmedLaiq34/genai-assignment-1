"""Export trained models to ONNX opset 17, dynamic batch. Implements CONTRACTS 3.9.

Only the Task 1 universal autoencoder is implemented here so far. Other names raise
NotImplementedError until their model classes exist.

export_model("t1_universal", ckpt_path, out_path) writes
    <out_path>                  the ONNX graph: input "input" (N,3,128,128) -> output "output"
    <out_path>.meta.json        source checkpoint, sha256 of both files, smoke flag, opset
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from genai.common.checkpoint import load_checkpoint, sha256_file
from genai.common.constants import IMG_SIZE, ONNX_OPSET
from genai.models.autoencoder import UniversalAE, count_parameters


def load_t1_model(ckpt_path) -> tuple:
    """Rebuild the Task 1 autoencoder from a checkpoint. Returns (model in eval mode, checkpoint dict)."""
    ckpt = load_checkpoint(ckpt_path)
    model = UniversalAE.from_config(ckpt["config"]["model"])
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt          # eval(): dropout off, BatchNorm uses running statistics


def export_model(name: str, ckpt_path, out_path) -> None:
    """Export checkpoint `ckpt_path` to the ONNX file `out_path` (see module docstring)."""
    if name != "t1_universal":
        raise NotImplementedError(f"ONNX export for '{name}' is not implemented yet")
    model, ckpt = load_t1_model(ckpt_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},   # batch size is free
        opset_version=ONNX_OPSET,
        dynamo=False,        # the classic TorchScript exporter: stable, honours opset 17 exactly
    )

    run_id = ckpt["config"].get("run_id", "")
    meta = {
        "name": name,
        "onnx_file": out_path.name,
        "onnx_sha256": sha256_file(out_path),
        "source_checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "run_id": run_id,
        "global_step": ckpt["global_step"],
        "smoke": bool(ckpt["config"].get("run", {}).get("smoke")) or run_id.endswith("_smoke"),
        "opset": ONNX_OPSET,
        "model_config": ckpt["config"]["model"],
        "parameters": count_parameters(model),
    }
    Path(str(out_path) + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"exported {out_path} (smoke={meta['smoke']})")
