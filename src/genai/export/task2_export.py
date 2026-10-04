"""ONNX export and parity checks for the four Task 2 models. Implements CONTRACTS 3.9.

    name            file (models/onnx/)    input    output
    t2_classifier   t2_classifier.onnx     input    logits  (N x 4, the softmax is applied by the backend)
    t2_salt         t2_ae_salt.onnx        input    output  (N x 3 x 128 x 128, [0,1])
    t2_blur         t2_ae_blur.onnx        input    output
    t2_occlusion    t2_ae_occlusion.onnx   input    output

The names are the keys of genai.common.constants.ONNX_FILES. Same settings as the Task 1 export:
opset 17, dynamic batch axis, the classic exporter (dynamo=False), model in eval() mode.

    export_task2_model(name, ckpt_path, out_path)   graph + <out_path>.meta.json sidecar
    verify_task2_parity(name, ckpt_path, onnx_path) PyTorch vs ONNX Runtime numbers (row in onnx_parity.csv)
    verify_routing_parity(classifier_ckpt, classifier_onnx)  ONNX classifier picks the same class as PyTorch
    export_all_task2(ckpts, out_dir)                the four exports in one call

Smoke / fixture models: a checkpoint from a smoke run (config run.smoke, or a run_id ending in _smoke)
gets "smoke": true in its sidecar and is REFUSED a place under models/onnx/ (write it to
artifacts/fixtures/task2/onnx/ instead), so a random-weight model can never be mistaken for a final one.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from genai.common.checkpoint import sha256_file
from genai.common.constants import IMG_SIZE, ONNX_FILES, ONNX_OPSET, ONNX_PARITY_TOL_MAX_ABS
from genai.common.paths import MODELS_ONNX
from genai.export.onnx_verify import FIELDS, PARITY_CSV, val_inputs
from genai.models.autoencoder import count_parameters
from genai.tasks.task2 import ONNX_KEYS
from genai.tasks.task2.routing import load_component

# ONNX_FILES key -> component name ("t2_salt" -> "salt", "t2_classifier" -> "classifier")
COMPONENT_FOR_NAME = {key: component for component, key in ONNX_KEYS.items()}


def _component(name: str) -> str:
    if name not in COMPONENT_FOR_NAME:
        raise ValueError(f"'{name}' is not a Task 2 model; expected one of {sorted(COMPONENT_FOR_NAME)}")
    return COMPONENT_FOR_NAME[name]


def _output_name(name: str) -> str:
    """The classifier's ONNX output is called 'logits', every specialist's is called 'output'."""
    return "logits" if name == "t2_classifier" else "output"


def _is_smoke(ckpt: dict) -> bool:
    cfg = ckpt["config"]
    return bool((cfg.get("run") or {}).get("smoke")) or cfg.get("run_id", "").endswith("_smoke")


def _inside_models_onnx(path: Path) -> bool:
    """True if `path` lies under the real models/onnx/ folder."""
    try:
        Path(path).resolve().relative_to(MODELS_ONNX.resolve())
        return True
    except ValueError:
        return False


def export_task2_model(name: str, ckpt_path, out_path) -> None:
    """Export checkpoint `ckpt_path` (the model called `name`) to the ONNX file `out_path`."""
    model, ckpt = load_component(_component(name), ckpt_path)    # also checks the checkpoint matches `name`
    out_path = Path(out_path)
    smoke = _is_smoke(ckpt)
    if smoke and _inside_models_onnx(out_path):
        raise ValueError(f"{ckpt_path} comes from a smoke run: refusing to write it to models/onnx/. "
                         f"Use another folder, e.g. artifacts/fixtures/task2/onnx/ (--out-dir).")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["input"], output_names=[_output_name(name)],
        dynamic_axes={"input": {0: "batch"}, _output_name(name): {0: "batch"}},   # batch size is free
        opset_version=ONNX_OPSET,
        dynamo=False,        # the classic TorchScript exporter, exactly as in the Task 1 export
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
        "smoke": smoke,
        "opset": ONNX_OPSET,
        "model_config": ckpt["config"]["model"],
        "parameters": count_parameters(model),
    }
    Path(str(out_path) + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"exported {out_path} (smoke={smoke})")


def export_all_task2(ckpts: dict, out_dir) -> dict:
    """Export the four models. ckpts = {"classifier": path, "salt": path, "blur": path, "occlusion": path}.

    Returns {ONNX_FILES key: written path}. Files are named as in ONNX_FILES (t2_classifier.onnx, ...).
    """
    written = {}
    for component, key in ONNX_KEYS.items():
        written[key] = Path(out_dir) / ONNX_FILES[key]
        export_task2_model(key, ckpts[component], written[key])
    return written


def _run_onnx(onnx_path, name: str, x: torch.Tensor) -> np.ndarray:
    """Feed x to the ONNX file with ONNX Runtime (CPU) and return its single output."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return session.run([_output_name(name)], {"input": x.numpy().astype(np.float32)})[0]


def _inputs_or_val(n: int, inputs, data_root) -> torch.Tensor:
    """The given inputs, or n val-manifest images. data_root has NO default here on purpose: a hidden
    default of "local" (the repository's data/ folder) fails on Kaggle, where the data is under
    /kaggle/input (the Task 1 export stage failed that way). Pass the run's config dict, a device
    profile name ("local" / "kaggle" / "colab") or a data path."""
    if inputs is not None:
        return inputs
    if data_root is None:
        raise ValueError("data_root is required to load the val inputs: pass the run's config, a device "
                         "profile name ('kaggle') or a path (or pass `inputs=` directly)")
    return val_inputs(n, data_root)


def verify_task2_parity(name: str, ckpt_path, onnx_path, n: int = 16, inputs=None, tag: str = "",
                        csv_path=None, data_root=None) -> dict:
    """Compare PyTorch and ONNX Runtime outputs of one Task 2 model, like onnx_verify.verify_parity.

    The inputs are n val-manifest images (n/4 of each of the 4 conditions) unless `inputs` is given;
    then `data_root` is required (see _inputs_or_val).
    Returns the result dict and appends it as one row to report/tables/onnx_parity.csv (or csv_path).
    """
    model, _ckpt = load_component(_component(name), ckpt_path)
    x = _inputs_or_val(n, inputs, data_root)

    with torch.no_grad():
        expected = model(x).numpy()
    got = _run_onnx(onnx_path, name, x)

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


def verify_routing_parity(classifier_ckpt, classifier_onnx, n: int = 16, inputs=None, data_root=None) -> dict:
    """Does the ONNX classifier route like the PyTorch classifier? (same argmax class on every input)

    Returns {"n_inputs", "n_agree", "passed", "pytorch_classes", "onnx_classes", ...}. `passed` is True
    only if every input gets the same class. Nothing is written to onnx_parity.csv: the numeric logits
    check is verify_task2_parity; this one is only the discrete routing decision.
    """
    model, _ckpt = load_component("classifier", classifier_ckpt)
    x = _inputs_or_val(n, inputs, data_root)

    with torch.no_grad():
        torch_classes = model(x).argmax(dim=1).numpy()
    onnx_classes = _run_onnx(classifier_onnx, "t2_classifier", x).argmax(axis=1)

    n_agree = int((torch_classes == onnx_classes).sum())
    return {
        "check": "routing_parity",
        "model": "t2_classifier",
        "n_inputs": int(x.shape[0]),
        "n_agree": n_agree,
        "passed": n_agree == int(x.shape[0]),
        "pytorch_classes": torch_classes.tolist(),
        "onnx_classes": onnx_classes.tolist(),
        "onnx_sha256": sha256_file(classifier_onnx),
        "checkpoint": str(classifier_ckpt),
    }
