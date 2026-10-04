"""ONNX export and parity check of the Task 4 generator (style-conditioned face-to-sketch). CONTRACTS 3.9.

    name            file (models/onnx/)      inputs                                  output
    t4_generator    t4_generator.onnx        photo  float32 [N,3,128,128] in [-1,1]  sketch float32 [N,1,128,128] in [-1,1]
                                             style  int64   [N]            (0, 1, 2)

Only the generator is exported (the discriminator is a training tool). Same settings as the Task 1 / Task 2
exports: opset 17, the classic exporter (dynamo=False), model in eval() mode, dynamic batch axis. The
batch axis is dynamic on ALL three tensors (both inputs and the output).

    load_t4_generator(ckpt_path)                      rebuild G from a run checkpoint or a promoted t4_generator.pt
    export_t4_generator(name, ckpt_path, out_path)    graph + <out_path>.meta.json sidecar
    verify_t4_parity(name, ckpt_path, onnx_path, ...) PyTorch vs ONNX Runtime numbers (row in onnx_parity.csv)

Smoke / fixture models: a checkpoint from a smoke run (config run.smoke, or a run_id ending in _smoke)
gets "smoke": true in its sidecar and is REFUSED a place under models/onnx/ (write it to
artifacts/fixtures/task4/onnx/ instead), so a random-weight model can never be mistaken for a final one.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from genai.common.checkpoint import load_checkpoint, sha256_file
from genai.common.constants import IMG_SIZE, ONNX_OPSET, ONNX_PARITY_TOL_MAX_ABS
from genai.export.onnx_verify import FIELDS, PARITY_CSV
from genai.export.task2_export import _inside_models_onnx, _is_smoke    # same smoke rules as Task 2
from genai.models.cgan import Generator, count_parameters

NUM_STYLES = 3
T4_NAME = "t4_generator"


def load_t4_generator(ckpt_path) -> tuple:
    """Rebuild the generator from a checkpoint. Returns (G in eval mode, checkpoint dict).

    Works for a run checkpoint (ckpt_best.pt / ckpt_last.pt: "model" holds G's weights) and for a promoted
    t4_generator.pt (split_checkpoint writes the same layout, with the discriminator removed).
    """
    ckpt = load_checkpoint(ckpt_path)
    G = Generator.from_config(ckpt["config"]["model"])
    G.load_state_dict(ckpt["model"])
    return G.eval(), ckpt          # eval(): dropout off (InstanceNorm has no running statistics to switch)


def export_t4_generator(name: str, ckpt_path, out_path) -> None:
    """Export checkpoint `ckpt_path` to the ONNX file `out_path` (see module docstring)."""
    if name != T4_NAME:
        raise ValueError(f"'{name}' is not the Task 4 generator; expected '{T4_NAME}'")
    G, ckpt = load_t4_generator(ckpt_path)
    out_path = Path(out_path)
    smoke = _is_smoke(ckpt)
    if smoke and _inside_models_onnx(out_path):
        raise ValueError(f"{ckpt_path} comes from a smoke run: refusing to write it to models/onnx/. "
                         f"Use another folder, e.g. artifacts/fixtures/task4/onnx/.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # dummy inputs only fix the shapes / dtypes the exporter traces with (values do not matter)
    dummy_photo = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    dummy_style = torch.zeros(1, dtype=torch.int64)
    torch.onnx.export(
        G, (dummy_photo, dummy_style), str(out_path),
        input_names=["photo", "style"], output_names=["sketch"],
        dynamic_axes={"photo": {0: "batch"}, "style": {0: "batch"}, "sketch": {0: "batch"}},   # batch is free
        opset_version=ONNX_OPSET,
        dynamo=False,        # the classic TorchScript exporter, exactly as in Task 1 / Task 2
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
        "parameters": count_parameters(G),
    }
    Path(str(out_path) + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"exported {out_path} (smoke={smoke})")


def val_inputs_t4(n: int, data_root) -> tuple:
    """The first n validation photos (sorted by pair id), each paired with every style id 0, 1, 2.

    Returns (photos [3n,3,128,128], styles int64 [3n]): first the n photos with style 0, then with style 1,
    then with style 2. The photo's own style is NOT used: the parity check covers all three styles for every photo.
    """
    from genai.fs2k.dataset import FS2KDataset          # imported here: not needed when `inputs` is given
    ds = FS2KDataset("val", data_root, augment=False)
    order = sorted(range(len(ds)), key=lambda i: ds.ids[i])[:n]
    photos = torch.stack([ds[i][0] for i in order])
    styles = torch.cat([torch.full((len(order),), k, dtype=torch.int64) for k in range(NUM_STYLES)])
    return photos.repeat(NUM_STYLES, 1, 1, 1), styles


def verify_t4_parity(name: str, ckpt_path, onnx_path, n: int = 16, inputs=None, tag: str = "",
                     csv_path=None, data_root=None) -> dict:
    """Compare PyTorch and ONNX Runtime outputs of the generator, like onnx_verify.verify_parity.

    The cases are n val photos x style ids 0, 1, 2 (16 x 3 = 48) unless `inputs=(photos, styles)` is given.
    data_root has NO default on purpose (the Task 1 export stage once failed on Kaggle because of a hidden
    "local" default): without `inputs` it is required. Pass the run's config dict, a device profile name
    ("local" / "kaggle" / "colab") or a data folder.
    Returns the result dict and appends it as one row to report/tables/onnx_parity.csv (or csv_path).
    """
    if name != T4_NAME:
        raise ValueError(f"'{name}' is not the Task 4 generator; expected '{T4_NAME}'")
    if inputs is not None:
        photos, styles = inputs
    elif data_root is None:
        raise ValueError("data_root is required to load the val photos: pass the run's config, a device "
                         "profile name ('kaggle') or a path (or pass `inputs=(photos, styles)` directly)")
    else:
        photos, styles = val_inputs_t4(n, data_root)
    styles = styles.to(torch.int64)

    G, _ckpt = load_t4_generator(ckpt_path)
    with torch.no_grad():
        expected = G(photos, styles).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = session.run(["sketch"], {"photo": photos.numpy().astype(np.float32),
                                   "style": styles.numpy().astype(np.int64)})[0]

    diff = np.abs(expected - got)
    result = {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": name, "tag": tag,
        "n_inputs": int(photos.shape[0]), "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
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
