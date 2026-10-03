"""ONNX export + ONNX Runtime parity for the Task 1 autoencoder (random weights, tiny model)."""
import csv
import json

import numpy as np
import onnxruntime as ort
import pytest
import torch

from genai.common.checkpoint import build_checkpoint, save_checkpoint
from genai.export.onnx_export import export_model
from genai.export.onnx_verify import verify_parity
from genai.models.autoencoder import UniversalAE

MODEL_CFG = {"in_ch": 3, "base_channels": 8, "depth": 4, "bottleneck_dim": 32, "dropout": 0.1}


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    d = tmp_path_factory.mktemp("onnx")
    torch.manual_seed(0)
    model = UniversalAE.from_config(MODEL_CFG)
    # run a few training-mode passes so BatchNorm running stats are not the trivial defaults
    for _ in range(3):
        model(torch.rand(4, 3, 128, 128))
    cfg = {"model": MODEL_CFG, "run": {"smoke": True}, "run_id": "20260101-0000_test_x_smoke"}
    ckpt = d / "ckpt_last.pt"
    save_checkpoint(ckpt, build_checkpoint(model, config=cfg, global_step=3))
    onnx_path = d / "t1_universal_ae.onnx"
    export_model("t1_universal", ckpt, onnx_path)
    return d, ckpt, onnx_path


def test_export_names_and_sidecar(exported):
    d, ckpt, onnx_path = exported
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["input"]
    assert [o.name for o in sess.get_outputs()] == ["output"]
    assert sess.get_inputs()[0].shape[1:] == [3, 128, 128]
    meta = json.loads((d / "t1_universal_ae.onnx.meta.json").read_text())
    assert meta["smoke"] is True and meta["opset"] == 17 and len(meta["onnx_sha256"]) == 64


def test_dynamic_batch_axis(exported):
    _d, _ckpt, onnx_path = exported
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for n in (1, 3, 7):
        out = sess.run(["output"], {"input": np.random.rand(n, 3, 128, 128).astype(np.float32)})[0]
        assert out.shape == (n, 3, 128, 128)


def test_parity_within_tolerance_and_csv_row(exported, tmp_path):
    _d, ckpt, onnx_path = exported
    csv_path = tmp_path / "parity.csv"
    x = torch.rand(16, 3, 128, 128)
    res = verify_parity("t1_universal", ckpt, onnx_path, inputs=x, tag="test", csv_path=csv_path)
    assert res["passed"] and res["max_abs_diff"] <= 1e-4 and res["n_inputs"] == 16
    rows = list(csv.DictReader(open(csv_path)))
    assert len(rows) == 1 and rows[0]["tag"] == "test"


def test_unknown_model_name_not_implemented(exported):
    _d, ckpt, onnx_path = exported
    with pytest.raises(NotImplementedError):
        export_model("t4_generator", ckpt, onnx_path)
