"""ONNX export + ONNX Runtime parity for the four Task 2 models (fixture checkpoints, random weights)."""
import csv
import json

import numpy as np
import onnxruntime as ort
import pytest
import torch

from genai.common.checkpoint import build_checkpoint, save_checkpoint
from genai.common.constants import ONNX_FILES
from genai.common.paths import MODELS_ONNX
from genai.export.onnx_verify import val_inputs
from genai.export.task2_export import (export_all_task2, export_task2_model, verify_routing_parity,
                                       verify_task2_parity)
from genai.models.autoencoder import UniversalAE
from t2_fixtures import TINY_AE, make_fixture_checkpoints

NAMES = {"t2_classifier": "classifier", "t2_salt": "salt", "t2_blur": "blur", "t2_occlusion": "occlusion"}


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Fixture checkpoints and the four ONNX files exported from them (in a tmp dir)."""
    d = tmp_path_factory.mktemp("t2_onnx")
    ckpts = make_fixture_checkpoints(d / "ckpts")
    onnx = export_all_task2(ckpts, d / "onnx")
    return ckpts, onnx


def session(path):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def test_file_names_follow_the_contract(exported):
    _ckpts, onnx = exported
    assert {k: v.name for k, v in onnx.items()} == {k: ONNX_FILES[k] for k in NAMES}


@pytest.mark.parametrize("name", list(NAMES))
def test_io_names_and_dynamic_batch(exported, name):
    _ckpts, onnx = exported
    sess = session(onnx[name])
    out_name = "logits" if name == "t2_classifier" else "output"
    assert [i.name for i in sess.get_inputs()] == ["input"]
    assert [o.name for o in sess.get_outputs()] == [out_name]
    assert sess.get_inputs()[0].shape[1:] == [3, 128, 128]
    expected_tail = (4,) if name == "t2_classifier" else (3, 128, 128)
    for n in (1, 3, 7):
        out = sess.run([out_name], {"input": np.random.rand(n, 3, 128, 128).astype(np.float32)})[0]
        assert out.shape == (n,) + expected_tail


@pytest.mark.parametrize("name", list(NAMES))
def test_sidecar_marks_smoke_and_has_hashes(exported, name):
    ckpts, onnx = exported
    meta = json.loads((onnx[name].parent / (onnx[name].name + ".meta.json")).read_text())
    assert meta["smoke"] is True and meta["opset"] == 17 and meta["name"] == name
    assert len(meta["onnx_sha256"]) == 64 and len(meta["checkpoint_sha256"]) == 64
    assert meta["onnx_file"] == ONNX_FILES[name] and meta["parameters"] > 0
    assert meta["run_id"].endswith("_smoke") and "model_config" in meta


def test_parity_on_val_inputs_with_all_four_conditions(exported, tiny_pets_root, tmp_path):
    ckpts, onnx = exported
    x = val_inputs(16, data_root=tiny_pets_root)
    assert x.shape == (16, 3, 128, 128)                           # 4 inputs of each of the 4 conditions
    csv_path = tmp_path / "parity.csv"
    for name, component in NAMES.items():
        res = verify_task2_parity(name, ckpts[component], onnx[name], inputs=x, tag="test", csv_path=csv_path)
        assert res["passed"] and res["max_abs_diff"] <= 1e-4 and res["n_inputs"] == 16 and res["model"] == name
    rows = list(csv.DictReader(open(csv_path)))
    assert [r["model"] for r in rows] == list(NAMES) and all(r["tag"] == "test" for r in rows)


def test_parity_default_inputs_come_from_data_root(exported, tiny_pets_root, tmp_path):
    ckpts, onnx = exported
    res = verify_task2_parity("t2_salt", ckpts["salt"], onnx["t2_salt"], data_root=tiny_pets_root,
                              csv_path=tmp_path / "p.csv")
    assert res["passed"] and res["n_inputs"] == 16


def test_val_data_root_has_no_hidden_local_default(exported, tmp_path):
    """The Task 1 export stage failed on Kaggle because data_root silently defaulted to the repository's
    data/ folder. Without `inputs` the Task 2 verifiers now refuse to guess."""
    ckpts, onnx = exported
    with pytest.raises(ValueError, match="data_root"):
        verify_task2_parity("t2_salt", ckpts["salt"], onnx["t2_salt"], csv_path=tmp_path / "p.csv")
    with pytest.raises(ValueError, match="data_root"):
        verify_routing_parity(ckpts["classifier"], onnx["t2_classifier"])


def test_parity_finds_data_in_a_nested_kaggle_style_folder(exported, tiny_pets_root, tmp_path):
    """/kaggle/input/datasets/<user>/<name>/... : the data sits in a nested folder of data_root."""
    import shutil
    ckpts, onnx = exported
    nested = tmp_path / "input" / "datasets" / "someone" / "pets-data"
    shutil.copytree(tiny_pets_root, nested)
    cfg = {"data_root": str(tmp_path / "input")}                   # what configs/devices/kaggle.yaml gives
    res = verify_task2_parity("t2_blur", ckpts["blur"], onnx["t2_blur"], data_root=cfg, csv_path=tmp_path / "p.csv")
    assert res["passed"] and res["n_inputs"] == 16
    assert verify_routing_parity(ckpts["classifier"], onnx["t2_classifier"], data_root=cfg)["passed"]


def test_routing_parity_passes(exported, tiny_pets_root):
    ckpts, onnx = exported
    res = verify_routing_parity(ckpts["classifier"], onnx["t2_classifier"], data_root=tiny_pets_root)
    assert res["passed"] and res["n_agree"] == res["n_inputs"] == 16
    x = torch.rand(8, 3, 128, 128)
    assert verify_routing_parity(ckpts["classifier"], onnx["t2_classifier"], inputs=x)["passed"]


def test_checkpoint_must_match_the_model_name(exported, tmp_path):
    ckpts, _onnx = exported
    with pytest.raises(ValueError, match="cond_id"):
        export_task2_model("t2_blur", ckpts["salt"], tmp_path / "x.onnx")        # salt weights as blur
    with pytest.raises(ValueError, match="component"):
        export_task2_model("t2_classifier", ckpts["salt"], tmp_path / "x.onnx")
    with pytest.raises(ValueError, match="not a Task 2 model"):
        export_task2_model("t1_universal", ckpts["salt"], tmp_path / "x.onnx")
    assert not (tmp_path / "x.onnx").exists()


def test_smoke_model_is_never_written_under_models_onnx(exported, tmp_path):
    ckpts, onnx = exported
    before = sorted(p.name for p in MODELS_ONNX.glob("*"))
    with pytest.raises(ValueError, match="smoke"):
        export_task2_model("t2_salt", ckpts["salt"], MODELS_ONNX / ONNX_FILES["t2_salt"])
    with pytest.raises(ValueError, match="smoke"):
        export_all_task2(ckpts, MODELS_ONNX)
    assert sorted(p.name for p in MODELS_ONNX.glob("*")) == before            # nothing was written there
    assert all(MODELS_ONNX.resolve() not in p.resolve().parents for p in onnx.values())


def test_non_smoke_checkpoint_is_not_marked_smoke(tmp_path):
    torch.manual_seed(1)
    model = UniversalAE.from_config(TINY_AE).eval()
    cfg = {"component": "specialist", "model": TINY_AE, "specialist": {"corruption": "blur", "cond_id": 2},
           "run": {"smoke": False}, "run_id": "20260101-0000_local_t2spec_blur"}
    ckpt = tmp_path / "blur_final.pt"
    save_checkpoint(ckpt, build_checkpoint(model, config=cfg, global_step=10))
    export_task2_model("t2_blur", ckpt, tmp_path / "t2_ae_blur.onnx")
    meta = json.loads((tmp_path / "t2_ae_blur.onnx.meta.json").read_text())
    assert meta["smoke"] is False
