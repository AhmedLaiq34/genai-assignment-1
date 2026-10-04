"""ONNX export + ONNX Runtime parity of the complete Task 3 soft mixture of experts (fixture models, random weights)."""
import csv
import json
import shutil

import numpy as np
import onnxruntime as ort
import pytest
import torch

from genai.common.constants import ONNX_FILES
from genai.common.paths import MODELS_ONNX
from genai.export import task3_export
from genai.export.onnx_verify import FIELDS, val_inputs
from genai.export.task3_export import SoftMoEExport, export_t3, verify_t3_parity
from genai.models.moe import SoftMoE
from genai.tasks.task3 import ONNX_KEY
from t3_fixtures import make_t3_sources
from test_task3_eval import make_t3_checkpoint

TAU = 2.0


def session(path):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Fixture sources, a (smoke) Task 3 checkpoint with tau 2 and the ONNX file exported from it (in a tmp dir)."""
    d = tmp_path_factory.mktemp("t3_onnx")
    paths, sha = make_t3_sources(d / "sources")
    ckpt = make_t3_checkpoint(d / "t3_soft_moe.pt", paths, sha, tau=TAU)
    onnx = d / "onnx" / ONNX_FILES[ONNX_KEY]
    export_t3(ONNX_KEY, ckpt, onnx)
    return paths, sha, ckpt, onnx


# ----------------------------------------------------------------------------- the graph
def test_file_name_follows_the_contract(exported):
    _paths, _sha, _ckpt, onnx = exported
    assert onnx.name == ONNX_FILES["t3_soft_moe"] == "t3_soft_moe.onnx"


def test_io_names_shapes_and_dynamic_batch(exported):
    _paths, _sha, _ckpt, onnx = exported
    sess = session(onnx)
    assert [i.name for i in sess.get_inputs()] == ["input"]
    assert [o.name for o in sess.get_outputs()] == ["output", "weights"]
    assert sess.get_inputs()[0].shape[1:] == [3, 128, 128]
    assert sess.get_outputs()[0].shape[1:] == [3, 128, 128] and sess.get_outputs()[1].shape[1:] == [4]
    assert all(isinstance(o.shape[0], str) for o in sess.get_outputs())          # the batch axis is symbolic on both outputs
    for n in (1, 5):
        out, w = sess.run(["output", "weights"], {"input": np.random.rand(n, 3, 128, 128).astype(np.float32)})
        assert out.shape == (n, 3, 128, 128) and w.shape == (n, 4)
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-5)                       # softmax is inside the graph
        assert out.min() >= 0 and out.max() <= 1                               # a convex combination of images in [0,1]


def test_wrapper_returns_output_and_weights_only(exported):
    paths, sha, _ckpt, _onnx = exported
    model = SoftMoE.load_from_task2(paths, tau=TAU, expected_sha256=sha)
    wrapper = SoftMoEExport(model)
    assert not wrapper.training and not wrapper.model.gate.training
    out = wrapper(torch.rand(2, 3, 128, 128))
    assert isinstance(out, tuple) and len(out) == 2 and out[0].shape == (2, 3, 128, 128) and out[1].shape == (2, 4)


def test_sidecar_marks_smoke_and_has_hashes(exported):
    _paths, _sha, ckpt, onnx = exported
    meta = json.loads((onnx.parent / (onnx.name + ".meta.json")).read_text())
    assert meta["smoke"] is True and meta["opset"] == 17 and meta["name"] == ONNX_KEY
    assert len(meta["onnx_sha256"]) == 64 and len(meta["checkpoint_sha256"]) == 64
    assert meta["onnx_file"] == ONNX_FILES[ONNX_KEY] and meta["parameters"] > 0
    assert meta["tau"] == TAU and meta["outputs"] == ["output", "weights"] and meta["inputs"] == ["input"]
    assert meta["run_id"].endswith("_smoke") and meta["model_config"]["tau"] == TAU
    assert set(meta["source_checkpoints"]) == {"classifier", "salt", "blur", "occlusion", "t1"}


def test_tau_is_baked_into_the_graph(exported, tmp_path):
    """Two checkpoints that differ only in tau give different weights, and the larger tau is the flatter."""
    paths, sha, _ckpt, _onnx = exported
    x = np.random.RandomState(0).rand(6, 3, 128, 128).astype(np.float32)
    top = {}
    for tau in (0.5, 5.0):
        ckpt = make_t3_checkpoint(tmp_path / f"tau{tau}.pt", paths, sha, tau=tau)
        export_t3(ONNX_KEY, ckpt, tmp_path / f"tau{tau}.onnx")
        _out, w = session(tmp_path / f"tau{tau}.onnx").run(["output", "weights"], {"input": x})
        top[tau] = w.max(axis=1)
    assert (top[0.5] > top[5.0]).all()


# ----------------------------------------------------------------------------- parity
def test_parity_on_val_inputs_with_all_four_conditions(exported, tiny_pets_root, tmp_path):
    _paths, _sha, ckpt, onnx = exported
    x = val_inputs(16, data_root=tiny_pets_root)
    assert x.shape == (16, 3, 128, 128)                                    # 4 inputs of each of the 4 conditions
    csv_path = tmp_path / "parity.csv"
    res = verify_t3_parity(ONNX_KEY, ckpt, onnx, inputs=x, tag="test", csv_path=csv_path)
    assert res["passed"] and res["n_inputs"] == 16 and res["model"] == ONNX_KEY
    assert res["output"]["max_abs_diff"] <= 1e-4 and res["weights"]["max_abs_diff"] <= 1e-4
    assert res["max_weight_sum_error"] <= 1e-5 and res["n_dominant_agree"] == 16
    rows = list(csv.DictReader(open(csv_path)))
    assert list(rows[0]) == FIELDS
    assert [r["model"] for r in rows] == ["t3_soft_moe", "t3_soft_moe:weights"] and all(r["tag"] == "test" for r in rows)
    assert all(r["passed"] == "True" and r["n_inputs"] == "16" and len(r["onnx_sha256"]) == 64 for r in rows)


def test_parity_default_inputs_come_from_data_root(exported, tiny_pets_root, tmp_path):
    _paths, _sha, ckpt, onnx = exported
    res = verify_t3_parity(ONNX_KEY, ckpt, onnx, data_root=tiny_pets_root, csv_path=tmp_path / "p.csv")
    assert res["passed"] and res["n_inputs"] == 16


def test_no_hidden_local_default_for_the_data_root(exported, tmp_path):
    """The Task 1 export stage failed on Kaggle because data_root silently defaulted to the repository's data/ folder."""
    _paths, _sha, ckpt, onnx = exported
    with pytest.raises(ValueError, match="data_root"):
        verify_t3_parity(ONNX_KEY, ckpt, onnx, csv_path=tmp_path / "p.csv")
    assert not (tmp_path / "p.csv").exists()


def test_parity_finds_data_in_a_nested_kaggle_style_folder(exported, tiny_pets_root, tmp_path):
    """/kaggle/input/datasets/<user>/<name>/... : the data sits in a nested folder of data_root."""
    _paths, _sha, ckpt, onnx = exported
    nested = tmp_path / "input" / "datasets" / "someone" / "pets-data"
    shutil.copytree(tiny_pets_root, nested)
    cfg = {"data_root": str(tmp_path / "input")}                           # what configs/devices/kaggle.yaml gives
    res = verify_t3_parity(ONNX_KEY, ckpt, onnx, data_root=cfg, csv_path=tmp_path / "p.csv")
    assert res["passed"] and res["n_inputs"] == 16


def test_parity_fails_for_an_onnx_file_of_another_model(exported, tmp_path):
    """A different model's graph must not pass (and the failing rows are still recorded)."""
    paths, sha, ckpt, _onnx = exported
    other_paths, other_sha = make_t3_sources(tmp_path / "other", seed=7)
    other_ckpt = make_t3_checkpoint(tmp_path / "other.pt", other_paths, other_sha, tau=TAU)
    export_t3(ONNX_KEY, other_ckpt, tmp_path / "other.onnx")
    res = verify_t3_parity(ONNX_KEY, ckpt, tmp_path / "other.onnx", inputs=torch.rand(8, 3, 128, 128),
                           csv_path=tmp_path / "p.csv")
    assert not res["passed"] and res["output"]["passed"] is False
    assert [r["passed"] for r in csv.DictReader(open(tmp_path / "p.csv"))] == ["False", "False"]


def test_dominant_branch_disagreement_fails_the_weights_row(exported, tmp_path, monkeypatch):
    """The check compares the argmax of the weights: feed it ONNX weights whose argmax is wrong."""
    _paths, _sha, ckpt, onnx = exported
    real = task3_export._run_onnx

    def rolled(onnx_path, x):                                              # the same numbers in a different branch order
        out, w = real(onnx_path, x)
        return out, np.roll(w, 1, axis=1)
    monkeypatch.setattr(task3_export, "_run_onnx", rolled)
    res = verify_t3_parity(ONNX_KEY, ckpt, onnx, inputs=torch.rand(8, 3, 128, 128), csv_path=tmp_path / "p.csv")
    assert not res["passed"] and res["weights"]["passed"] is False and res["output"]["passed"] is True
    assert res["n_dominant_agree"] < 8


# ----------------------------------------------------------------------------- what may be exported and where
def test_smoke_model_is_never_written_under_models_onnx(exported):
    _paths, _sha, ckpt, onnx = exported
    before = sorted(p.name for p in MODELS_ONNX.glob("*"))
    with pytest.raises(ValueError, match="smoke"):
        export_t3(ONNX_KEY, ckpt, MODELS_ONNX / ONNX_FILES[ONNX_KEY])
    assert sorted(p.name for p in MODELS_ONNX.glob("*")) == before        # nothing was written there
    assert MODELS_ONNX.resolve() not in onnx.resolve().parents


def test_non_smoke_checkpoint_is_not_marked_smoke(exported, tmp_path):
    paths, sha, _ckpt, _onnx = exported
    ckpt = make_t3_checkpoint(tmp_path / "final.pt", paths, sha, smoke=False)
    export_t3(ONNX_KEY, ckpt, tmp_path / "t3_soft_moe.onnx")              # a tmp folder: allowed either way
    meta = json.loads((tmp_path / "t3_soft_moe.onnx.meta.json").read_text())
    assert meta["smoke"] is False


def test_checkpoint_and_name_must_match(exported, tmp_path):
    paths, _sha, _ckpt, _onnx = exported
    with pytest.raises(ValueError, match="soft_moe"):
        export_t3(ONNX_KEY, paths["classifier"], tmp_path / "x.onnx")        # a Task 2 checkpoint is not a Task 3 one
    with pytest.raises(ValueError, match="not the Task 3 model"):
        export_t3("t2_salt", paths["salt"], tmp_path / "x.onnx")
    with pytest.raises(ValueError, match="not the Task 3 model"):
        verify_t3_parity("t2_salt", paths["salt"], tmp_path / "x.onnx", inputs=torch.rand(1, 3, 128, 128))
    assert not (tmp_path / "x.onnx").exists()
