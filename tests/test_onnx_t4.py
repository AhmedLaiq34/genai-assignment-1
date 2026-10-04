"""ONNX export + ONNX Runtime parity for the Task 4 generator (random-weight fixture checkpoint, CPU).

Files are written under pytest's tmp_path, never under models/onnx/ and never into report/tables/onnx_parity.csv.
"""
import csv
import json

import numpy as np
import onnxruntime as ort
import pytest
import torch
from fs2k_fixture import build_mini_fs2k_data

from genai.common.checkpoint import build_checkpoint, save_checkpoint
from genai.common.paths import MODELS_ONNX
from genai.export.task4_export import (export_t4_generator, load_t4_generator, val_inputs_t4,
                                       verify_t4_parity)
from genai.models.cgan import Generator
from genai.tasks.task4.train import split_checkpoint

MODEL = {"base_channels": 8, "style_dim": 8, "dropout": 0.1, "num_styles": 3}


def make_fixture_checkpoint(path, smoke=True, model_cfg=None, seed=0):
    """A checkpoint with random weights in the layout of a run checkpoint (model = G state). Returns the path."""
    torch.manual_seed(seed)
    model_cfg = model_cfg or MODEL
    G = Generator(**{k: model_cfg[k] for k in ("base_channels", "style_dim", "dropout")})
    run_id = "20260101-0000_test_t4fixture" + ("_smoke" if smoke else "")
    cfg = {"model": model_cfg, "run": {"smoke": smoke}, "run_id": run_id}
    save_checkpoint(path, build_checkpoint(G, config=cfg, global_step=3))
    return path


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    d = tmp_path_factory.mktemp("t4_onnx")
    ckpt = make_fixture_checkpoint(d / "ckpt_best.pt")
    onnx = d / "onnx" / "t4_generator.onnx"
    export_t4_generator("t4_generator", ckpt, onnx)
    return ckpt, onnx


def session(path):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def test_io_names_types_and_dynamic_batch(exported):
    _ckpt, onnx = exported
    sess = session(onnx)
    assert [i.name for i in sess.get_inputs()] == ["photo", "style"]
    assert [o.name for o in sess.get_outputs()] == ["sketch"]
    assert sess.get_inputs()[0].type == "tensor(float)" and sess.get_inputs()[1].type == "tensor(int64)"
    assert sess.get_inputs()[0].shape[1:] == [3, 128, 128]
    for n in (1, 3, 7):
        out = sess.run(["sketch"], {"photo": np.random.rand(n, 3, 128, 128).astype(np.float32),
                                    "style": np.arange(n, dtype=np.int64) % 3})[0]
        assert out.shape == (n, 1, 128, 128) and out.min() >= -1 and out.max() <= 1     # tanh range


def test_sidecar_marks_smoke_and_has_hashes(exported):
    _ckpt, onnx = exported
    meta = json.loads((onnx.parent / (onnx.name + ".meta.json")).read_text())
    assert meta["smoke"] is True and meta["opset"] == 17 and meta["name"] == "t4_generator"
    assert len(meta["onnx_sha256"]) == 64 and len(meta["checkpoint_sha256"]) == 64
    assert meta["onnx_file"] == "t4_generator.onnx" and meta["parameters"] > 0
    assert meta["run_id"].endswith("_smoke") and meta["model_config"] == MODEL and meta["global_step"] == 3


@pytest.mark.parametrize("batch", [1, 5])
@pytest.mark.parametrize("style_id", [0, 1, 2])
def test_parity_all_styles_and_batch_sizes(exported, tmp_path, style_id, batch):
    ckpt, onnx = exported
    torch.manual_seed(style_id)
    x = torch.rand(batch, 3, 128, 128) * 2 - 1                    # photos in [-1,1]
    s = torch.full((batch,), style_id, dtype=torch.int64)
    res = verify_t4_parity("t4_generator", ckpt, onnx, inputs=(x, s), csv_path=tmp_path / "p.csv")
    assert res["passed"] and res["max_abs_diff"] <= 1e-4 and res["n_inputs"] == batch


def test_parity_appends_one_row_per_call_with_the_onnx_verify_fields(exported, tmp_path):
    from genai.export.onnx_verify import FIELDS
    ckpt, onnx = exported
    x = torch.rand(4, 3, 128, 128) * 2 - 1
    s = torch.tensor([0, 1, 2, 1])
    csv_path = tmp_path / "sub" / "parity.csv"
    r1 = verify_t4_parity("t4_generator", ckpt, onnx, inputs=(x, s), tag="a", csv_path=csv_path)
    verify_t4_parity("t4_generator", ckpt, onnx, inputs=(x, s), tag="b", csv_path=csv_path)
    assert list(r1) == FIELDS
    rows = list(csv.DictReader(open(csv_path)))
    assert [r["tag"] for r in rows] == ["a", "b"] and all(r["model"] == "t4_generator" for r in rows)
    assert rows[0]["n_inputs"] == "4" and rows[0]["passed"] == "True"


def test_style_input_matters_in_the_onnx_graph(exported):
    _ckpt, onnx = exported
    sess = session(onnx)
    photo = np.random.rand(1, 3, 128, 128).astype(np.float32) * 2 - 1
    outs = [sess.run(["sketch"], {"photo": photo, "style": np.array([k], dtype=np.int64)})[0] for k in range(3)]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert np.abs(outs[a] - outs[b]).max() > 1e-3             # different styles -> clearly different sketches


def test_without_inputs_data_root_is_required(exported, tmp_path):
    ckpt, onnx = exported
    with pytest.raises(ValueError, match="data_root"):
        verify_t4_parity("t4_generator", ckpt, onnx, csv_path=tmp_path / "p.csv")
    assert not (tmp_path / "p.csv").exists()


def test_parity_on_val_photos_with_all_three_styles(exported, tmp_path):
    """val photos x style 0, 1, 2 from the mini FS2K fixture (it has only 7 val photos: all are used)."""
    ckpt, onnx = exported
    mini = build_mini_fs2k_data(tmp_path / "fs2k")
    photos, styles = val_inputs_t4(16, mini["data_root"])
    assert photos.shape == (21, 3, 128, 128) and styles.tolist() == [0] * 7 + [1] * 7 + [2] * 7
    res = verify_t4_parity("t4_generator", ckpt, onnx, data_root=mini["data_root"], csv_path=tmp_path / "p.csv")
    assert res["passed"] and res["n_inputs"] == 21
    # data_root may also be a cfg-like dict
    res = verify_t4_parity("t4_generator", ckpt, onnx, n=4, data_root={"data_root": str(mini["data_root"])},
                           csv_path=tmp_path / "p.csv")
    assert res["passed"] and res["n_inputs"] == 12


def test_unknown_model_name_is_refused(exported, tmp_path):
    ckpt, _onnx = exported
    with pytest.raises(ValueError, match="t4_generator"):
        export_t4_generator("t1_universal", ckpt, tmp_path / "x.onnx")
    assert not (tmp_path / "x.onnx").exists()


def test_smoke_model_is_never_written_under_models_onnx(exported):
    ckpt, _onnx = exported
    before = sorted(p.name for p in MODELS_ONNX.glob("*"))
    with pytest.raises(ValueError, match="smoke"):
        export_t4_generator("t4_generator", ckpt, MODELS_ONNX / "t4_generator.onnx")
    assert sorted(p.name for p in MODELS_ONNX.glob("*")) == before


def test_non_smoke_checkpoint_is_not_marked_smoke(tmp_path):
    ckpt = make_fixture_checkpoint(tmp_path / "final.pt", smoke=False)
    export_t4_generator("t4_generator", ckpt, tmp_path / "t4_generator.onnx")
    meta = json.loads((tmp_path / "t4_generator.onnx.meta.json").read_text())
    assert meta["smoke"] is False


def test_load_accepts_a_promoted_generator_file(tmp_path):
    """split_checkpoint (training) writes t4_generator.pt: the loader and the export must accept it too."""
    G = Generator(**{k: MODEL[k] for k in ("base_channels", "style_dim", "dropout")})
    cfg = {"model": MODEL, "run": {"smoke": True}, "run_id": "x_smoke"}
    run_ckpt = build_checkpoint(G, config=cfg, global_step=5)
    run_ckpt.update(discriminator={"dummy": torch.zeros(1)})
    save_checkpoint(tmp_path / "ckpt_best.pt", run_ckpt)
    g_path, _d_path = split_checkpoint(tmp_path / "ckpt_best.pt", tmp_path / "promote")
    G2, ckpt = load_t4_generator(g_path)
    assert not G2.training and ckpt["role"] == "generator"
    photo, style = torch.rand(2, 3, 128, 128), torch.tensor([0, 2])
    with torch.no_grad():
        assert torch.equal(G.eval()(photo, style), G2(photo, style))
    export_t4_generator("t4_generator", g_path, tmp_path / "from_promoted.onnx")
    res = verify_t4_parity("t4_generator", g_path, tmp_path / "from_promoted.onnx",
                           inputs=(photo, style), csv_path=tmp_path / "p.csv")
    assert res["passed"]
