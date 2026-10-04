"""Tests for the FastAPI backend (app/backend). Uses a tiny ONNX model made inside the test
(output = input * 0.5), so the real model is not needed."""
import base64
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))  # so `import app.main` finds the backend

from fastapi.testclient import TestClient  # noqa: E402

import app.main as backend  # noqa: E402
from app.preprocess import preprocess_bytes  # noqa: E402


def make_tiny_onnx(path: Path) -> None:
    """Graph: output = input * 0.5 with a dynamic batch axis (same I/O names as the real model)."""
    half = helper.make_tensor("half", TensorProto.FLOAT, [], [0.5])
    node = helper.make_node("Mul", ["input", "half"], ["output"])
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, 128, 128])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, 128, 128])
    graph = helper.make_graph([node], "tiny", [inp], [out], initializer=[half])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))
    Path(str(path) + ".meta.json").write_text(json.dumps({"smoke": True, "run_id": "test"}))


def set_models_dir(monkeypatch, path: Path) -> None:
    monkeypatch.setattr(backend, "MODELS_DIR", path)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    make_tiny_onnx(tmp_path / "t1_universal_ae.onnx")
    set_models_dir(monkeypatch, tmp_path)
    with TestClient(backend.app) as c:  # `with` runs the startup event (loads the models)
        yield c


@pytest.fixture()
def client_no_model(tmp_path, monkeypatch):
    set_models_dir(monkeypatch, tmp_path)  # empty folder
    with TestClient(backend.app) as c:
        yield c


def photo_bytes(fmt="PNG", size=(200, 150), seed=0) -> bytes:
    arr = np.random.default_rng(seed).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


def png_of(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    m = r.json()["models"]
    assert m["t1_universal"]["loaded"] is True
    assert len(m["t1_universal"]["sha256"]) == 64
    assert m["t1_universal"]["smoke"] is True
    assert m["t2_classifier"]["loaded"] is False  # not in the folder


def test_samples(client):
    r = client.get("/api/samples")
    assert r.status_code == 200
    items = r.json()
    assert len(items) == 8
    img = client.get(items[0]["url"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(img.content)).size == (128, 128)
    assert client.get("/api/samples/nope").status_code == 404


def test_universal_no_corruption_is_not_corrupted(client):
    data = photo_bytes()
    r = client.post("/api/universal", files={"file": ("a.png", data, "image/png")},
                    data={"corruption": "none", "seed": "1"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["corruption_applied"] == "none" and j["params"] == {"type": "clean"}
    # The returned input is exactly the preprocessed upload (no second corruption).
    expected = np.rint(preprocess_bytes(data) * 255).astype(np.uint8).transpose(1, 2, 0)
    assert np.array_equal(png_of(j["input_png_b64"]), expected)
    # The tiny model halves the input.
    out = png_of(j["output_png_b64"]).astype(int)
    assert np.abs(out - expected.astype(int) // 2).max() <= 1
    assert set(j["timing_ms"]) == {"preprocess", "inference", "total"}


@pytest.mark.parametrize("corruption", ["salt_pepper", "gaussian_blur", "occlusion"])
def test_universal_severity_corrupts_deterministically(client, corruption):
    data = photo_bytes(fmt="JPEG")
    args = dict(files={"file": ("a.jpg", data, "image/jpeg")},
                data={"corruption": corruption, "severity": "high", "seed": "7"})
    a = client.post("/api/universal", **args).json()
    b = client.post("/api/universal", **args).json()
    clean = np.rint(preprocess_bytes(data) * 255).astype(np.uint8).transpose(1, 2, 0)
    assert a["params"]["type"] == corruption
    assert not np.array_equal(png_of(a["input_png_b64"]), clean)
    assert a["input_png_b64"] == b["input_png_b64"]  # same seed -> same image


def test_universal_custom_params_and_sample(client):
    sid = client.get("/api/samples").json()[0]["id"]
    custom = {"n_rects": 2, "coverage": 0.2}
    r = client.post("/api/universal", data={"sample_id": sid, "corruption": "occlusion",
                                            "params": json.dumps(custom), "seed": "3"})
    assert r.status_code == 200, r.text
    p = r.json()["params"]
    assert p["n_rects"] == 2 and abs(p["achieved_coverage"] - 0.2) <= 0.02
    r = client.post("/api/universal", data={"sample_id": sid, "corruption": "gaussian_blur",
                                            "params": json.dumps({"kernel": 5, "sigma": 1.2})})
    assert r.json()["params"]["kernel"] == 5
    # invalid custom params -> 422
    r = client.post("/api/universal", data={"sample_id": sid, "corruption": "gaussian_blur",
                                            "params": json.dumps({"kernel": 4, "sigma": 1})})
    assert r.status_code == 422


def test_bad_inputs(client):
    r = client.post("/api/universal", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 415 and "detail" in r.json()
    gif = io.BytesIO()
    Image.new("RGB", (8, 8)).save(gif, format="GIF")
    r = client.post("/api/universal", files={"file": ("x.gif", gif.getvalue(), "image/gif")})
    assert r.status_code == 415
    assert client.post("/api/universal", data={}).status_code == 400  # no image at all
    assert client.post("/api/universal", data={"sample_id": "nope"}).status_code == 404
    sid = client.get("/api/samples").json()[0]["id"]
    r = client.post("/api/universal", data={"sample_id": sid, "corruption": "fog"})
    assert r.status_code == 422


def test_upload_too_large(client, monkeypatch):
    monkeypatch.setattr(backend, "MAX_UPLOAD_BYTES", 1000)
    r = client.post("/api/universal", files={"file": ("a.png", photo_bytes(), "image/png")})
    assert r.status_code == 413


def test_missing_model_gives_503_but_health_ok(client_no_model):
    assert client_no_model.get("/api/health").status_code == 200
    assert client_no_model.get("/api/health").json()["models"]["t1_universal"]["loaded"] is False
    sid = client_no_model.get("/api/samples").json()[0]["id"]
    r = client_no_model.post("/api/universal", data={"sample_id": sid})
    assert r.status_code == 503 and "detail" in r.json()


def test_preprocessing_matches_training_cache(tmp_path):
    """Backend preprocessing == scripts/prepare_pets.load_128 (used to build the training cache)."""
    spec = importlib.util.spec_from_file_location("prepare_pets", ROOT / "scripts" / "prepare_pets.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    for fmt, ext in [("JPEG", "jpg"), ("PNG", "png")]:
        path = tmp_path / f"img.{ext}"
        path.write_bytes(photo_bytes(fmt=fmt, size=(300, 220), seed=5))
        cache_version = prep.load_128(path)  # uint8 [128,128,3]
        backend_version = preprocess_bytes(path.read_bytes())  # float32 [3,128,128]
        assert backend_version.dtype == np.float32 and backend_version.shape == (3, 128, 128)
        expected = cache_version.transpose(2, 0, 1).astype(np.float32) / 255.0
        assert np.array_equal(backend_version, expected)


def test_real_smoke_model_if_present():
    """If the exported smoke ONNX exists, it must load and produce a 128x128 output."""
    path = ROOT / "models" / "onnx" / "t1_universal_ae.onnx"
    if not path.exists():
        pytest.skip("no exported ONNX")
    with TestClient(backend.app) as c:  # default MODELS_DIR
        sid = c.get("/api/samples").json()[0]["id"]
        r = c.post("/api/universal", data={"sample_id": sid, "corruption": "salt_pepper"})
        assert r.status_code == 200, r.text
        assert png_of(r.json()["output_png_b64"]).shape == (128, 128, 3)
