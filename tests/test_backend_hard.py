"""Tests for POST /api/hard (hard-routed restoration, Task 2).

Real models are not needed. Tiny ONNX files are built inside each test:
  - classifier: ignores the image and always outputs the same logits (a chosen class wins),
  - specialists: output = input * k, with a different k per specialist, so the output
    shows which specialist ran.
"""
import base64
import io
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))  # so `import app.main` finds the backend

from fastapi.testclient import TestClient  # noqa: E402

import app.main as backend  # noqa: E402
from app.preprocess import preprocess_bytes  # noqa: E402

CLASS_NAMES = ["clean", "salt_pepper", "gaussian_blur", "occlusion"]
# Specialist name -> (ONNX file name, multiplier k). The k values differ so the experts can be told apart.
SPECIALISTS = {
    "salt": ("t2_ae_salt.onnx", 0.5),
    "blur": ("t2_ae_blur.onnx", 0.25),
    "occlusion": ("t2_ae_occlusion.onnx", 0.75),
}
EXPERT_OF_CLASS = {1: "salt", 2: "blur", 3: "occlusion"}


def _save(graph, path: Path) -> None:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))


def make_classifier(path: Path, winner: int) -> None:
    """Graph: logits = [N,4] equal to 10 for class `winner` and 0 for the others (any input)."""
    logits = np.zeros((1, 4), dtype=np.float32)
    logits[0, winner] = 10.0
    nodes = [
        # mean over C,H,W -> [N]; times 0 -> still [N] but carries the batch size; then [N,1] + [1,4] -> [N,4]
        helper.make_node("ReduceMean", ["input"], ["m"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("Mul", ["m", "zero"], ["z"]),
        helper.make_node("Unsqueeze", ["z", "ax"], ["z2"]),
        helper.make_node("Add", ["z2", "const"], ["logits"]),
    ]
    inits = [
        numpy_helper.from_array(np.zeros((), dtype=np.float32), "zero"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "ax"),
        numpy_helper.from_array(logits, "const"),
    ]
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, 128, 128])
    out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", 4])
    _save(helper.make_graph(nodes, "fake_classifier", [inp], [out], initializer=inits), path)


def make_specialist(path: Path, k: float) -> None:
    """Graph: output = input * k (same I/O names as the real specialists)."""
    factor = helper.make_tensor("k", TensorProto.FLOAT, [], [k])
    node = helper.make_node("Mul", ["input", "k"], ["output"])
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, 128, 128])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, 128, 128])
    _save(helper.make_graph([node], "fake_specialist", [inp], [out], initializer=[factor]), path)


def make_universal(path: Path) -> None:
    make_specialist(path, 0.5)  # same graph; only used to check /api/universal still works


@contextmanager
def backend_client(tmp_path, monkeypatch, winner=None, experts=("salt", "blur", "occlusion"), universal=False):
    """A TestClient whose model folder holds a fake classifier (predicting class `winner`;
    None = no classifier file) and the listed fake specialists."""
    if winner is not None:
        make_classifier(tmp_path / "t2_classifier.onnx", winner)
    for name in experts:
        file_name, k = SPECIALISTS[name]
        make_specialist(tmp_path / file_name, k)
    if universal:
        make_universal(tmp_path / "t1_universal_ae.onnx")
    monkeypatch.setattr(backend, "MODELS_DIR", tmp_path)
    with TestClient(backend.app) as c:  # `with` runs the startup (loads the models)
        yield c


def photo_bytes(fmt="PNG", size=(200, 150), seed=0) -> bytes:
    arr = np.random.default_rng(seed).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


def png_of(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


def post_hard(client, data=None, **form):
    """POST a random upload (clean unless `corruption` is given) to /api/hard."""
    data = photo_bytes() if data is None else data
    return client.post("/api/hard", files={"file": ("a.png", data, "image/png")}, data=form)


def test_identity_bypass_when_predicted_clean(tmp_path, monkeypatch):
    data = photo_bytes()
    with backend_client(tmp_path, monkeypatch, winner=0) as client:
        r = post_hard(client, data, corruption="none")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["predicted"] == "clean" and j["predicted_id"] == 0
    assert j["expert"] == "identity" and j["identity_bypass"] is True
    assert j["timing_ms"]["expert"] == 0.0
    # Output is exactly the input (the clean upload, resized): no expert touched it.
    expected = np.rint(preprocess_bytes(data) * 255).astype(np.uint8).transpose(1, 2, 0)
    assert np.array_equal(png_of(j["input_png_b64"]), expected)
    assert np.array_equal(png_of(j["output_png_b64"]), expected)


@pytest.mark.parametrize("winner", [1, 2, 3])
def test_expert_choice(tmp_path, monkeypatch, winner):
    with backend_client(tmp_path, monkeypatch, winner=winner) as client:
        r = post_hard(client, corruption="salt_pepper", severity="high", seed="5")
    assert r.status_code == 200, r.text
    j = r.json()
    expert = EXPERT_OF_CLASS[winner]
    assert j["predicted"] == CLASS_NAMES[winner] and j["predicted_id"] == winner
    assert j["expert"] == expert and j["identity_bypass"] is False
    # The output carries the signature of the chosen specialist: output = input * k.
    k = SPECIALISTS[expert][1]
    inp = png_of(j["input_png_b64"]).astype(float)
    out = png_of(j["output_png_b64"]).astype(float)
    assert np.abs(out - inp * k).max() <= 1.0
    for other, (_, other_k) in SPECIALISTS.items():  # ... and not of the other two
        if other != expert:
            assert np.abs(out - inp * other_k).max() > 5.0
    assert j["timing_ms"]["expert"] > 0.0


def test_response_shape_and_probs(tmp_path, monkeypatch):
    with backend_client(tmp_path, monkeypatch, winner=2) as client:
        r = post_hard(client, corruption="gaussian_blur", severity="low", seed="9")
    j = r.json()
    assert set(j) == {"input_png_b64", "output_png_b64", "corruption_applied", "params", "seed",
                      "probs", "predicted", "predicted_id", "expert", "identity_bypass", "timing_ms"}
    assert j["corruption_applied"] == "gaussian_blur" and j["params"]["type"] == "gaussian_blur"
    assert j["seed"] == 9
    # 4 probabilities in class order (clean, salt_pepper, gaussian_blur, occlusion), summing to 1.
    probs = j["probs"]
    assert len(probs) == 4 and abs(sum(probs) - 1.0) < 1e-4
    assert int(np.argmax(probs)) == 2 and probs[2] > 0.99  # logit 10 vs 0 -> ~0.9999
    assert set(j["timing_ms"]) == {"preprocess", "classifier", "expert", "total"}


def test_missing_classifier_gives_503(tmp_path, monkeypatch):
    with backend_client(tmp_path, monkeypatch, winner=None) as client:  # specialists exist, classifier not
        assert client.get("/api/health").status_code == 200  # the server still starts
        r = post_hard(client)
        assert r.status_code == 503 and "t2_classifier" in r.json()["detail"]


def test_missing_specialist_gives_503_only_when_needed(tmp_path, monkeypatch):
    # Only the salt specialist exists. A blur prediction needs the blur one -> 503.
    with backend_client(tmp_path, monkeypatch, winner=2, experts=("salt",)) as client:
        r = post_hard(client)
        assert r.status_code == 503 and "t2_blur" in r.json()["detail"]
    # The classifier predicting salt only needs the salt specialist -> works.
    with backend_client(tmp_path, monkeypatch, winner=1, experts=("salt",)) as client:
        assert post_hard(client).status_code == 200
    # Predicting clean needs no specialist at all.
    with backend_client(tmp_path, monkeypatch, winner=0, experts=()) as client:
        assert post_hard(client).status_code == 200


def test_bad_inputs(tmp_path, monkeypatch):
    with backend_client(tmp_path, monkeypatch, winner=0) as client:
        r = client.post("/api/hard", files={"file": ("x.png", b"not an image", "image/png")})
        assert r.status_code == 415 and "detail" in r.json()
        assert client.post("/api/hard", data={}).status_code == 400  # no image at all
        assert client.post("/api/hard", data={"sample_id": "nope"}).status_code == 404
        r = post_hard(client, corruption="fog")  # invalid corruption
        assert r.status_code == 422
        r = post_hard(client, corruption="gaussian_blur", severity="extreme")
        assert r.status_code == 422


def test_sample_and_custom_params_use_the_universal_path(tmp_path, monkeypatch):
    with backend_client(tmp_path, monkeypatch, winner=3) as client:
        sid = client.get("/api/samples").json()[0]["id"]
        r = client.post("/api/hard", data={"sample_id": sid, "corruption": "occlusion",
                                           "params": '{"n_rects": 2, "coverage": 0.2}', "seed": "3"})
        assert r.status_code == 200, r.text
        p = r.json()["params"]
        assert p["n_rects"] == 2 and abs(p["achieved_coverage"] - 0.2) <= 0.02


def test_same_input_as_universal(tmp_path, monkeypatch):
    """Both endpoints corrupt identically (same seed -> same corrupted input image), and
    /api/universal still works in the same client."""
    data = photo_bytes(fmt="JPEG")
    args = dict(files={"file": ("a.jpg", data, "image/jpeg")},
                data={"corruption": "salt_pepper", "severity": "high", "seed": "7"})
    with backend_client(tmp_path, monkeypatch, winner=1, universal=True) as client:
        hard = client.post("/api/hard", **args)
        uni = client.post("/api/universal", **args)
    assert hard.status_code == 200 and uni.status_code == 200
    assert hard.json()["input_png_b64"] == uni.json()["input_png_b64"]
    assert hard.json()["params"] == uni.json()["params"]
    assert set(uni.json()["timing_ms"]) == {"preprocess", "inference", "total"}


def test_health_lists_task2_models(tmp_path, monkeypatch):
    with backend_client(tmp_path, monkeypatch, winner=0, experts=("salt",)) as client:
        models = client.get("/api/health").json()["models"]
    for key in ("t2_classifier", "t2_salt", "t2_blur", "t2_occlusion"):
        assert key in models
    assert models["t2_classifier"]["loaded"] is True and models["t2_salt"]["loaded"] is True
    assert models["t2_blur"]["loaded"] is False and models["t2_occlusion"]["loaded"] is False


def test_softmax_is_stable_and_sums_to_one():
    from app.hard import softmax
    p = softmax(np.array([1000.0, 0.0, -1000.0, 999.0], dtype=np.float32))  # exp(1000) would overflow
    assert np.isfinite(p).all() and abs(p.sum() - 1.0) < 1e-9 and p.argmax() == 0
