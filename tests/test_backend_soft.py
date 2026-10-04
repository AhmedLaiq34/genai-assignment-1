"""Tests for POST /api/soft (soft mixture of experts, Task 3).

The real model is not needed. A tiny random model with the same ONNX interface is built once:
  input "input" [N,3,128,128]  ->  outputs "output" [N,3,128,128] and "weights" [N,4]
  (weights = softmax of a small gate over the four branches identity, salt, blur, occlusion;
   output = the weighted sum of the four branch images).

Until main.py contains the /api/soft hookup, the module fixture `soft_endpoint` installs the exact
snippet from artifacts/handoff/soft_main_snippet.py.txt into the running app (and removes it again
after the tests). Once the lead has pasted the snippet into main.py the fixture does nothing.
"""
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
# T3_BACKEND_DIR lets a check run against a patched copy of the backend (default: the real one).
sys.path.insert(0, os.environ.get("T3_BACKEND_DIR", str(ROOT / "app" / "backend")))

from fastapi.testclient import TestClient  # noqa: E402

import app.main as backend  # noqa: E402
from genai.tasks.task3 import BRANCH_NAMES  # noqa: E402
# Helpers of the /api/hard tests: fake classifier / specialists / universal model, upload and PNG helpers.
from test_backend_hard import (SPECIALISTS, make_classifier, make_specialist,  # noqa: E402
                               make_universal, photo_bytes, png_of)

SNIPPET = ROOT / "artifacts" / "handoff" / "soft_main_snippet.py.txt"
SOFT_FILE = "t3_soft_moe.onnx"


# --------------------------------------------------------------------------------------
# The tiny stand-in model and its ONNX export
# --------------------------------------------------------------------------------------
class TinySoftMoE(nn.Module):
    """Same interface as the real exported SoftMoE: forward(x) -> (output, weights)."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.gate = nn.Linear(3, 4)  # looks at the mean colour of the image
        with torch.no_grad():
            self.gate.weight.mul_(20.0)  # so that different colours give clearly different weights
        self.experts = nn.ModuleList(nn.Conv2d(3, 3, 1) for _ in range(3))  # random 1x1 "experts"

    def forward(self, x):
        w = torch.softmax(self.gate(x.mean(dim=(2, 3))), dim=1)  # [N,4]
        branches = [x] + [torch.sigmoid(e(x)) for e in self.experts]  # branch 0 = the input itself
        out = sum(w[:, k, None, None, None] * b for k, b in enumerate(branches))
        return out, w


def export_tiny_soft(path: Path, model: nn.Module) -> None:
    torch.onnx.export(
        model.eval(), torch.zeros(2, 3, 128, 128), str(path), opset_version=17, dynamo=False,
        input_names=["input"], output_names=["output", "weights"],
        dynamic_axes={"input": {0: "N"}, "output": {0: "N"}, "weights": {0: "N"}},
    )


@pytest.fixture(scope="module")
def tiny_model():
    return TinySoftMoE().eval()


@pytest.fixture(scope="module")
def soft_dir(tmp_path_factory, tiny_model):
    """A model folder that holds only the tiny soft model."""
    d = tmp_path_factory.mktemp("soft_models")
    export_tiny_soft(d / SOFT_FILE, tiny_model)
    return d


@pytest.fixture(scope="module", autouse=True)
def soft_endpoint():
    """Make sure /api/soft is the real endpoint (see the module docstring); undo afterwards."""
    if hasattr(backend, "soft_routing"):  # main.py already has the hookup
        yield
        return
    routes = list(backend.app.router.routes)
    backend.app.router.routes[:] = [r for r in routes if getattr(r, "path", "") != "/api/soft"]  # drop the 501 stub
    exec(compile(SNIPPET.read_text(encoding="utf-8"), str(SNIPPET), "exec"), backend.__dict__)
    yield
    backend.app.router.routes[:] = routes
    for name in ("soft_routing", "soft"):
        backend.__dict__.pop(name, None)


@contextmanager
def soft_client(models_dir, monkeypatch):
    """A TestClient whose model folder is `models_dir` (the startup loads the models in it)."""
    monkeypatch.setattr(backend, "MODELS_DIR", models_dir)
    with TestClient(backend.app) as c:
        yield c


def post_soft(client, data=None, **form):
    """POST a random upload (clean unless `corruption` is given) to /api/soft."""
    data = photo_bytes() if data is None else data
    return client.post("/api/soft", files={"file": ("a.png", data, "image/png")}, data=form)


def colour_png(rgb) -> bytes:
    """A solid-colour 64x64 PNG (a different gate input than a random photo)."""
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), tuple(rgb)).save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------
def test_response_fields(soft_dir, monkeypatch):
    with soft_client(soft_dir, monkeypatch) as client:
        r = post_soft(client, corruption="salt_pepper", severity="high", seed="5")
    assert r.status_code == 200, r.text
    j = r.json()
    assert set(j) == {"input_png_b64", "output_png_b64", "corruption_applied", "params", "seed",
                      "weights", "dominant", "dominant_id", "ranking", "timing_ms"}
    assert j["corruption_applied"] == "salt_pepper" and j["params"]["type"] == "salt_pepper"
    assert j["seed"] == 5
    # Four weights in branch order (identity, salt, blur, occlusion) that sum to 1.
    w = j["weights"]
    assert len(w) == 4 and all(0.0 <= v <= 1.0 for v in w) and abs(sum(w) - 1.0) < 1e-4
    assert j["dominant"] in BRANCH_NAMES
    assert j["dominant_id"] == int(np.argmax(w)) and j["dominant"] == BRANCH_NAMES[j["dominant_id"]]
    # The ranking lists all four branches, strongest first.
    assert sorted(j["ranking"]) == sorted(BRANCH_NAMES) and j["ranking"][0] == j["dominant"]
    assert [w[BRANCH_NAMES.index(n)] for n in j["ranking"]] == sorted(w, reverse=True)
    # Timing: three numbers, the inference time is a real measurement.
    t = j["timing_ms"]
    assert set(t) == {"preprocess", "inference", "total"}
    assert t["inference"] > 0.0 and t["total"] >= t["inference"]
    # Both images are 128x128 RGB PNGs.
    assert png_of(j["input_png_b64"]).shape == (128, 128, 3)
    assert png_of(j["output_png_b64"]).shape == (128, 128, 3)


def test_output_and_weights_come_from_the_model(soft_dir, tiny_model, monkeypatch):
    """The endpoint returns what the model computes for the (corrupted) input image."""
    with soft_client(soft_dir, monkeypatch) as client:
        j = post_soft(client, corruption="gaussian_blur", severity="medium", seed="2").json()
    inp = png_of(j["input_png_b64"]).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    with torch.no_grad():
        out, w = tiny_model(torch.from_numpy(np.ascontiguousarray(inp)))
    # The PNG is 8-bit, so the input the model saw differs from the decoded PNG by at most 0.5/255.
    assert np.abs(np.array(j["weights"]) - w[0].numpy()).max() < 0.02
    expected = np.rint(out[0].numpy() * 255).transpose(1, 2, 0)
    assert np.abs(png_of(j["output_png_b64"]).astype(float) - expected).max() <= 3.0


def test_weights_depend_on_the_image(soft_dir, monkeypatch):
    with soft_client(soft_dir, monkeypatch) as client:
        red = post_soft(client, colour_png((250, 10, 10))).json()
        blue = post_soft(client, colour_png((10, 10, 250))).json()
    assert red["corruption_applied"] == "none" and red["params"] == {"type": "clean"}
    assert np.abs(np.array(red["weights"]) - np.array(blue["weights"])).max() > 0.05
    for j in (red, blue):
        assert abs(sum(j["weights"]) - 1.0) < 1e-4


def test_sample_and_custom_params_use_the_universal_path(soft_dir, monkeypatch):
    with soft_client(soft_dir, monkeypatch) as client:
        sid = client.get("/api/samples").json()[0]["id"]
        r = client.post("/api/soft", data={"sample_id": sid, "corruption": "occlusion",
                                           "params": '{"n_rects": 2, "coverage": 0.2}', "seed": "3"})
    assert r.status_code == 200, r.text
    p = r.json()["params"]
    assert p["n_rects"] == 2 and abs(p["achieved_coverage"] - 0.2) <= 0.02


def test_missing_model_gives_503(tmp_path, monkeypatch):
    with soft_client(tmp_path, monkeypatch) as client:  # empty model folder
        assert client.get("/api/health").status_code == 200  # the server still starts
        r = post_soft(client)
        assert r.status_code == 503 and "t3_soft_moe" in r.json()["detail"]


def test_bad_inputs(soft_dir, monkeypatch):
    with soft_client(soft_dir, monkeypatch) as client:
        r = client.post("/api/soft", files={"file": ("x.png", b"not an image", "image/png")})
        assert r.status_code == 415 and "detail" in r.json()
        r = client.post("/api/soft", files={"file": ("x.png", b"", "image/png")})
        assert r.status_code == 400  # empty upload
        assert client.post("/api/soft", data={}).status_code == 400  # no image at all
        assert client.post("/api/soft", data={"sample_id": "nope"}).status_code == 404
        assert post_soft(client, corruption="fog").status_code == 422
        assert post_soft(client, corruption="gaussian_blur", severity="extreme").status_code == 422
        assert post_soft(client, corruption="salt_pepper", params="[1, 2]").status_code == 422
        assert post_soft(client, corruption="salt_pepper", params='{"p": 5}').status_code == 422


def test_health_lists_soft_model(soft_dir, tmp_path, monkeypatch):
    with soft_client(soft_dir, monkeypatch) as client:
        assert client.get("/api/health").json()["models"]["t3_soft_moe"]["loaded"] is True
    with soft_client(tmp_path, monkeypatch) as client:
        assert client.get("/api/health").json()["models"]["t3_soft_moe"]["loaded"] is False


def test_universal_and_hard_are_unchanged(soft_dir, tmp_path, monkeypatch):
    """With every model present, /api/universal and /api/hard still answer as before and the
    three endpoints corrupt the same upload identically."""
    shutil.copy(soft_dir / SOFT_FILE, tmp_path / SOFT_FILE)
    make_classifier(tmp_path / "t2_classifier.onnx", winner=1)
    for file_name, k in SPECIALISTS.values():
        make_specialist(tmp_path / file_name, k)
    make_universal(tmp_path / "t1_universal_ae.onnx")
    data = photo_bytes(fmt="JPEG")
    args = dict(files={"file": ("a.jpg", data, "image/jpeg")},
                data={"corruption": "salt_pepper", "severity": "high", "seed": "7"})
    with soft_client(tmp_path, monkeypatch) as client:
        uni = client.post("/api/universal", **args)
        hard = client.post("/api/hard", **args)
        soft = client.post("/api/soft", **args)
    assert uni.status_code == 200 and hard.status_code == 200 and soft.status_code == 200
    assert set(uni.json()["timing_ms"]) == {"preprocess", "inference", "total"}
    assert set(hard.json()) >= {"probs", "predicted", "expert", "identity_bypass"}
    assert set(hard.json()["timing_ms"]) == {"preprocess", "classifier", "expert", "total"}
    assert hard.json()["predicted"] == "salt_pepper" and hard.json()["expert"] == "salt"
    assert uni.json()["input_png_b64"] == hard.json()["input_png_b64"] == soft.json()["input_png_b64"]
    assert uni.json()["params"] == hard.json()["params"] == soft.json()["params"]
