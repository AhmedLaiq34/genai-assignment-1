"""Tests for POST /api/sketch and the sketch sample endpoints (Task 4).

No real model is needed: a tiny random-weight generator with the CONTRACTS 3.9 names is
exported to ONNX inside a tmp model folder. It embeds the style id and adds it to a conv of
the photo, so the style really changes the output.
"""
import base64
import io
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))  # so `import app.main` finds the backend

from fastapi.testclient import TestClient  # noqa: E402

import app.main as backend  # noqa: E402


class TinyGenerator(nn.Module):
    """photo [N,3,128,128] + style [N] -> sketch [N,1,128,128] in [-1,1]."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(3, 4)
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.out = nn.Conv2d(4, 1, 1)
        with torch.no_grad():
            self.embed.weight.mul_(5.0)  # make the style difference clearly visible

    def forward(self, photo, style):
        h = self.conv(photo) + self.embed(style)[:, :, None, None]  # style is added to every pixel
        return torch.tanh(self.out(h))


def export_tiny(path: Path) -> None:
    torch.onnx.export(
        TinyGenerator().eval(), (torch.zeros(1, 3, 128, 128), torch.zeros(1, dtype=torch.long)), str(path),
        input_names=["photo", "style"], output_names=["sketch"],
        dynamic_axes={"photo": {0: "N"}, "style": {0: "N"}, "sketch": {0: "N"}},
        opset_version=17, dynamo=False,
    )


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("t4_models")
    export_tiny(d / "t4_generator.onnx")
    return d


@contextmanager
def backend_client(models_dir, monkeypatch, samples_dir=None):
    monkeypatch.setattr(backend, "MODELS_DIR", models_dir)
    if samples_dir is not None:
        monkeypatch.setattr(backend, "SAMPLES_SKETCH_DIR", samples_dir)
    with TestClient(backend.app) as c:  # `with` runs the startup (loads the models)
        yield c


def photo_bytes(fmt="PNG", size=(200, 150), seed=0) -> bytes:
    arr = np.random.default_rng(seed).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


def decode(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def post_sketch(client, data=None, **form):
    data = photo_bytes() if data is None else data
    return client.post("/api/sketch", files={"file": ("a.png", data, "image/png")}, data=form)


def test_sketch_ok(tiny_model_dir, monkeypatch):
    with backend_client(tiny_model_dir, monkeypatch) as client:
        r = post_sketch(client, style="2")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["style_id"] == 1 and j["style_label"] == "Style 2"
    photo, sketch = decode(j["photo_png_b64"]), decode(j["sketch_png_b64"])
    assert photo.size == (128, 128) and photo.mode == "RGB"
    assert sketch.size == (128, 128) and sketch.mode == "L"  # real grayscale
    assert set(j["timing_ms"]) == {"preprocess", "inference", "total"}


def test_style_id_is_style_minus_one(tiny_model_dir, monkeypatch):
    with backend_client(tiny_model_dir, monkeypatch) as client:
        ids = [post_sketch(client, style=str(s)).json()["style_id"] for s in (1, 2, 3)]
    assert ids == [0, 1, 2]


def test_styles_give_different_sketches(tiny_model_dir, monkeypatch):
    data = photo_bytes()
    with backend_client(tiny_model_dir, monkeypatch) as client:
        sk = [np.asarray(decode(post_sketch(client, data, style=str(s)).json()["sketch_png_b64"]), dtype=np.int16)
              for s in (1, 2, 3)]
    assert np.abs(sk[0] - sk[1]).max() > 0 and np.abs(sk[1] - sk[2]).max() > 0 and np.abs(sk[0] - sk[2]).max() > 0


def test_photo_is_the_preprocessed_upload(tiny_model_dir, monkeypatch):
    # The returned photo is the 128x128 image the model saw (here: resized, not corrupted).
    data = photo_bytes()
    with backend_client(tiny_model_dir, monkeypatch) as client:
        j = post_sketch(client, data, style="1").json()
    expected = np.asarray(Image.open(io.BytesIO(data)).convert("RGB").resize((128, 128), Image.BICUBIC))
    assert np.array_equal(np.asarray(decode(j["photo_png_b64"])), expected)


@pytest.mark.parametrize("style", ["0", "4", "-1", "abc", ""])
def test_bad_style_is_422(tiny_model_dir, monkeypatch, style):
    with backend_client(tiny_model_dir, monkeypatch) as client:
        assert post_sketch(client, style=style).status_code == 422


def test_missing_style_is_422(tiny_model_dir, monkeypatch):
    with backend_client(tiny_model_dir, monkeypatch) as client:
        assert post_sketch(client).status_code == 422


def test_missing_model_is_503_but_server_up(tmp_path, monkeypatch):
    with backend_client(tmp_path, monkeypatch) as client:  # empty model folder
        r = post_sketch(client, style="1")
        h = client.get("/api/health")
    assert r.status_code == 503
    assert h.status_code == 200
    assert h.json()["models"]["t4_generator"]["loaded"] is False


def test_health_lists_generator(tiny_model_dir, monkeypatch):
    with backend_client(tiny_model_dir, monkeypatch) as client:
        m = client.get("/api/health").json()["models"]["t4_generator"]
    assert m["loaded"] is True and m["file"] == "t4_generator.onnx"


def test_bad_uploads(tiny_model_dir, monkeypatch):
    with backend_client(tiny_model_dir, monkeypatch) as client:
        assert post_sketch(client, b"not an image", style="1").status_code == 415
        assert post_sketch(client, b"", style="1").status_code == 400
        assert client.post("/api/sketch", data={"style": "1"}).status_code == 400  # neither file nor sample
        assert post_sketch(client, photo_bytes(fmt="BMP"), style="1").status_code == 415  # not PNG/JPEG/WebP


def test_too_large_upload_is_413(tiny_model_dir, monkeypatch):
    monkeypatch.setattr(backend, "MAX_UPLOAD_BYTES", 100)
    with backend_client(tiny_model_dir, monkeypatch) as client:
        assert post_sketch(client, style="1").status_code == 413


# ---------------------------------------------------------------- sample photos
def make_samples(folder: Path, n=2) -> Path:
    folder.mkdir()
    for i in range(n):
        (folder / f"face{i}.png").write_bytes(photo_bytes(size=(64, 80), seed=i))
    return folder


def test_sample_listing_and_fetch(tiny_model_dir, tmp_path, monkeypatch):
    samples = make_samples(tmp_path / "samples_sketch")
    with backend_client(tiny_model_dir, monkeypatch, samples) as client:
        lst = client.get("/api/sketch/samples").json()
        assert lst == [{"id": "face0", "url": "/api/sketch/samples/face0"},
                       {"id": "face1", "url": "/api/sketch/samples/face1"}]
        r = client.get(lst[0]["url"])
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        assert Image.open(io.BytesIO(r.content)).size == (64, 80)
        assert client.get("/api/sketch/samples/nope").status_code == 404


def test_sample_traversal_rejected(tiny_model_dir, tmp_path, monkeypatch):
    (tmp_path / "secret.png").write_bytes(photo_bytes())  # next to the samples folder
    samples = make_samples(tmp_path / "samples_sketch")
    with backend_client(tiny_model_dir, monkeypatch, samples) as client:
        for bad in ("..%2Fsecret", "../secret", "..%5Csecret", "%2e%2e%2fsecret"):
            assert client.get(f"/api/sketch/samples/{bad}").status_code in (404, 405)
        r = client.post("/api/sketch", data={"style": "1", "sample_id": "../secret"})
        assert r.status_code == 404


def test_sketch_from_sample_id(tiny_model_dir, tmp_path, monkeypatch):
    samples = make_samples(tmp_path / "samples_sketch")
    with backend_client(tiny_model_dir, monkeypatch, samples) as client:
        r = client.post("/api/sketch", data={"style": "3", "sample_id": "face1"})
        assert r.status_code == 200 and r.json()["style_id"] == 2
        assert client.post("/api/sketch", data={"style": "3", "sample_id": "zzz"}).status_code == 404
        both = client.post("/api/sketch", data={"style": "1", "sample_id": "face0"},
                           files={"file": ("a.png", photo_bytes(), "image/png")})
        assert both.status_code == 400


def test_missing_or_empty_samples_folder(tiny_model_dir, tmp_path, monkeypatch):
    with backend_client(tiny_model_dir, monkeypatch, tmp_path / "does_not_exist") as client:
        assert client.get("/api/sketch/samples").json() == []
    empty = tmp_path / "empty"
    empty.mkdir()
    with backend_client(tiny_model_dir, monkeypatch, empty) as client:
        assert client.get("/api/sketch/samples").json() == []
