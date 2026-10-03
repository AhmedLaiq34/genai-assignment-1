"""FastAPI backend (CONTRACTS 3.10). Implemented now: health, samples, /api/universal.
/api/hard, /api/soft and /api/sketch answer 501 until their models exist.

Run locally from app/backend:  uvicorn app.main:app --port 8000
Env: MODELS_DIR (folder with the .onnx files), SAMPLES_DIR, MAX_UPLOAD_MB.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from genai.common import constants as C
from genai.pets import corruptions  # the SAME code used to train; never re-implemented here

from .preprocess import BadImage, preprocess_bytes, to_png_bytes

HERE = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("MODELS_DIR", HERE.parent.parent.parent / "models" / "onnx"))
SAMPLES_DIR = Path(os.environ.get("SAMPLES_DIR", HERE.parent / "samples"))
MAX_UPLOAD_BYTES = int(float(os.environ.get("MAX_UPLOAD_MB", "10")) * 1024 * 1024)

CORRUPTION_IDS = {"none": 0, "salt_pepper": 1, "gaussian_blur": 2, "occlusion": 3}



@asynccontextmanager
async def lifespan(_app):
    # Startup: load every ONNX model that exists (a missing one is only reported, not fatal).
    SESSIONS.clear()
    for key in C.ONNX_FILES:
        load_model(key)
    yield


app = FastAPI(title="GenAI A1 backend", lifespan=lifespan)

# --------------------------------------------------------------------------------------
# ONNX model registry: sessions are created at startup. A missing model never stops the
# server; it is reported by /api/health and the endpoint using it answers 503.
# --------------------------------------------------------------------------------------
SESSIONS: dict = {}  # model key -> ort.InferenceSession
INFO: dict = {}  # model key -> dict shown by /api/health


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(key: str) -> None:
    """Try to load one model; record the outcome in INFO[key]."""
    path = MODELS_DIR / C.ONNX_FILES[key]
    info = {"file": path.name, "loaded": False}
    INFO[key] = info
    if not path.exists():
        info["error"] = "file not found"
        return
    info["size_bytes"] = path.stat().st_size
    info["sha256"] = _sha256(path)
    meta_path = path.with_name(path.name + ".meta.json")  # sidecar written by the exporter
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        info["smoke"] = bool(meta.get("smoke", False))  # True = tiny smoke-test model
        info["run_id"] = meta.get("run_id")
    try:
        SESSIONS[key] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        info["loaded"] = True
        info.pop("error", None)
    except Exception as e:  # corrupt / incompatible file
        info["error"] = f"{e.__class__.__name__}: {e}"


def get_session(key: str) -> ort.InferenceSession:
    """Session for a model, retrying the load once (so a model mounted later works). 503 if absent."""
    if key not in SESSIONS:
        load_model(key)
    if key not in SESSIONS:
        raise HTTPException(503, f"Model '{key}' is not available: {INFO[key].get('error')}")
    return SESSIONS[key]


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "models_dir": str(MODELS_DIR),
        "models": INFO,
        "onnxruntime": ort.__version__,
        "torch": torch.__version__,
    }


# --------------------------------------------------------------------------------------
# Samples: a few clean images from the official TEST set, DISPLAY ONLY (never used for
# training or tuning). Generated once by make_samples.py and committed as small PNGs.
# --------------------------------------------------------------------------------------
def _sample_ids() -> list:
    return sorted(p.stem for p in SAMPLES_DIR.glob("*.png"))


def _sample_path(sample_id: str) -> Path:
    # Only accept ids that are in the listing: this also blocks path tricks like "../x".
    if sample_id not in _sample_ids():
        raise HTTPException(404, f"Unknown sample '{sample_id}'.")
    return SAMPLES_DIR / f"{sample_id}.png"


@app.get("/api/samples")
def list_samples():
    return [{"id": i, "url": f"/api/samples/{i}"} for i in _sample_ids()]


@app.get("/api/samples/{sample_id}")
def get_sample(sample_id: str):
    return FileResponse(_sample_path(sample_id), media_type="image/png")


# --------------------------------------------------------------------------------------
# Corruption parameters. THE ONE PLACE where UI/API params are mapped to the keys that
# genai.pets.corruptions expects.
#   API custom params                 ->  genai params
#   salt_pepper    {"p"}              ->  {"type","p"}
#   gaussian_blur  {"kernel","sigma"} ->  {"type","kernel","sigma"}
#   occlusion      {"n_rects","coverage"} -> {"type","n_rects","target_coverage","rects",...}
#                  (rects are drawn with the seeded rng; they may overlap; union coverage
#                  lands within +-2 pp of the request)
# Without custom params the fixed test severities low/medium/high are used.
# --------------------------------------------------------------------------------------
def build_params(corruption: str, severity: str, custom: Optional[dict],
                 rng: np.random.Generator) -> dict:
    cond_id = CORRUPTION_IDS[corruption]
    if custom is None:
        return corruptions.test_params(cond_id, severity, rng)
    try:
        if corruption == "salt_pepper":
            p = float(custom["p"])
            if not 0.0 <= p <= 1.0:
                raise ValueError("p must be in [0, 1]")
            return {"type": "salt_pepper", "p": p}
        if corruption == "gaussian_blur":
            kernel, sigma = int(custom["kernel"]), float(custom["sigma"])
            if kernel < 3 or kernel > 15 or kernel % 2 == 0:
                raise ValueError("kernel must be an odd integer between 3 and 15")
            if sigma <= 0:
                raise ValueError("sigma must be > 0")
            return {"type": "gaussian_blur", "kernel": kernel, "sigma": sigma}
        # occlusion
        n, target = int(custom["n_rects"]), float(custom["coverage"])
        if not 1 <= n <= 5 or not 0.02 <= target <= 0.6:
            raise ValueError("n_rects must be 1-5 and coverage 0.02-0.6")
        rects, achieved = corruptions._sample_rects(
            rng, n, target, target - 0.02, target + 0.02, non_overlapping=False, equal_shares=True)
        return {"type": "occlusion", "n_rects": n, "target_coverage": target,
                "rects": rects, "achieved_coverage": achieved}
    except (KeyError, TypeError, ValueError, RuntimeError) as e:
        raise HTTPException(422, f"Invalid custom params for {corruption}: {e}")


def _b64(png: bytes) -> str:
    return base64.b64encode(png).decode("ascii")  # raw base64, no data: prefix


async def read_input(file: Optional[UploadFile], sample_id: Optional[str]) -> bytes:
    """Bytes of the uploaded file or of the chosen sample; 4xx on any problem."""
    if file is not None and sample_id:
        raise HTTPException(400, "Send either 'file' or 'sample_id', not both.")
    if file is not None:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"File too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")
        if not data:
            raise HTTPException(400, "Uploaded file is empty.")
        return data
    if sample_id:
        return _sample_path(sample_id).read_bytes()
    raise HTTPException(400, "Send an image as 'file' or choose a 'sample_id'.")


@app.post("/api/universal")
async def universal(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    corruption: str = Form("none"),
    severity: str = Form("medium"),
    params: Optional[str] = Form(None),
    seed: int = Form(42),
):
    """Preprocess -> (optional corruption) -> universal autoencoder -> PNGs."""
    if corruption not in CORRUPTION_IDS:
        raise HTTPException(422, f"corruption must be one of {list(CORRUPTION_IDS)}.")
    if severity not in C.SEVERITY_NAMES:
        raise HTTPException(422, f"severity must be one of {list(C.SEVERITY_NAMES)}.")
    custom = None
    if params:
        try:
            custom = json.loads(params)
            if not isinstance(custom, dict):
                raise ValueError
        except ValueError:
            raise HTTPException(422, "'params' must be a JSON object.")
    session = get_session("t1_universal")  # 503 if the model file is missing

    t0 = time.perf_counter()
    data = await read_input(file, sample_id)
    try:
        img01 = preprocess_bytes(data)  # float32 [3,128,128] in [0,1]
    except BadImage as e:
        raise HTTPException(415, str(e))
    applied = {"type": "clean"}
    if corruption != "none":  # no corruption selected -> the upload is NOT corrupted again
        rng = np.random.default_rng(seed)
        applied = build_params(corruption, severity, custom, rng)
        img01 = corruptions.apply(torch.from_numpy(img01), applied, rng).numpy()
    t1 = time.perf_counter()

    out = session.run(None, {session.get_inputs()[0].name: img01[None]})[0][0]  # [3,128,128]
    t2 = time.perf_counter()

    return {
        "input_png_b64": _b64(to_png_bytes(img01)),  # after corruption
        "output_png_b64": _b64(to_png_bytes(out)),
        "corruption_applied": corruption,
        "params": applied,
        "seed": seed,
        "timing_ms": {
            "preprocess": round((t1 - t0) * 1000, 2),
            "inference": round((t2 - t1) * 1000, 2),
            "total": round((time.perf_counter() - t0) * 1000, 2),
        },
    }


# --------------------------------------------------------------------------------------
# Not implemented yet (Tasks 2-4). Clear 501 so the frontend can show a message.
# --------------------------------------------------------------------------------------
def _not_implemented(name: str) -> JSONResponse:
    return JSONResponse(status_code=501, content={"detail": f"/api/{name} is not implemented yet."})


@app.post("/api/hard")
def hard():
    return _not_implemented("hard")


@app.post("/api/soft")
def soft():
    return _not_implemented("soft")


@app.post("/api/sketch")
def sketch():
    return _not_implemented("sketch")
