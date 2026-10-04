"""Soft mixture-of-experts restoration (Task 3): one ONNX model does everything.

Inside the model (CONTRACTS 3.9, docs/TASK3_PLAN.md D1/D7):
    w     = softmax(gate(x) / tau)                     four weights, they sum to 1
    x_hat = w0 * x + w1 * salt(x) + w2 * blur(x) + w3 * occlusion(x)

Branch order is fixed everywhere (same as the corruption classes, CONTRACTS 3.2):
    0 identity (the input itself), 1 salt, 2 blur, 3 occlusion.
The ONNX file has the input "input" [N,3,128,128] and two outputs: "output" [N,3,128,128] and
"weights" [N,4]. The softmax and tau are already inside the graph, so nothing is computed here
except naming the strongest branch. Upload handling, preprocessing and corruption are shared
with the other endpoints and live in main.py.
"""
from __future__ import annotations

import time

import numpy as np

from genai.tasks.task3 import BRANCH_NAMES  # ("identity", "salt", "blur", "occlusion")


def run_soft(img01: np.ndarray, get_session):
    """Restore one image [3,128,128] in [0,1] with the soft mixture of experts.

    get_session(key) returns the ONNX session of a model (and raises HTTP 503 if its file is
    missing). It is passed in so this file does not import main.py.

    Returns (output, routing, ms):
      output   the restored image [3,128,128]
      routing  dict with weights, dominant, dominant_id, ranking
      ms       dict with the time spent in the model ("inference")
    """
    session = get_session("t3_soft_moe")  # 503 here if the model file is missing

    # One forward pass gives both the picture and the four weights.
    t0 = time.perf_counter()
    output, weights = session.run(["output", "weights"], {"input": img01[None]})
    inference_ms = round((time.perf_counter() - t0) * 1000, 2)

    w = weights[0]  # [4], order: identity, salt, blur, occlusion
    order = np.argsort(-w, kind="stable")  # branch ids from the largest weight to the smallest
    dominant_id = int(order[0])

    routing = {
        "weights": [round(float(v), 6) for v in w],  # same order as BRANCH_NAMES
        "dominant": BRANCH_NAMES[dominant_id],  # "identity" | "salt" | "blur" | "occlusion"
        "dominant_id": dominant_id,
        "ranking": [BRANCH_NAMES[int(i)] for i in order],  # strongest contributor first
    }
    return output[0], routing, {"inference": inference_ms}
