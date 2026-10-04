"""Hard-routed restoration (Task 2): classifier -> softmax -> argmax -> one specialist.

    clean            -> identity bypass (the input is returned unchanged, no expert runs)
    salt_pepper      -> specialist "salt"
    gaussian_blur    -> specialist "blur"
    occlusion        -> specialist "occlusion"

Class order is fixed everywhere (CONTRACTS 3.2): 0 clean, 1 salt_pepper, 2 gaussian_blur, 3 occlusion.
This file only does the model part. Upload handling, preprocessing and corruption are shared
with /api/universal and live in main.py.
"""
from __future__ import annotations

import time

import numpy as np

from genai.common import constants as C
from genai.tasks.task2 import EXPERT_FOR_CLASS, ONNX_KEYS  # class id -> expert name, expert -> ONNX key


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax: subtracting the maximum first avoids overflow in exp()."""
    z = logits.astype(np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def run_hard(img01: np.ndarray, get_session):
    """Route one image [3,128,128] in [0,1].

    get_session(key) returns the ONNX session of a model (and raises HTTP 503 if its file is
    missing). It is passed in so this file does not import main.py.

    Returns (output, routing, ms):
      output   the restored image [3,128,128] (equal to img01 when the prediction is clean)
      routing  dict with probs, predicted, predicted_id, expert, identity_bypass
      ms       dict with the time spent in the "classifier" and the "expert" (0.0 if bypassed)
    """
    x = img01[None]  # add the batch axis: [1,3,128,128]

    # 1. classifier: logits [1,4] -> probabilities -> most likely class
    classifier = get_session(ONNX_KEYS["classifier"])
    t0 = time.perf_counter()
    logits = classifier.run(["logits"], {"input": x})[0][0]  # [4]
    t1 = time.perf_counter()
    probs = softmax(logits)
    predicted_id = int(np.argmax(probs))
    expert = EXPERT_FOR_CLASS[predicted_id]
    bypass = expert == "identity"

    # 2. clean -> identity bypass; otherwise only the chosen specialist is loaded and run
    if bypass:
        output = img01
        expert_ms = 0.0
    else:
        specialist = get_session(ONNX_KEYS[expert])  # 503 here if this specialist's file is missing
        t2 = time.perf_counter()
        output = specialist.run(["output"], {"input": x})[0][0]  # [3,128,128]
        expert_ms = round((time.perf_counter() - t2) * 1000, 2)

    routing = {
        "probs": [round(float(p), 6) for p in probs],  # class order: clean, salt_pepper, gaussian_blur, occlusion
        "predicted": C.CLASS_NAMES[predicted_id],
        "predicted_id": predicted_id,
        "expert": expert,  # "salt" | "blur" | "occlusion" | "identity"
        "identity_bypass": bypass,
    }
    return output, routing, {"classifier": round((t1 - t0) * 1000, 2), "expert": expert_ms}
