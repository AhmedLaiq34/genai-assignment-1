"""Task 2: corruption classifier, three specialist autoencoders, hard-routed restoration.

Names shared by every Task 2 module live here, so that the classifier, the specialists, the
routing/evaluation code, the ONNX export and the app all use the same spelling.
"""
from __future__ import annotations

from genai.common import constants as C

# Specialist short name -> corruption class id (0 clean, 1 salt_pepper, 2 gaussian_blur, 3 occlusion).
# The short names match the ONNX file names (t2_ae_salt.onnx, ...).
SPECIALIST_COND_ID = {"salt": 1, "blur": 2, "occlusion": 3}

# What each component is called when it is promoted to models/checkpoints/<name>.pt (CONTRACTS 3.8).
PROMOTED_NAMES = {
    "classifier": "t2_classifier",
    "salt": "t2_ae_salt",
    "blur": "t2_ae_blur",
    "occlusion": "t2_ae_occlusion",
}

# Component name -> key in genai.common.constants.ONNX_FILES (CONTRACTS 3.9).
ONNX_KEYS = {
    "classifier": "t2_classifier",
    "salt": "t2_salt",
    "blur": "t2_blur",
    "occlusion": "t2_occlusion",
}

# Expert that handles each predicted class; class 0 (clean) is the identity bypass, no expert.
EXPERT_FOR_CLASS = {0: "identity", 1: "salt", 2: "blur", 3: "occlusion"}

assert all(C.ONNX_FILES[key] for key in ONNX_KEYS.values())   # the keys above really exist
