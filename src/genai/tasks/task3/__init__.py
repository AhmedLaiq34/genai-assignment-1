"""Task 3: jointly trained soft mixture-of-experts restoration (gate + identity + three experts).

Names shared by every Task 3 module live here, so that the model, the training loop, the
evaluation, the ONNX export, the app and the Kaggle runner all use the same spelling.
The design is fixed in docs/TASK3_PLAN.md (section D, "Shared names").
"""
from __future__ import annotations

from genai.common import constants as C

# The four branches of the mixture, in the order of the gate's four logits (= corruption class ids
# 0 clean, 1 salt_pepper, 2 gaussian_blur, 3 occlusion). Branch 0 is the identity: it gets the input as is.
BRANCH_NAMES = ("identity", "salt", "blur", "occlusion")

# The three experts (the Task 2 specialists), same short names as in Task 2.
EXPERT_NAMES = ("salt", "blur", "occlusion")

# sha256 of the four Task 2 checkpoints Task 3 starts from (Task 2 report section 5, models/MANIFEST.json).
# They are read-only: every run checks them before loading and again at the end.
TASK2_SHA256 = {
    "classifier": "7d4a9a1f8a073e64e8cc21deec4db98d51ad48dc04eb9b5b4215f49102ece028",
    "salt": "4f8b2edb835c3b967fc8ef4ea1d43a4092841e3bfde4bd583dfb2c3a086eafee",
    "blur": "0d2aede1d0636d4f65115b2c0d064319cc7a0fd22b55a24fdf9d1af9a750d74c",
    "occlusion": "d8e82568179939796c1461265bed6af9412f5a727f6093bcc2b9370c2b64e1b4",
}

# Source name -> file name of the checkpoint. "t1" is optional (a comparison column only).
SOURCE_FILES = {
    "classifier": "t2_classifier.pt",
    "salt": "t2_ae_salt.pt",
    "blur": "t2_ae_blur.pt",
    "occlusion": "t2_ae_occlusion.pt",
    "t1": "t1_universal_ae.pt",
}

PROMOTED_NAME = "t3_soft_moe"   # models/checkpoints/<name>.pt (CONTRACTS 3.8)
ONNX_KEY = "t3_soft_moe"        # key in genai.common.constants.ONNX_FILES (CONTRACTS 3.9)
TRACKER_GROUP = "task3"         # W&B group
RUN_GROUP = "task3"             # runs/<group>/<run_id>/ folder

assert C.ONNX_FILES[ONNX_KEY]   # the key above really exists
assert set(EXPERT_NAMES) <= set(SOURCE_FILES) and set(TASK2_SHA256) == {"classifier", *EXPERT_NAMES}
