"""Fixed project constants. Implements CONTRACTS §3.1, §3.2, §3.3, §3.4, §3.7, §3.9."""
from __future__ import annotations

# §3.2 class / branch IDs (fixed order everywhere)
CLASS_NAMES = ("clean", "salt_pepper", "gaussian_blur", "occlusion")
NUM_CLASSES = len(CLASS_NAMES)

# §3.1 / §3.4
IMG_SIZE = 128
SEED = 42
VAL_FRACTION_PETS = 0.2
VAL_FRACTION_FS2K = 0.15

# §3.3 training distributions
SALT_P_RANGE = (0.02, 0.15)
BLUR_KERNELS = (3, 5, 7)
BLUR_SIGMA_RANGE = (0.5, 2.5)
OCC_N_RECTS = (1, 2, 3)
OCC_COVERAGE_RANGE = (0.10, 0.35)

# §3.3 fixed test severities (low, medium, high)
SEVERITY_NAMES = ("low", "medium", "high")
TEST_SALT_P = (0.03, 0.08, 0.15)
TEST_BLUR = ((3, 0.7), (5, 1.5), (7, 2.5))  # (kernel, sigma)
TEST_OCC = ((1, 0.10), (2, 0.20), (3, 0.35))  # (n_rects, target union coverage)
TEST_OCC_TOLERANCE = 0.01  # +/- 1 percentage point

# §3.7 FS2K style IDs (UI "Style 1/2/3" -> 0/1/2)
STYLE_IDS = (0, 1, 2)

# §3.9 ONNX
ONNX_OPSET = 17
ONNX_FILES = {
    "t1_universal": "t1_universal_ae.onnx",
    "t2_classifier": "t2_classifier.onnx",
    "t2_salt": "t2_ae_salt.onnx",
    "t2_blur": "t2_ae_blur.onnx",
    "t2_occlusion": "t2_ae_occlusion.onnx",
    "t3_soft_moe": "t3_soft_moe.onnx",
    "t4_generator": "t4_generator.onnx",
}
ONNX_PARITY_TOL_MAX_ABS = 1e-4  # REC
