from genai.common import constants as c


def test_class_order():
    assert c.CLASS_NAMES == ("clean", "salt_pepper", "gaussian_blur", "occlusion")


def test_basics():
    assert c.IMG_SIZE == 128 and c.SEED == 42
    assert c.VAL_FRACTION_PETS == 0.2 and c.VAL_FRACTION_FS2K == 0.15
    assert c.STYLE_IDS == (0, 1, 2) and c.ONNX_OPSET == 17


def test_severity_tables():
    assert c.TEST_SALT_P == (0.03, 0.08, 0.15)
    assert c.TEST_BLUR == ((3, 0.7), (5, 1.5), (7, 2.5))
    assert c.TEST_OCC == ((1, 0.10), (2, 0.20), (3, 0.35))
    assert c.SEVERITY_NAMES == ("low", "medium", "high")
