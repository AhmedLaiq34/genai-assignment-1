import pytest

from genai.pets import split as sp


def _ids(n, prefix="tv"):
    return [f"{prefix}_{i}" for i in range(n)]


def test_counts_and_no_overlap():
    s = sp.make_split(_ids(3680), _ids(3669, "te"))
    assert len(s["train"]) == 2944 and len(s["val"]) == 736  # floor(0.8*3680)
    assert not set(s["train"]) & set(s["val"])
    assert not set(s["train"] + s["val"]) & set(s["test"]["ids"])
    assert sorted(s["train"] + s["val"]) == sorted(_ids(3680))
    assert s["test"]["locked"] is True


def test_floor_for_odd_n():
    s = sp.make_split(_ids(11), _ids(2, "te"))
    assert len(s["train"]) == 8 and len(s["val"]) == 3


def test_reproducible_and_input_order_independent():
    a = sp.make_split(_ids(500), _ids(50, "te"))
    b = sp.make_split(list(reversed(_ids(500))), _ids(50, "te"))
    assert a == b
    assert sp.split_sha256(a) == sp.split_sha256(b)


def test_roundtrip_and_tamper_detection(tmp_path):
    s = sp.make_split(_ids(100), _ids(10, "te"))
    path = tmp_path / "split.json"
    sp.write_split(s, path)
    assert sp.read_split(path) == s
    s["train"][0], s["train"][1] = s["train"][1], s["train"][0]  # edit the order
    sp.write_split(s, path)
    with pytest.raises(ValueError):
        sp.read_split(path)
