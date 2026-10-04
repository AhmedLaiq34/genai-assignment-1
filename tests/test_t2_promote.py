"""Promoting the four Task 2 checkpoints into a COPY of the real models/MANIFEST.json layout.

Task 1's last overnight stage crashed because the real manifest is {"version": 1, "models": [...]} and a dry
run into a temp folder never read the real file. This test starts from a copy of the real file.
"""
import json
import shutil

import pytest

from genai.common.checkpoint import promote, sha256_file
from genai.common.paths import MODELS_CKPT
from genai.tasks.task2 import PROMOTED_NAMES

REAL_MANIFEST = MODELS_CKPT.parent / "MANIFEST.json"


@pytest.mark.skipif(not REAL_MANIFEST.exists(), reason="no models/MANIFEST.json in this checkout")
def test_promote_four_checkpoints_into_a_copy_of_the_real_manifest(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    shutil.copyfile(REAL_MANIFEST, models / "MANIFEST.json")
    before = json.loads((models / "MANIFEST.json").read_text(encoding="utf-8"))
    # the real file is a bare list today, an earlier version was {"version": 1, "models": [...]}: both must work
    before_entries = before["models"] if isinstance(before, dict) else before

    for component, name in PROMOTED_NAMES.items():
        ckpt = tmp_path / f"{component}.pt"
        ckpt.write_bytes(component.encode() * 100)                      # content does not matter here
        dest = promote(ckpt, name, root=models)
        assert dest == models / "checkpoints" / f"{name}.pt"

    after = json.loads((models / "MANIFEST.json").read_text(encoding="utf-8"))
    assert isinstance(after, dict) == isinstance(before, dict)         # the layout is kept
    if isinstance(before, dict):
        assert after["version"] == before["version"]                    # the wrapper is kept
    after_entries = after["models"] if isinstance(after, dict) else after
    names = [m["name"] for m in after_entries]
    assert all(n in names for n in PROMOTED_NAMES.values())            # the four Task 2 entries are there
    assert all(m["name"] in names for m in before_entries)             # entries of other tasks are kept
    assert len(names) == len(set(names))
    for entry in after_entries:
        if entry["name"] in PROMOTED_NAMES.values():
            assert entry["sha256"] == sha256_file(models / "checkpoints" / f"{entry['name']}.pt")
