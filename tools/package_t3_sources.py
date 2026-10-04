"""Build dist/cloud_models/t3_sources.zip: the five checkpoints Task 3 starts from, for a PRIVATE Kaggle dataset.

Task 3 does not train anything from scratch: the gate is the Task 2 classifier and the three experts are the Task 2
specialists (plus the Task 1 autoencoder as a comparison column). Kaggle has no copy of models/checkpoints, so this
script puts the files into one zip. The zip holds (at its root, no sub folder):

    t2_classifier.pt  t2_ae_salt.pt  t2_ae_blur.pt  t2_ae_occlusion.pt  t1_universal_ae.pt
    SOURCES.json      {file name: sha256}   (the runner on Kaggle checks the files against it, and against TASK2_SHA256)

The script REFUSES (and writes nothing) if
  * a file is missing,
  * the sha256 of a Task 2 file differs from TASK2_SHA256 (genai.tasks.task3),
  * the sha256 of a file differs from its entry in models/MANIFEST.json (the T1 file is only checked there),
  * models/MANIFEST.json has no entry for a file.

    python tools/package_t3_sources.py
    python tools/package_t3_sources.py --manifest M.json --ckpt-dir DIR --out OUT.zip     (tests use fixture files)

models/MANIFEST.json is either a bare list of entries or {"version": 1, "models": [...]}: both are read.
The checkpoints are only READ here; nothing under models/ is changed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genai.common.checkpoint import sha256_file          # noqa: E402  (after the sys.path line on purpose)
from genai.tasks.task3 import SOURCE_FILES, TASK2_SHA256  # noqa: E402

DEFAULT_MANIFEST = ROOT / "models" / "MANIFEST.json"
DEFAULT_CKPT_DIR = ROOT / "models" / "checkpoints"
DEFAULT_OUT = ROOT / "dist" / "cloud_models" / "t3_sources.zip"


def read_manifest(manifest_path) -> dict:
    """{entry name: sha256} from models/MANIFEST.json (bare list or {"models": [...]})."""
    loaded = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    entries = loaded["models"] if isinstance(loaded, dict) else loaded
    return {e["name"]: e["sha256"] for e in entries}


def package(manifest_path=DEFAULT_MANIFEST, ckpt_dir=DEFAULT_CKPT_DIR, out_path=DEFAULT_OUT,
            expected=TASK2_SHA256, include_t1: bool = True) -> dict:
    """Check the files, then write the zip. Returns {file name: sha256}. Raises ValueError on any problem.

    expected: sha256 of the four Task 2 files, by source name (tests pass the hashes of their fixture files).
    """
    ckpt_dir, out_path = Path(ckpt_dir), Path(out_path)
    manifest = read_manifest(manifest_path)
    names = [n for n in SOURCE_FILES if n != "t1" or include_t1]       # classifier, salt, blur, occlusion, t1
    problems, hashes = [], {}
    for name in names:
        file_name = SOURCE_FILES[name]
        path = ckpt_dir / file_name
        if not path.is_file():
            problems.append(f"{file_name}: file not found in {ckpt_dir}")
            continue
        actual = sha256_file(path)
        hashes[file_name] = actual
        entry = path.stem                                              # the manifest name is the file name without .pt
        if name in expected and actual != expected[name]:
            problems.append(f"{file_name}: sha256 {actual} is not the Task 2 hash {expected[name]}")
        if entry not in manifest:
            problems.append(f"{file_name}: no entry '{entry}' in {manifest_path}")
        elif manifest[entry] != actual:
            problems.append(f"{file_name}: sha256 {actual} differs from the manifest entry {manifest[entry]}")
    if problems:
        raise ValueError("refusing to package, nothing written:\n  " + "\n  ".join(problems))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")                   # write then rename: never a half-written zip
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name in names:
            z.write(ckpt_dir / SOURCE_FILES[name], SOURCE_FILES[name])
        z.writestr("SOURCES.json", json.dumps(hashes, indent=2))
    os.replace(tmp, out_path)
    return hashes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="models/MANIFEST.json (bare list or {'models': [...]})")
    ap.add_argument("--ckpt-dir", default=str(DEFAULT_CKPT_DIR), help="folder with the five .pt files")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="the zip to write")
    ap.add_argument("--no-t1", action="store_true", help="leave the Task 1 checkpoint out (it is only a comparison column)")
    args = ap.parse_args()
    try:
        hashes = package(args.manifest, args.ckpt_dir, args.out, include_t1=not args.no_t1)
    except ValueError as err:
        sys.exit(str(err))
    out = Path(args.out)
    digest = sha256_file(out)
    (out.parent / (out.name + ".sha256")).write_text(f"{digest}  {out.name}\n", encoding="utf-8")
    print(f"{out}  {out.stat().st_size / 2**20:.1f} MB, {len(hashes)} checkpoints, zip sha256 {digest[:16]}...")
    for file_name, h in hashes.items():
        print(f"  {file_name:22s} {h}")


if __name__ == "__main__":
    main()
