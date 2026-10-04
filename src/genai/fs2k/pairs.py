"""FS2K photo-sketch pair resolution. Implements CONTRACTS 3.7.

Facts verified on the real FS2K download (data/raw/fs2k/FS2K/), 2026-10-04:
  * anno_train.json / anno_test.json are JSON lists of dicts (1058 train, 1046 test entries).
  * "image_name" looks like "photo1/image0110".
  * "style" is an int that is already 0, 1 or 2 (train counts 357/350/351, test 619/381/46).
  * photo  = photo/<image_name>.jpg
  * sketch = sketch/sketch<N>/sketch<NNNN>.<ext>  (N = digit of "photoN", NNNN = digits of "imageNNNN";
    sketch2 files are .png, the others .jpg)
These rules are also written in configs/data_fs2k.yaml (section "pairs") and used as the defaults here.

A resolved pair is a dict {"id", "photo", "sketch", "style"}; the two paths are relative to the
folder that holds the annotation files (called raw_root here).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

STYLE_IDS = (0, 1, 2)
STYLE_FIELD = "style"
PHOTO_TEMPLATE = "photo/{image_name}"
SKETCH_TEMPLATE = "sketch/sketch{folder}/sketch{number}"
EXTENSIONS = (".jpg", ".jpeg", ".png")

_NAME_RE = re.compile(r"^photo(\d+)/image(\d+)$")  # "photo1/image0110"


def find_raw_root(raw_root, anno_name: str = "anno_train.json", hint: str | None = None) -> Path:
    """Return the folder that really contains the annotation file.

    raw_root may be that folder, or its parent (the real download sits in data/raw/fs2k/FS2K/).
    `hint` is a sub-folder name tried first (config key raw_subdir)."""
    raw_root = Path(raw_root)
    candidates = [raw_root]
    if hint:
        candidates.append(raw_root / hint)
    if raw_root.is_dir():
        candidates += sorted(p for p in raw_root.iterdir() if p.is_dir())  # one level down
    for c in candidates:
        if (c / anno_name).is_file():
            return c
    raise FileNotFoundError(f"{anno_name} not found in {raw_root} (or in one sub-folder of it)")


def load_annotations(path) -> list:
    """Read an anno_*.json file and return the raw entries (a list of dicts)."""
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        raise ValueError(f"{path}: expected a JSON list of dicts")
    return entries


def style_of(entry: dict, field: str = STYLE_FIELD) -> int:
    """The style of one entry as an int in {0, 1, 2}. Anything else raises ValueError."""
    if field not in entry:
        raise ValueError(f"entry {entry.get('image_name')!r} has no field {field!r}")
    value = entry[field]
    if isinstance(value, bool) or not isinstance(value, int) or value not in STYLE_IDS:
        raise ValueError(f"entry {entry.get('image_name')!r}: style {value!r} is not one of {STYLE_IDS}")
    return value


def pair_id(entry: dict) -> str:
    """"photo1/image0110" -> "photo1_image0110" (safe as a file name)."""
    return str(entry["image_name"]).replace("/", "_")


def _with_extension(root: Path, stem: str, extensions) -> str | None:
    """First existing file root/<stem><ext> for the allowed extensions (as a posix path), else None."""
    for ext in extensions:
        if (root / (stem + ext)).is_file():
            return stem + ext
    return None


def resolve_pairs(raw_root, anno_json, style_field: str = STYLE_FIELD, extensions=EXTENSIONS,
                  photo_template: str = PHOTO_TEMPLATE, sketch_template: str = SKETCH_TEMPLATE) -> list:
    """Resolve every entry of one annotation file to an existing photo and sketch.

    anno_json is the annotation file name (e.g. "anno_train.json"); only its name is used and the
    file is looked up in raw_root (or one folder below it, see find_raw_root).
    Raises FileNotFoundError listing EVERY entry whose photo or sketch is missing."""
    root = find_raw_root(raw_root, Path(anno_json).name)  # the folder that holds the annotation file
    anno_path = root / Path(anno_json).name
    pairs, problems = [], []
    for entry in load_annotations(anno_path):
        name = str(entry["image_name"])
        match = _NAME_RE.match(name)
        if match is None:
            problems.append(f"{name}: image_name does not look like 'photoN/imageNNNN'")
            continue
        folder, number = match.groups()
        photo = _with_extension(root, photo_template.format(image_name=name), extensions)
        sketch = _with_extension(root, sketch_template.format(folder=folder, number=number), extensions)
        if photo is None:
            problems.append(f"{name}: photo not found ({photo_template.format(image_name=name)}.*)")
        if sketch is None:
            problems.append(f"{name}: sketch not found ({sketch_template.format(folder=folder, number=number)}.*)")
        if photo is None or sketch is None:
            continue
        pairs.append({"id": pair_id(entry), "photo": photo, "sketch": sketch, "style": style_of(entry, style_field)})
    if problems:
        raise FileNotFoundError(f"{len(problems)} unresolved FS2K entries in {anno_path.name}:\n  " + "\n  ".join(problems))
    return pairs
