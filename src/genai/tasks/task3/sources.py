"""Finding and checking the source checkpoints of Task 3 (the four Task 2 files, and Task 1 as an option).

Task 3 starts from the finished Task 2 checkpoints. They are read-only and their sha256 is checked
before anything is loaded (CONTRACTS 3.8, decision B12):

    find_sources(cfg)       where are the files?       -> {"classifier": Path, ..., "t1": Path or None}
    verify_sources(paths)   are they the right bytes?  -> {name: sha256}, ValueError otherwise
    source_record(paths)    what goes into checkpoints / summary.json -> {name: {"file", "sha256"}}

Nothing here ever creates random weights as a substitute for a missing or wrong file.
"""
from __future__ import annotations

from pathlib import Path

from genai.common.checkpoint import sha256_file
from genai.common.paths import MODELS_CKPT
from genai.tasks.task1.config import resolve
from genai.tasks.task3 import EXPERT_NAMES, SOURCE_FILES, TASK2_SHA256

REQUIRED = ("classifier", *EXPERT_NAMES)      # the four Task 2 files; "t1" is optional


def _find_in(folder: Path, names) -> dict:
    """{name: Path} for every name whose file exists somewhere under `folder` (recursive).

    The Kaggle input folder nests datasets (input/datasets/<user>/<dataset>/...), hence the recursive
    search. If a file name occurs more than once, the shallowest path wins (then alphabetical order);
    a wrong file would be caught by the hash check anyway.
    """
    found = {}
    for name in names:
        matches = sorted(folder.rglob(SOURCE_FILES[name]), key=lambda p: (len(p.parts), str(p)))
        matches = [p for p in matches if p.is_file()]
        if matches:
            found[name] = matches[0]
    return found


def find_sources(cfg: dict, need_t1: bool = False) -> dict:
    """Locate the five source checkpoints. Returns {"classifier", "salt", "blur", "occlusion", "t1"} -> Path
    (the value of "t1" is None when Task 1's file was not found and need_t1 is False).

    Search order, the first place that holds ALL required files wins:
      1. cfg["sources"]["dir"]       an explicit folder (when set)
      2. cfg["data_root"]            searched recursively (Kaggle: /kaggle/input with nested dataset folders)
      3. the repository's models/checkpoints
    There is no hidden "local" default: data_root comes from the run's config (decision D46). If no place
    holds all required files, FileNotFoundError lists every place that was searched and what it lacked.
    """
    required = REQUIRED + (("t1",) if need_t1 else ())
    places = []                                   # (label, folder or None)
    explicit = (cfg.get("sources") or {}).get("dir")
    if explicit:
        places.append(("cfg['sources']['dir']", resolve(explicit)))
    if cfg.get("data_root"):
        places.append(("cfg['data_root']", resolve(cfg["data_root"])))
    places.append(("repository models/checkpoints", Path(MODELS_CKPT)))

    report = []
    for label, folder in places:
        if not folder.is_dir():
            report.append(f"  {label}: {folder} (folder does not exist)")
            continue
        found = _find_in(folder, SOURCE_FILES)    # looks for all five names
        missing = [SOURCE_FILES[n] for n in required if n not in found]
        if not missing:
            return {name: found.get(name) for name in SOURCE_FILES}
        report.append(f"  {label}: {folder} (missing: {', '.join(missing)})")
    if not cfg.get("data_root"):
        report.append("  cfg['data_root'] is not set, so no data folder was searched")
    raise FileNotFoundError("Task 3 source checkpoints not found. Places searched:\n" + "\n".join(report))


def verify_sources(paths: dict, expected: dict = TASK2_SHA256, t1_sha: str | None = None) -> dict:
    """Hash the source files and compare with the expected sha256 values. Returns {name: sha256}.

    expected: {name: sha256} of the files that MUST be present and identical (default: the four Task 2
              checkpoints). A "t1" key inside it is treated like the others but may be absent from paths.
    t1_sha:   expected sha256 of the Task 1 file; checked only when given. If paths["t1"] is set it is
              hashed and returned either way.
    All problems are collected and raised together as ONE ValueError that names every file with both
    hashes (expected and found), or the missing path.
    """
    expected = dict(expected)
    if t1_sha is not None:
        expected["t1"] = t1_sha

    names = list(expected)
    if paths.get("t1") and "t1" not in names:
        names.append("t1")                        # Task 1's file is hashed when it is there, even if nothing is expected

    problems, result = [], {}
    for name in names:
        path = paths.get(name)
        file_name = SOURCE_FILES.get(name, name)
        if name == "t1" and path is None and t1_sha is None:
            continue                              # Task 1 is optional: absent and no hash demanded is fine
        if path is None or not Path(path).is_file():
            problems.append(f"{name} ({file_name}): file not found at {path!r}")
            continue
        found = sha256_file(path)
        result[name] = found
        want = expected.get(name)
        if want is not None and found != want:
            problems.append(f"{name} ({file_name}): sha256 mismatch, expected {want}, found {found} ({path})")
    if problems:
        raise ValueError("source checkpoints failed the check (never replaced by random weights):\n  "
                         + "\n  ".join(problems))
    return result


def source_record(paths: dict) -> dict:
    """{name: {"file": base name, "sha256": hex}} for every source that exists (None entries are skipped).

    Stored as config.source_checkpoints in the Task 3 checkpoints and in summary.json.
    """
    return {name: {"file": Path(p).name, "sha256": sha256_file(p)} for name, p in paths.items() if p is not None}
