"""Task 3 Kaggle runner (tools/t3_pipeline.py) and the packaging of the source checkpoints (tools/package_t3_sources.py).

Most tests need no model code: packaging runs on fixture files with a fake manifest, the supervisor pieces (2-second exit poll,
stall kill, STATUS relay, --set parsing, time projection, budget gate) run on fakes. The full `--dry-run` of the pipeline needs
the training / study / evaluation / export code of Task 3; it is skipped, with the missing names in the reason, until that code exists.
"""
import importlib
import json
import os
import subprocess
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import package_t3_sources as pkg   # noqa: E402
import t3_pipeline as t3           # noqa: E402

from genai.common.checkpoint import sha256_file   # noqa: E402
from genai.tasks.task3 import SOURCE_FILES        # noqa: E402

try:
    from t3_fixtures import make_t3_sources       # noqa: E402  (Agent M's helper)
except ImportError:                                # the fixture file is missing: the tests that need it skip
    make_t3_sources = None

needs_fixtures = pytest.mark.skipif(make_t3_sources is None, reason="tests/t3_fixtures.py (make_t3_sources) not available yet")


# ------------------------------------------------------------------------------ helpers
def write_manifest(path: Path, paths: dict, layout: str = "list") -> Path:
    """A fake models/MANIFEST.json for the fixture checkpoints: entry name = file name without .pt."""
    entries = [{"name": p.stem, "sha256": sha256_file(p), "size": p.stat().st_size, "source": "fixture"} for p in paths.values()]
    path.write_text(json.dumps(entries if layout == "list" else {"version": 1, "models": entries}), encoding="utf-8")
    return path


def make_paths(tmp_path, dry=False) -> "t3.Paths":
    """A Paths object whose log, state and status files live in tmp_path (the real ones are under artifacts/)."""
    p = t3.Paths(dry, tag="pytest")
    p.logs = tmp_path / "logs"
    p.state, p.status, p.summary = p.logs / "t3_state.json", p.logs / "t3_status.log", p.logs / "t3_summary.json"
    return p


def tiny_cfg() -> dict:
    """The config values the time projection reads (pinned here: the test does not depend on configs/task3_moe.yaml)."""
    return {"n_trials": 12, "timeout_minutes": 16, "epochs_per_trial": {"warmup": 1, "joint": 3}, "trial_train_subset": None,
            "train": {"batch_size": 64, "warmup_epochs": 2, "joint_epochs": 12, "val_every_epochs": 1, "train_subset": None},
            "budget": {"max_minutes": 35}}


BENCH = {"warmup_s_per_step": 0.1, "joint_s_per_step": 0.3, "val_full_s": 14.0, "val_subset_s": 4.0, "peak_mb": 3000.0, "params": 4065717}


# ------------------------------------------------------------------------------ packaging
@needs_fixtures
def test_package_writes_zip_with_five_checkpoints_and_sources_json(tmp_path):
    paths, sha = make_t3_sources(tmp_path / "ckpt")
    manifest = write_manifest(tmp_path / "MANIFEST.json", paths)
    out = tmp_path / "dist" / "t3_sources.zip"
    hashes = pkg.package(manifest, tmp_path / "ckpt", out, expected=sha)
    assert set(hashes) == set(SOURCE_FILES.values())                      # the five files
    with zipfile.ZipFile(out) as z:
        assert sorted(z.namelist()) == sorted([*SOURCE_FILES.values(), "SOURCES.json"])
        assert json.loads(z.read("SOURCES.json")) == hashes               # {file: sha256}
        for name, h in hashes.items():                                    # what is in the zip is what was checked
            z.extract(name, tmp_path / "unzipped")
            assert sha256_file(tmp_path / "unzipped" / name) == h
    assert hashes[SOURCE_FILES["salt"]] == sha["salt"]


@needs_fixtures
def test_package_accepts_the_wrapped_manifest_layout(tmp_path):
    paths, sha = make_t3_sources(tmp_path / "ckpt")
    manifest = write_manifest(tmp_path / "MANIFEST.json", paths, layout="wrapped")
    assert len(pkg.package(manifest, tmp_path / "ckpt", tmp_path / "t3.zip", expected=sha)) == 5


@needs_fixtures
def test_package_refuses_a_wrong_task2_hash_and_writes_nothing(tmp_path):
    paths, sha = make_t3_sources(tmp_path / "ckpt")
    manifest = write_manifest(tmp_path / "MANIFEST.json", paths)
    out = tmp_path / "t3.zip"
    with pytest.raises(ValueError, match="t2_ae_salt.pt"):
        pkg.package(manifest, tmp_path / "ckpt", out, expected=dict(sha, salt="0" * 64))
    assert not out.exists() and not list(tmp_path.glob("*.tmp"))


@needs_fixtures
def test_package_refuses_when_the_manifest_disagrees_with_a_file(tmp_path):
    paths, sha = make_t3_sources(tmp_path / "ckpt")
    manifest = write_manifest(tmp_path / "MANIFEST.json", paths)
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    for e in entries:                                                     # the T1 entry is the only check on the T1 file
        if e["name"] == "t1_universal_ae":
            e["sha256"] = "1" * 64
    manifest.write_text(json.dumps(entries), encoding="utf-8")
    with pytest.raises(ValueError, match="t1_universal_ae.pt"):
        pkg.package(manifest, tmp_path / "ckpt", tmp_path / "t3.zip", expected=sha)


@needs_fixtures
def test_package_refuses_a_missing_file_and_a_missing_manifest_entry(tmp_path):
    paths, sha = make_t3_sources(tmp_path / "ckpt")
    manifest = write_manifest(tmp_path / "MANIFEST.json", paths)
    entries = [e for e in json.loads(manifest.read_text(encoding="utf-8")) if e["name"] != "t2_ae_blur"]
    manifest.write_text(json.dumps(entries), encoding="utf-8")
    with pytest.raises(ValueError, match="no entry 't2_ae_blur'"):
        pkg.package(manifest, tmp_path / "ckpt", tmp_path / "t3.zip", expected=sha)
    paths["occlusion"].unlink()
    with pytest.raises(ValueError, match="t2_ae_occlusion.pt: file not found"):
        pkg.package(manifest, tmp_path / "ckpt", tmp_path / "t3.zip", expected=sha)


@needs_fixtures
def test_package_without_t1_has_four_files(tmp_path):
    paths, sha = make_t3_sources(tmp_path / "ckpt")
    manifest = write_manifest(tmp_path / "MANIFEST.json", paths)
    hashes = pkg.package(manifest, tmp_path / "ckpt", tmp_path / "t3.zip", expected=sha, include_t1=False)
    assert SOURCE_FILES["t1"] not in hashes and len(hashes) == 4


@needs_fixtures
def test_package_cli_refuses_fixture_files_because_the_default_hashes_are_the_real_task2_ones(tmp_path, monkeypatch):
    paths, _ = make_t3_sources(tmp_path / "ckpt")
    manifest = write_manifest(tmp_path / "MANIFEST.json", paths)
    monkeypatch.setattr(sys, "argv", ["package_t3_sources.py", "--manifest", str(manifest), "--ckpt-dir", str(tmp_path / "ckpt"),
                                      "--out", str(tmp_path / "t3.zip")])
    with pytest.raises(SystemExit) as err:
        pkg.main()
    assert "refusing to package" in str(err.value)
    assert not (tmp_path / "t3.zip").exists()


REAL_MANIFEST = ROOT / "models" / "MANIFEST.json"
REAL_FILES = [ROOT / "models" / "checkpoints" / f for f in SOURCE_FILES.values()]


@pytest.mark.skipif(not REAL_MANIFEST.exists() or not all(f.exists() for f in REAL_FILES),
                    reason="the real checkpoints / models/MANIFEST.json are not in this checkout")
def test_package_the_real_checkpoints_into_tmp(tmp_path):
    """Read-only on models/: the real files and the real manifest layout pass the real Task 2 hashes."""
    hashes = pkg.package(REAL_MANIFEST, ROOT / "models" / "checkpoints", tmp_path / "t3_sources.zip")
    assert len(hashes) == 5
    assert hashes["t2_classifier.pt"] == "7d4a9a1f8a073e64e8cc21deec4db98d51ad48dc04eb9b5b4215f49102ece028"


# ------------------------------------------------------------------------------ supervisor pieces
def test_constants():
    assert t3.PIPELINE_VERSION == "t3-v1"
    assert t3.POLL_SECONDS == 2 and t3.STALL_CHECK_SECONDS == 30
    assert t3.STAGES == ["prepare", "study", "final_train", "evaluate", "export_promote"]


class FakeProc:
    """Stands in for subprocess.Popen: poll() says 'still running' `polls` times, then 'exited with code 0'."""
    pid, returncode = 4242, 0

    def __init__(self, polls):
        self.polls = polls

    def poll(self):
        if self.polls > 0:
            self.polls -= 1
            return None
        return 0

    def wait(self):
        return self.returncode


def test_exit_check_sleeps_poll_seconds_between_polls(tmp_path, monkeypatch):
    p = make_paths(tmp_path)
    p.logs.mkdir(parents=True)
    sleeps = []
    monkeypatch.setattr(t3.subprocess, "Popen", lambda *a, **k: FakeProc(3))
    monkeypatch.setattr(t3.time, "sleep", sleeps.append)
    code = t3.run_child(p, {"stages": {"prepare": {"attempts": 1}}}, "prepare")
    assert code == 0
    assert sleeps == [t3.POLL_SECONDS] * 3        # one sleep of POLL_SECONDS before each of the three 'still running' answers


def test_stalled_child_is_killed(tmp_path, monkeypatch):
    p = make_paths(tmp_path)
    p.logs.mkdir(parents=True)
    killed = []
    proc = FakeProc(10 ** 6)                       # never exits by itself
    monkeypatch.setattr(t3.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(t3.time, "sleep", lambda s: None)
    monkeypatch.setattr(t3, "STALL_CHECK_SECONDS", 0)      # check at every poll ...
    monkeypatch.setattr(t3, "STALL_MINUTES", -1)           # ... and treat any log age as 'silent for too long'
    monkeypatch.setattr(t3, "kill_tree", killed.append)
    assert t3.run_child(p, {"stages": {"study": {"attempts": 1}}}, "study") == -9
    assert killed == [4242]
    assert "killing the stuck process" in p.status.read_text(encoding="utf-8")


def test_status_lines_of_a_child_reach_the_status_log(tmp_path):
    p = make_paths(tmp_path)
    log = tmp_path / "child.log"
    log.write_bytes(b"noise\nSTATUS: sources verified\nmore noise\nSTATUS: benchmark 0.1 s/step\nSTATUS: half a li")
    offset = t3.relay_status_lines(p, "prepare", log, 0)
    text = p.status.read_text(encoding="utf-8")
    assert "prepare: sources verified" in text and "prepare: benchmark 0.1 s/step" in text
    assert "half a li" not in text                                      # an incomplete last line waits for the next call
    with open(log, "ab") as f:
        f.write(b"ne\n")
    t3.relay_status_lines(p, "prepare", log, offset)
    assert "prepare: half a line" in p.status.read_text(encoding="utf-8")


def test_set_overrides_parse_and_apply():
    assert t3.parse_set(["timeout_minutes=10", "train.joint_epochs=8", "run.smoke=true", "x.lr=1e-4", "name=abc"]) == {
        "timeout_minutes": 10, "train.joint_epochs": 8, "run.smoke": True, "x.lr": 1e-4, "name": "abc"}
    with pytest.raises(ValueError, match="KEY=VALUE"):
        t3.parse_set(["timeout_minutes"])
    cfg = {"timeout_minutes": 16, "train": {"joint_epochs": 12}}
    t3.apply_budget(cfg, {"timeout_minutes": 10, "train.joint_epochs": 8}, strict=True)
    assert cfg == {"timeout_minutes": 10, "train": {"joint_epochs": 8}}
    with pytest.raises(ValueError, match="train.joint_epoch"):          # a typo is refused, not ignored
        t3.apply_budget(cfg, {"train.joint_epoch": 8}, strict=True)
    t3.apply_budget(cfg, {"train.other": 1}, strict=False)              # the final config may lack a key
    assert cfg["train"]["other"] == 1


def test_set_items_reach_the_child_command_line():
    p = t3.Paths(True, "t", "kaggle", set_items=["timeout_minutes=10", "train.joint_epochs=8"])
    args = p.cli_args()
    assert args.count("--set") == 2 and "timeout_minutes=10" in args and "--dry-run" in args
    assert p.budget == {"timeout_minutes": 10, "train.joint_epochs": 8}


def test_time_projection_follows_the_documented_rule():
    cfg, proj = tiny_cfg(), None
    proj = t3.project_minutes(cfg, BENCH)
    # trial = (1 x 0.1 + 3 x 0.3) x 46 + 4 x 4 + 5 = 67 s ; 12 trials = 13.4 min (below the 16 min cap + one trial)
    assert proj["trial_s"] == 67.0 and proj["study_min"] == 13.4
    # final = (2 x 0.1 + 12 x 0.3) x 46 + (14 + 1) validations x 14 s = 174.8 + 210 s = 6.4 min
    assert proj["final_min"] == 6.4
    assert proj["total_min"] == round(2 + 13.4 + 6.4 + 2 + 1 + 1, 1) and proj["limit_min"] == 35
    cfg["timeout_minutes"] = 5                                          # the timeout cap wins: 5 min + one trial
    assert t3.project_minutes(cfg, BENCH)["study_min"] == round((5 * 60 + 67) / 60, 1)
    cfg["train"]["val_every_epochs"] = 2                                # 7 + 1 validations instead of 15
    assert t3.project_minutes(cfg, BENCH)["final_min"] == round(((2 * 0.1 + 12 * 0.3) * 46 + 8 * 14) / 60, 1)


def test_range_edge_flags():
    space = [{"name": "joint_lr", "type": "float", "low": 1e-5, "high": 2e-4, "log": True},
             {"name": "recon_l1_share", "type": "float", "low": 0.3, "high": 0.95, "log": False},
             {"name": "tau", "type": "float", "low": 0.5, "high": 5.0, "log": True},
             {"name": "choice", "type": "categorical", "choices": [1, 2]}]
    flags = t3.range_edge_flags(space, {"joint_lr": 1.1e-5, "recon_l1_share": 0.94, "tau": 1.6, "choice": 2})
    assert flags["joint_lr"]["edge"] == "low" and flags["recon_l1_share"]["edge"] == "high"
    assert flags["tau"]["edge"] is None and 0.4 < flags["tau"]["position"] < 0.6
    assert "choice" not in flags                                        # only float ranges have ends


# ------------------------------------------------------------------------------ the budget gate of `prepare`
def run_prepare(monkeypatch, tmp_path, dry, seconds_scale):
    """Run stage_prepare with fake config, sources and benchmark; returns the SystemExit code or None."""
    import genai.tasks.task3.train as train_mod
    bench = {k: (v * seconds_scale if k.endswith(("_s_per_step", "_s")) else v) for k, v in BENCH.items()}
    cfg = tiny_cfg() | {"train": tiny_cfg()["train"] | {"amp": True}}
    notes = {}
    monkeypatch.setattr(train_mod, "measure_step_times", lambda *a, **k: bench, raising=False)
    monkeypatch.setattr(t3, "study_cfg", lambda p, shrink=True: cfg)
    monkeypatch.setattr(t3, "check_sources", lambda c: {"classifier": "a" * 64})
    monkeypatch.setattr(t3, "state_note", lambda p, key, value: notes.update({key: value}))
    try:
        t3.stage_prepare(make_paths(tmp_path, dry=dry), {})
    except SystemExit as err:
        return err.code, notes
    return None, notes


def test_prepare_passes_when_the_projection_fits(monkeypatch, tmp_path, capsys):
    code, notes = run_prepare(monkeypatch, tmp_path, dry=False, seconds_scale=1.0)
    out = capsys.readouterr().out
    assert code is None and notes["benchmark"]["projection"]["total_min"] < 35
    assert "sources verified (sha256): classifier=" in out and "projected minutes of the REAL run" in out
    assert t3.STATUS_MARK + "benchmark" in out and notes["sources_before"] == {"classifier": "a" * 64}


def test_prepare_stops_with_the_gate_code_when_too_slow_but_not_in_a_dry_run(monkeypatch, tmp_path, capsys):
    code, _ = run_prepare(monkeypatch, tmp_path, dry=False, seconds_scale=3.0)       # projection about 70 min
    assert code == t3.GATE_EXIT_CODE
    assert "GATE" in capsys.readouterr().out
    code, notes = run_prepare(monkeypatch, tmp_path, dry=True, seconds_scale=3.0)    # the dry run only reports it
    assert code is None and notes["benchmark"]["projection"]["total_min"] > 35


def test_supervisor_stops_at_once_on_the_gate_code_without_retries(monkeypatch, tmp_path):
    p = make_paths(tmp_path)
    calls = []
    monkeypatch.setattr(t3, "preflight", lambda p: None)
    monkeypatch.setattr(t3, "run_child", lambda p, state, stage: calls.append(stage) or t3.GATE_EXIT_CODE)
    assert t3.supervise(p) == 2
    assert calls == ["prepare"]                                          # one attempt, no pause, no later stage
    assert "STOPPED" in p.status.read_text(encoding="utf-8")


def test_supervisor_stops_after_repeated_failures_without_progress(monkeypatch, tmp_path):
    p = make_paths(tmp_path)
    calls = []
    monkeypatch.setattr(t3, "preflight", lambda p: None)
    monkeypatch.setattr(t3, "run_child", lambda p, state, stage: calls.append(stage) or 1)
    monkeypatch.setattr(t3.time, "sleep", lambda s: None)                # no 60 s retry pause in a test
    p.stage_log("prepare").parent.mkdir(parents=True, exist_ok=True)
    p.stage_log("prepare").write_text("Traceback: boom\n", encoding="utf-8")
    assert t3.supervise(p) == 1
    assert calls == ["prepare"] * t3.MAX_CONSECUTIVE_FAILURES


# ------------------------------------------------------------------------------ the full dry run (needs the Task 3 code)
REQUIRED = {"genai.tasks.task3.train": ("run_training", "measure_step_times"),
            "genai.tasks.task3.tune": ("run_study", "write_final_config", "dry_run_overrides"),
            "genai.tasks.task3.evaluate": ("run_evaluation",),
            "genai.tasks.task3.sources": ("find_sources", "verify_sources"),
            "genai.export.task3_export": ("export_t3", "verify_t3_parity")}


def missing_task3_code() -> list:
    """Names from plan D1 to D7 that the stages call and that do not exist yet (stubs raise NotImplementedError)."""
    missing = []
    for module, names in REQUIRED.items():
        try:
            mod = importlib.import_module(module)
        except ImportError:
            missing.append(f"{module} (module)")
            continue
        missing += [f"{module}.{n}" for n in names if not hasattr(mod, n)]
    if not (ROOT / "configs" / "task3_moe.yaml").exists():
        missing.append("configs/task3_moe.yaml")
    return missing


@needs_fixtures
def test_dry_run_end_to_end_in_a_fake_nested_input_folder(tmp_path, tiny_pets_root):
    """Plan section E: `--dry-run --data-root <fake input>`: tiny pets data + the unzipped t3_sources.zip under nested folders."""
    missing = missing_task3_code()
    if missing:
        pytest.skip("Task 3 code not complete yet, missing: " + ", ".join(missing))

    # 1. the sources, packaged exactly like the student does, then unzipped into a nested folder like Kaggle's /kaggle/input
    paths, sha = make_t3_sources(tmp_path / "ckpt")
    manifest = write_manifest(tmp_path / "MANIFEST.json", paths)
    pkg.package(manifest, tmp_path / "ckpt", tmp_path / "t3_sources.zip", expected=sha)
    inputs = tmp_path / "input"
    with zipfile.ZipFile(tmp_path / "t3_sources.zip") as z:
        z.extractall(inputs / "datasets" / "me" / "t3-sources")
    shutil.copytree(tiny_pets_root, inputs / "datasets" / "me" / "pets-data")     # 64 train images, 64 val rows
    (tmp_path / "sha.json").write_text(json.dumps(sha), encoding="utf-8")         # fixture hashes (the real run uses TASK2_SHA256)
    tmp_dir = ROOT / "artifacts" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 2. the pipeline in a child process; --set shrinks the dry-run budgets further to the tiny dataset
    cmd = [sys.executable, str(ROOT / "tools" / "t3_pipeline.py"), "--dry-run", "--data-root", str(inputs),
           "--expected-sources", str(tmp_path / "sha.json"), "--tag", "pytest_t3",
           "--set", "train.batch_size=16", "--set", "train.train_subset=32", "--set", "trial_train_subset=32",
           "--set", "train.val_subset=32", "--set", "trial_val_subset=32"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", TMP=str(tmp_dir), TEMP=str(tmp_dir), TRACKER="none")
    done = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1500)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-2000:]
    assert "ALL STAGES DONE" in done.stdout

    # 3. what it wrote
    logs = ROOT / "artifacts" / "dryrun_t3" / "logs"
    summary = json.loads((logs / "t3_summary.json").read_text(encoding="utf-8"))
    assert summary["pipeline_version"] == "t3-v1" and summary["dry_run"] is True
    assert sum(summary["study"]["counts"].values()) >= 2
    assert summary["sources"]["unchanged"] is True and summary["sources"]["before"]
    assert summary["parity"]["passed"] is True and summary["benchmark"]["projection"]["total_min"] > 0
    assert summary["evaluation"]["summary"] is not None and summary["final_curve"]
    assert all(summary["stage_seconds"][s] is not None for s in t3.STAGES)
    assert "sources verified (sha256)" in (logs / "t3_status.log").read_text(encoding="utf-8")
