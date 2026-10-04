"""Unattended Task 3 pipeline (Kaggle or local), modelled on tools/t2_pipeline.py. Plan: docs/TASK3_PLAN.md (section C, D10).

Task 3 = the jointly trained soft mixture of experts (gate = Task 2 classifier, experts = the three Task 2 specialists).
Stages, in order (every one resumable; finished stages are skipped on a re-run):
    prepare          find the five source checkpoints and check their sha256; measure seconds per step on THIS GPU;
                     project the minutes of the real run and STOP (exit code 3) if the projection is above budget.max_minutes
    study            Optuna study t3_moe                                  (configs/task3_moe.yaml)
    final_train      configs/task3_moe_final.yaml = study config + best trial; then the final training (warm-up + joint)
    evaluate         validation evaluation: input, Task 1, Task 2 (oracle and predicted), Task 3 on the same tensors
    export_promote   ONNX (outputs "output" and "weights") + numeric parity, promotion of t3_soft_moe,
                     second sha256 check of the five source checkpoints (they must be unchanged)

Usage (repository root):
    python tools/t3_pipeline.py --device-profile kaggle                  # notebooks/kaggle_t3_pipeline.ipynb does this
    python tools/t3_pipeline.py --dry-run                                # tiny end-to-end check in artifacts/dryrun_t3/
    python tools/t3_pipeline.py --dry-run --data-root <folder>           # rehearse a Kaggle-like nested input folder locally
    python tools/t3_pipeline.py --device-profile kaggle --set timeout_minutes=10 --set train.joint_epochs=8

--set KEY=VALUE (repeatable) overrides one budget value (dotted key as in the YAML, value read as YAML). The key must exist
in configs/task3_moe.yaml (a typo stops the run). Overrides are applied to the study config and to the final config, and they
are echoed into the status log. Plan section C lists the order in which to cut the budget with them.

Differences to tools/t2_pipeline.py: 5 stages instead of 11; the exit check of a child runs every POLL_SECONDS = 2 s (Task 2 slept
30 s before every check and lost up to 30 s per stage; the stall check still runs every 30 s); the `prepare` stage with the
benchmark and the budget gate; source hashes checked at the start (prepare) and at the end (export_promote).
The supervisor code repeats tools/t2_pipeline.py (a Task 2 file that is not edited); candidate refactor (decisions D22, D44).

Never reads the test set: evaluation is on the VAL manifest only (final_test=False everywhere).
No data root defaults to "local" anywhere: every call gets the run's config, so it works under /kaggle/input (D46).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genai.tasks.task3 import ONNX_KEY, PROMOTED_NAME, RUN_GROUP, SOURCE_FILES, TASK2_SHA256   # noqa: E402

PIPELINE_VERSION = "t3-v1"   # printed by the notebook (Cell 1) and the supervisor: proves which code Kaggle runs

CONFIG = "configs/task3_moe.yaml"
FINAL_CONFIG = "configs/task3_moe_final.yaml"
STAGES = ["prepare", "study", "final_train", "evaluate", "export_promote"]
POLL_SECONDS = 2                         # a finished child is noticed within 2 s (Task 2: 30 s)
STALL_CHECK_SECONDS = 30                 # the stall check (and the copy of STATUS lines) every 30 s, as before
STALL_MINUTES = 20                       # one epoch + validation takes well under a minute
MAX_CONSECUTIVE_FAILURES = 4             # without progress
RETRY_PAUSE_SECONDS = 60
GATE_EXIT_CODE = 3                       # `prepare` exits with it when the projected time is above the budget: no retries
STATUS_MARK = "STATUS: "                 # a child prints lines that start with this; the supervisor copies them to the status log
WANDB_ENV = {"TRACKER": "wandb", "WANDB_ENTITY": "ahmedlaiq34", "WANDB_PROJECT": "genai-a1",
             "WANDB_INIT_TIMEOUT": "300", "WANDB_SILENT": "false"}

# Benchmark and time projection (plan section C). Values marked "estimate" come from the plan, not from a measurement.
BENCH_STEPS, BENCH_WARMUP = 20, 5        # timed steps and warm-up steps of each step type
TRAIN_IMAGES = 2944                      # training images of the pets split: 46 steps per epoch at batch 64 (plan B10)
TRIAL_OVERHEAD_S = 5                     # estimate: reload the four source checkpoints and build the loaders per trial
PREPARE_MIN, EVALUATE_MIN, EXPORT_MIN, OVERHEAD_MIN = 2, 2, 1, 1   # estimates (plan C, upper values; overhead = 5 child starts + polling)
CUT_LIST = ("timeout_minutes=10", "train.joint_epochs=8", "epochs_per_trial.joint=2", "train.val_every_epochs=2")   # plan C, in this order


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------------------------ paths and state
class Paths:
    """Every file the pipeline writes. The dry run redirects all of them into artifacts/dryrun_t3/."""

    def __init__(self, dry: bool, tag: str = "t3", profile: str = "local", num_workers=None, data_root=None,
                 set_items=None, expected_sources=None):
        self.dry, self.tag, self.profile = dry, tag, profile
        self.data_root = data_root       # optional override of the profile's data_root (tests a Kaggle-like layout locally)
        # 0 data workers locally (Windows worker crash, D19); elsewhere the profile's own value
        self.num_workers = num_workers if num_workers is not None else (0 if profile == "local" else None)
        self.set_items = list(set_items or [])          # the raw "KEY=VALUE" strings of --set
        self.budget = parse_set(self.set_items)         # {dotted key: value}
        self.expected_sources = expected_sources        # tests only: JSON file with other sha256 values (see overrides())
        self.tmp = ROOT / "artifacts" / "dryrun_t3"
        self.logs = (self.tmp / "logs") if dry else (ROOT / "artifacts" / "logs" / tag)
        self.state = self.logs / "t3_state.json"
        self.status = self.logs / "t3_status.log"
        self.summary = self.logs / "t3_summary.json"
        self.final_yaml = (self.tmp / "task3_moe_final.yaml") if dry else (ROOT / FINAL_CONFIG)
        self.onnx_dir = (self.tmp / "onnx") if dry else (ROOT / "models" / "onnx")
        self.parity_csv = (self.tmp / "onnx_parity.csv") if dry else (ROOT / "report" / "tables" / "onnx_parity.csv")
        self.models_root = (self.tmp / "models") if dry else None      # None = the real models/ folder

    def stage_log(self, stage: str) -> Path:
        return self.logs / f"t3_{stage}.log"

    def overrides(self) -> dict:
        """Dotted overrides for load_config: workers, data root, and (tests only) the expected source hashes."""
        out = {} if self.num_workers is None else {"num_workers": self.num_workers}
        if self.data_root:
            out["data_root"] = self.data_root
        if self.expected_sources:
            out["sources.expected_sha256"] = json.loads(Path(self.expected_sources).read_text(encoding="utf-8"))
        return out

    def cli_args(self) -> list:
        args = ["--tag", self.tag, "--device-profile", self.profile]
        if self.num_workers is not None:
            args += ["--num-workers", str(self.num_workers)]
        if self.data_root:
            args += ["--data-root", self.data_root]
        if self.expected_sources:
            args += ["--expected-sources", self.expected_sources]
        for item in self.set_items:
            args += ["--set", item]
        return args + (["--dry-run"] if self.dry else [])


def load_state(p: Paths) -> dict:
    if p.state.exists():
        return json.loads(p.state.read_text(encoding="utf-8"))
    return {"stages": {s: {"done": False, "attempts": 0} for s in STAGES}, "wandb_mode": "online", "run_id": None}


def save_state(p: Paths, state: dict) -> None:
    p.logs.mkdir(parents=True, exist_ok=True)
    tmp = p.state.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, p.state)


def state_note(p: Paths, key: str, value) -> None:
    """Child processes record results in the state file (the supervisor re-reads it)."""
    state = load_state(p)
    state.setdefault("results", {})[key] = value
    save_state(p, state)


def status(p: Paths, msg: str) -> None:
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    p.logs.mkdir(parents=True, exist_ok=True)
    with open(p.status, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def say(msg: str) -> None:
    """Called inside a child stage: the supervisor copies this line into the status log (and the notebook output)."""
    print(STATUS_MARK + msg, flush=True)


# ------------------------------------------------------------------------------ configs and --set overrides
def parse_set(items) -> dict:
    """['timeout_minutes=10', 'train.joint_epochs=8'] -> {'timeout_minutes': 10, 'train.joint_epochs': 8}."""
    import yaml
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--set expects KEY=VALUE (for example timeout_minutes=10), got {item!r}")
        value = yaml.safe_load(value)                  # "10" -> 10, "true" -> True, "0.5" -> 0.5
        if isinstance(value, str) and any(c.isdigit() for c in value):
            try:
                value = float(value)                   # YAML reads "1e-4" as text (it wants "1.0e-4"); a learning rate is a number
            except ValueError:
                pass
        out[key.strip()] = value
    return out


def has_key(cfg: dict, dotted: str) -> bool:
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def apply_budget(cfg: dict, budget: dict, strict: bool) -> dict:
    """Apply the --set values. strict=True refuses a key that is not in the config (a typo would be ignored otherwise)."""
    from genai.tasks.task1.config import set_dotted
    for dotted, value in budget.items():
        if strict and not has_key(cfg, dotted):
            raise ValueError(f"--set {dotted}: this key is not in {CONFIG}")
        set_dotted(cfg, dotted, value)
    return cfg


def study_cfg(p: Paths, shrink: bool = True) -> dict:
    """Study config = configs/task3_moe.yaml + device profile. The dry run shrinks it (unless shrink=False);
    the --set overrides come last, so they win over the dry-run values."""
    from genai.tasks.task1.config import load_config
    cfg = load_config(CONFIG, p.profile, p.overrides())
    if p.dry and shrink:
        from genai.tasks.task3.tune import dry_run_overrides
        cfg = dry_run_overrides(cfg, p.tmp)                     # study name + "_dryrun", tmp folders, run.smoke
        cfg.update(n_trials=2, timeout_minutes=10, trial_train_subset=256, trial_val_subset=64)
        cfg["epochs_per_trial"] = {"warmup": 1, "joint": 1}
        cfg["pruner"] = {"name": "Median", "n_startup_trials": 1, "n_warmup_steps": 1}
        cfg["train"].update(warmup_epochs=1, joint_epochs=1, train_subset=256, val_subset=64)
    return apply_budget(cfg, p.budget, strict=True)


def final_cfg(p: Paths) -> dict:
    """Final config = the file the final_train stage wrote (study config + best trial) + device profile.
    Evaluation and export also take this config as their data root (D46)."""
    from genai.tasks.task1.config import load_config
    cfg = load_config(p.final_yaml, p.profile, p.overrides())
    if p.dry:
        cfg["output_root"] = cfg["persist_root"] = str(p.tmp)
        cfg["train"].update(warmup_epochs=1, joint_epochs=1, train_subset=256, val_subset=64, sample_every_epochs=1)
        cfg["run"]["smoke"] = True
    return apply_budget(cfg, p.budget, strict=False)            # strict only at the study (a key may be absent here)


def expected_hashes(cfg: dict) -> dict:
    """sha256 the Task 2 files must have: TASK2_SHA256, unless a test supplies fixture hashes in cfg['sources']."""
    return (cfg.get("sources") or {}).get("expected_sha256") or TASK2_SHA256


def check_sources(cfg: dict) -> dict:
    """Find the source checkpoints (under the run's data_root, nested folders allowed), check their sha256
    against TASK2_SHA256 (and T1 against SOURCES.json of the package). Returns {name: sha256}. Never loads weights."""
    from genai.tasks.task3.sources import find_sources, verify_sources
    paths = find_sources(cfg)
    t1_sha = None
    sources_json = Path(paths["classifier"]).parent / "SOURCES.json"      # written by tools/package_t3_sources.py
    if paths.get("t1") and sources_json.exists():
        t1_sha = json.loads(sources_json.read_text(encoding="utf-8")).get(SOURCE_FILES["t1"])
    return verify_sources(paths, expected=expected_hashes(cfg), t1_sha=t1_sha)


def open_study(cfg: dict):
    import optuna
    from genai.tasks.task1.config import resolve
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.load_study(study_name=cfg["study"], storage=f"sqlite:///{resolve(cfg['storage']).as_posix()}")


def study_counts(cfg: dict) -> dict:
    try:
        study = open_study(cfg)
    except Exception:
        return {}
    counts = {}
    for t in study.trials:
        counts[t.state.name] = counts.get(t.state.name, 0) + 1
    return counts


def run_dir(p: Paths, state: dict) -> Path:
    """<output_root>/runs/task3/<run_id>: the folder of the final training."""
    from genai.tasks.task1.config import load_config, resolve
    # not final_cfg(p): that file does not exist before the final_train stage has written it
    root = str(p.tmp) if p.dry else load_config(CONFIG, p.profile, p.overrides())["output_root"]
    return resolve(root) / "runs" / RUN_GROUP / state["run_id"]


def best_ckpt(p: Paths, state: dict) -> Path:
    return run_dir(p, state) / "ckpt_best.pt"


# ------------------------------------------------------------------------------ time projection (pure functions)
def steps_per_epoch(n_images, batch_size: int) -> int:
    """Training steps of one epoch: all training images (or the subset) in balanced batches."""
    return max(1, int(n_images or TRAIN_IMAGES) // int(batch_size))


def project_minutes(cfg: dict, bench: dict) -> dict:
    """Projected minutes of the REAL run from the measured seconds per step (plan section C). Estimates, not measurements.

    trial   = (warm-up epochs x warm-up step + joint epochs x joint step) x steps per epoch
              + one subset validation per epoch + TRIAL_OVERHEAD_S
    study   = min(n_trials x trial, timeout_minutes x 60 + one trial)   (Optuna lets the running trial finish after the timeout)
    final   = (warm-up epochs x warm-up step + joint epochs x joint step) x steps per epoch
              + one full validation per val_every_epochs epochs + one more (epoch 0)
    total   = prepare + study + final + evaluate + export + overhead (the last four are the plan's estimates)
    """
    train, ept = cfg["train"], cfg["epochs_per_trial"]
    w, j = bench["warmup_s_per_step"], bench["joint_s_per_step"]
    trial_steps = steps_per_epoch(cfg.get("trial_train_subset"), train["batch_size"])
    trial_s = ((ept["warmup"] * w + ept["joint"] * j) * trial_steps
               + (ept["warmup"] + ept["joint"]) * bench["val_subset_s"] + TRIAL_OVERHEAD_S)
    study_s = min(cfg["n_trials"] * trial_s, cfg["timeout_minutes"] * 60 + trial_s)
    final_steps = steps_per_epoch(train.get("train_subset"), train["batch_size"])
    epochs = train["warmup_epochs"] + train["joint_epochs"]
    n_val = math.ceil(epochs / max(1, int(train.get("val_every_epochs", 1)))) + 1
    final_s = (train["warmup_epochs"] * w + train["joint_epochs"] * j) * final_steps + n_val * bench["val_full_s"]
    total = PREPARE_MIN + study_s / 60 + final_s / 60 + EVALUATE_MIN + EXPORT_MIN + OVERHEAD_MIN
    return {"trial_s": round(trial_s, 1), "study_min": round(study_s / 60, 1), "final_min": round(final_s / 60, 1),
            "prepare_min": PREPARE_MIN, "evaluate_min": EVALUATE_MIN, "export_min": EXPORT_MIN,
            "overhead_min": OVERHEAD_MIN, "total_min": round(total, 1), "limit_min": cfg["budget"]["max_minutes"]}


def range_edge_flags(space: list, params: dict) -> dict:
    """For each float parameter of the study: where the best value sits in its range (0 = low end, 1 = high end, on the
    log scale for log parameters). 'edge' is 'low' / 'high' when it is within 10 % of an end (plan section G)."""
    flags = {}
    for spec in space:
        name = spec["name"]
        if spec["type"] != "float" or name not in params:
            continue
        lo, hi, v = spec["low"], spec["high"], params[name]
        if spec.get("log"):
            lo, hi, v = math.log(lo), math.log(hi), math.log(v)
        pos = (v - lo) / (hi - lo)
        flags[name] = {"value": params[name], "position": round(pos, 3),
                       "edge": "low" if pos <= 0.1 else "high" if pos >= 0.9 else None}
    return flags


# ------------------------------------------------------------------------------ the stages (child process)
def stage_prepare(p, state):
    """Sources + benchmark + gate. Prints STATUS lines that the supervisor copies into the status log."""
    from genai.tasks.task1.config import is_tbd
    from genai.tasks.task3.train import measure_step_times
    cfg = study_cfg(p, shrink=False)          # the REAL budgets: the projection is about the real run, also in the dry run
    if p.budget:
        say(f"budget overrides (--set): {p.budget}")
    ept, train = cfg["epochs_per_trial"], cfg["train"]
    say(f"budgets in effect: study n_trials={cfg['n_trials']} timeout_minutes={cfg['timeout_minutes']} "
        f"epochs_per_trial={ept['warmup']}+{ept['joint']} (warm-up+joint) | final {train['warmup_epochs']}+{train['joint_epochs']} epochs, "
        f"val every {train.get('val_every_epochs', 1)} | batch {train['batch_size']}, amp {train['amp']}")
    for key, value in (("n_trials", cfg["n_trials"]), ("timeout_minutes", cfg["timeout_minutes"]), *ept.items()):
        if is_tbd(value):
            raise RuntimeError(f"{key} in {CONFIG} is still TBD_AFTER_BENCHMARK: set it (or pass --set) and run again")

    hashes = check_sources(cfg)                                # raises ValueError on a missing file or a wrong hash
    say("sources verified (sha256): " + ", ".join(f"{name}={h}" for name, h in hashes.items()))
    state_note(p, "sources_before", hashes)

    bench = measure_step_times(cfg, int(train["batch_size"]), bool(train["amp"]), steps=BENCH_STEPS, warmup=BENCH_WARMUP)
    proj = project_minutes(cfg, bench)
    say(f"benchmark (batch {train['batch_size']}, amp {train['amp']}, {BENCH_WARMUP} warm-up + {BENCH_STEPS} timed steps): "
        f"warm-up {bench['warmup_s_per_step']:.3f} s/step, joint {bench['joint_s_per_step']:.3f} s/step, "
        f"full validation {bench['val_full_s']:.1f} s, subset validation {bench['val_subset_s']:.1f} s, "
        f"peak memory {bench.get('peak_mb', float('nan')):.0f} MB")
    say(f"projected minutes of the REAL run (estimate): one trial {proj['trial_s']} s, study {proj['study_min']}, "
        f"final training {proj['final_min']}, prepare/evaluate/export/overhead "
        f"{proj['prepare_min']}+{proj['evaluate_min']}+{proj['export_min']}+{proj['overhead_min']}, "
        f"TOTAL {proj['total_min']} min (limit budget.max_minutes = {proj['limit_min']}; notebook setup not included)")
    state_note(p, "benchmark", dict(bench, batch=train["batch_size"], amp=train["amp"], projection=proj))
    if proj["total_min"] > proj["limit_min"]:
        if p.dry:
            say("dry run: the projection is above the limit; a real run would stop here (no gate stop in the dry run)")
        else:
            say(f"GATE: projected {proj['total_min']} min is above {proj['limit_min']} min. Nothing was trained. Cut the budget "
                f"with BUDGET_OVERRIDES in notebook Cell 2, in this order: {', '.join(CUT_LIST)} (plan section C), and run again")
            sys.exit(GATE_EXIT_CODE)


def stage_study(p, state):
    from genai.tasks.task1.config import resolve
    from genai.tasks.task3.tune import run_study
    cfg = study_cfg(p)
    out_dir = run_study(cfg)                                   # resumes; exports trials.csv + plots at the end
    shutil.copyfile(resolve(cfg["storage"]), Path(out_dir) / Path(cfg["storage"]).name)   # DB next to the exports
    say(f"study counts {study_counts(cfg)}")


def stage_final_train(p, state):
    """Write the final config from the best trial, then train, resuming from this run's own ckpt_last.pt."""
    from genai.tasks.task3.train import run_training
    from genai.tasks.task3.tune import write_final_config
    rd = run_dir(p, state)
    last = rd / "ckpt_last.pt"
    if last.exists() and p.final_yaml.exists():                # a restart in the middle of the training keeps its config
        say("resuming: the final config of the first attempt is kept")
    else:
        cfg = study_cfg(p)
        write_final_config(cfg, open_study(cfg), p.final_yaml)     # base config + best parameters + final epochs
        say(f"final config written to {p.final_yaml}")
    cfg = final_cfg(p)
    cfg["run_id"] = state["run_id"]                            # fixed by the supervisor: restarts reuse it
    if last.exists():
        resume = str(last)
    else:
        resume = None
        if rd.exists():                                        # crashed before the first checkpoint: start clean
            shutil.rmtree(rd)
    run_training(cfg, resume=resume)
    (rd / "DONE").write_text(now(), encoding="utf-8")


def stage_evaluate(p, state):
    from genai.tasks.task3.evaluate import run_evaluation
    out = run_evaluation(final_cfg(p), str(best_ckpt(p, state)), final_test=False)
    state_note(p, "eval_dir", str(out))
    summary = Path(out) / "summary.json"
    if summary.exists():
        state_note(p, "eval_summary", json.loads(summary.read_text(encoding="utf-8")))


def stage_export_promote(p, state):
    from genai.common.checkpoint import promote
    from genai.common.constants import ONNX_FILES
    from genai.export.task3_export import export_t3, verify_t3_parity
    cfg = final_cfg(p)                                         # on Kaggle the data is under /kaggle/input: the config knows
    ckpt = best_ckpt(p, state)
    onnx_path = p.onnx_dir / ONNX_FILES[ONNX_KEY]
    export_t3(ONNX_KEY, ckpt, onnx_path)
    meta = json.loads(Path(str(onnx_path) + ".meta.json").read_text(encoding="utf-8"))
    if meta["smoke"] and not p.dry:
        raise RuntimeError(f"{ONNX_KEY}: exported model is flagged smoke; refusing to treat it as final")
    r = verify_t3_parity(ONNX_KEY, ckpt, onnx_path, n=16, tag="dryrun" if p.dry else "final",
                         csv_path=p.parity_csv, data_root=cfg)
    print(ONNX_KEY, json.dumps(r, default=str))
    if not r["passed"]:
        raise RuntimeError(f"ONNX parity failed for {ONNX_KEY}: {json.dumps(r, default=str)}")
    state_note(p, "parity", json.loads(json.dumps(r, default=str)))
    state_note(p, "promoted", str(promote(ckpt, PROMOTED_NAME, root=p.models_root)))
    # the Task 2 checkpoints must be byte-identical to the ones the run started from
    after = check_sources(cfg)
    before = load_state(p).get("results", {}).get("sources_before")
    state_note(p, "sources_after", after)
    if before is not None and before != after:
        raise RuntimeError(f"source checkpoints changed during the run: before {before}, after {after}")
    say("sources verified again at the end: unchanged")


STAGE_FUNCS = {name: globals()[f"stage_{name}"] for name in STAGES}


# ------------------------------------------------------------------------------ progress measures
def progress_marker(p: Paths, state: dict, stage: str) -> int:
    """A number that grows when a stage makes progress (tells 'stuck' from 'slow but moving')."""
    try:
        if stage == "study":
            c = study_counts(study_cfg(p))
            return c.get("COMPLETE", 0) + c.get("PRUNED", 0)
        if stage == "final_train" and state.get("run_id"):
            metrics = run_dir(p, state) / "metrics.jsonl"
            return sum(1 for _ in open(metrics, encoding="utf-8")) if metrics.exists() else 0
    except Exception:
        pass                                                   # a marker must never stop the supervisor
    return 0


# ------------------------------------------------------------------------------ supervisor
def kill_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.call(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        os.kill(pid, 9)


def relay_status_lines(p: Paths, stage: str, log_path: Path, offset: int) -> int:
    """Copy the STATUS lines a child printed since `offset` into the status log. Returns the new offset."""
    with open(log_path, "rb") as f:
        f.seek(offset)
        data = f.read()
    whole = data[:data.rfind(b"\n") + 1]                       # only complete lines; a partial line is read next time
    for line in whole.decode("utf-8", errors="replace").splitlines():
        if line.startswith(STATUS_MARK):
            status(p, f"{stage}: {line[len(STATUS_MARK):].strip()}")
    return offset + len(whole)


def run_child(p: Paths, state: dict, stage: str) -> int:
    """Start one stage as a child process; kill it if its log stops moving. Returns the exit code."""
    tmp_dir = ROOT / "artifacts" / "tmp"                       # temp files next to the repository (D36)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", TMP=str(tmp_dir), TEMP=str(tmp_dir), **WANDB_ENV)
    if state.get("wandb_mode") == "offline":
        env["WANDB_MODE"] = "offline"
    if p.dry:
        env["TRACKER"] = "none"                                # the dry run adds no runs to the W&B project
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", stage] + p.cli_args()
    log_path = p.stage_log(stage)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n===== {now()} start {stage} (attempt {state['stages'][stage]['attempts']}) =====\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        offset, last_check = log_path.stat().st_size, time.time()
        while proc.poll() is None:
            time.sleep(POLL_SECONDS)                           # exit check every POLL_SECONDS ...
            if time.time() - last_check >= STALL_CHECK_SECONDS:     # ... stall check every STALL_CHECK_SECONDS
                last_check = time.time()
                offset = relay_status_lines(p, stage, log_path, offset)
                silent = time.time() - log_path.stat().st_mtime
                if silent > STALL_MINUTES * 60:
                    status(p, f"{stage}: log silent for {silent / 60:.0f} min -> killing the stuck process")
                    kill_tree(proc.pid)
                    proc.wait()
                    return -9
        relay_status_lines(p, stage, log_path, offset)
    return proc.returncode


def wandb_failure_in_log(p: Paths, stage: str) -> bool:
    tail = p.stage_log(stage).read_text(encoding="utf-8", errors="replace")[-20000:]
    return any(k in tail for k in ("wandb.errors", "CommError", "wandb: ERROR", "UsageError"))


def wandb_sync(p: Paths) -> None:
    offline_dirs = sorted((ROOT / "wandb").glob("offline-run-*"))
    if offline_dirs:
        status(p, f"wandb sync of {len(offline_dirs)} offline run(s)")
        subprocess.call([sys.executable, "-m", "wandb", "sync", "--include-offline", *map(str, offline_dirs)],
                        cwd=ROOT, env=dict(os.environ, **WANDB_ENV))


def write_summary(p: Paths, state: dict) -> None:
    """t3_summary.json, the facts for the report (plan D10): study counts and best trial, range-edge flags, timeout,
    stage seconds, benchmark, final curve, evaluation summary, parity, source hashes before and after."""
    from genai.tasks.task1.config import resolve
    results = state.get("results", {})
    cfg = study_cfg(p)
    study = open_study(cfg)
    counts = study_counts(cfg)
    answered = counts.get("COMPLETE", 0) + counts.get("PRUNED", 0)
    summary = {"written": now(), "pipeline_version": PIPELINE_VERSION, "dry_run": p.dry, "tag": p.tag,
               "budget_overrides": p.budget, "wandb_mode": state.get("wandb_mode"), "run_id": state.get("run_id"),
               "stage_seconds": {s: v.get("seconds") for s, v in state["stages"].items()},
               "stage_attempts": {s: v["attempts"] for s, v in state["stages"].items()}}
    summary["study"] = {"name": study.study_name, "counts": counts, "answered": answered,
                        "n_trials_budget": cfg["n_trials"], "timeout_minutes": cfg["timeout_minutes"],
                        "timeout_hit": answered < cfg["n_trials"],      # the study stopped before its trial budget: the timeout
                        "epochs_per_trial": cfg["epochs_per_trial"], "pruner": cfg["pruner"], "search_space": cfg["tuned_params"]}
    info_file = Path(resolve(cfg["study_dir"])) / "study_run.json"       # written by run_study: pruned reasons, timeout flag
    if info_file.exists():
        summary["study"]["run_info"] = json.loads(info_file.read_text(encoding="utf-8"))
        summary["study"]["timeout_hit"] = summary["study"]["run_info"].get("timeout_hit", summary["study"]["timeout_hit"])
    try:
        best = study.best_trial
        summary["study"].update(best_trial=best.number, best_value=best.value, best_params=best.params,
                                best_user_attrs=best.user_attrs,
                                range_edge_flags=range_edge_flags(cfg["tuned_params"], best.params))
    except ValueError:
        summary["study"]["best_trial"] = None                  # no COMPLETE trial
    summary["benchmark"] = results.get("benchmark")            # measured seconds per step + projected minutes
    projected = (results.get("benchmark") or {}).get("projection", {})
    summary["projected_vs_actual_min"] = {                     # to re-estimate the next run
        "study": [projected.get("study_min"), round((state["stages"]["study"].get("seconds") or 0) / 60, 1)],
        "final_train": [projected.get("final_min"), round((state["stages"]["final_train"].get("seconds") or 0) / 60, 1)]}
    metrics = run_dir(p, state) / "metrics.jsonl" if state.get("run_id") else None
    if metrics is not None and metrics.exists():
        rows = [json.loads(line) for line in open(metrics, encoding="utf-8")]
        keys = ("epoch", "stage", "train_loss", "val_J", "val_ssim", "val_gate_accuracy", "val_mean_w", "collapsed")
        summary["final_curve"] = [{k: r.get(k) for k in keys} for r in rows]
        eligible = [r for r in rows if r.get("stage") == "joint" and not r.get("collapsed") and "val_J" in r]
        if eligible:                                           # the epoch the best checkpoint comes from (plan B14)
            summary["final_best_epoch"] = min(eligible, key=lambda r: r["val_J"])["epoch"]
    summary["evaluation"] = {"dir": results.get("eval_dir"), "summary": results.get("eval_summary")}
    summary["parity"] = results.get("parity")
    summary["promoted"] = results.get("promoted")
    before, after = results.get("sources_before"), results.get("sources_after")
    summary["sources"] = {"before": before, "after": after, "unchanged": before == after}
    p.summary.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    status(p, f"summary written to {p.summary}")


def preflight(p: Paths) -> None:
    """Information only: GPU, W&B login. Nothing here stops the run."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        status(p, f"preflight: GPU {out}")
    except Exception as err:
        status(p, f"preflight: nvidia-smi failed ({err})")
    # also in the dry run: it is the one place where a missing or unticked Kaggle secret shows up early
    try:
        import wandb
        status(p, f"preflight: wandb logged in as entity {wandb.Api(timeout=30).viewer.entity}")
    except Exception as err:
        status(p, f"preflight: wandb check failed ({err}); runs will be retried and may go offline")


def supervise(p: Paths) -> int:
    from genai.tasks.task1.config import make_run_id
    state = load_state(p)
    status(p, f"supervisor start (code version {PIPELINE_VERSION}, dry_run={p.dry}); "
              f"stages done: {[s for s in STAGES if state['stages'][s]['done']]}")
    status(p, f"budget overrides (--set): {p.set_items if p.set_items else 'none'}")
    if p.expected_sources:
        status(p, f"WARNING: expected source hashes come from {p.expected_sources} (tests only), not from TASK2_SHA256")
    preflight(p)
    for stage in STAGES:
        info = state["stages"][stage]
        if info["done"]:
            continue
        if stage == "final_train" and not state.get("run_id"):
            state["run_id"] = make_run_id(p.profile, f"t3moe_final_{p.tag}", smoke=p.dry)
            save_state(p, state)
        status(p, f"{stage}: starting")
        t0, failures = time.time(), 0
        while True:
            before = progress_marker(p, state, stage)
            info["attempts"] += 1
            save_state(p, state)
            code = run_child(p, state, stage)
            state = load_state(p) | {"stages": state["stages"], "wandb_mode": state["wandb_mode"],
                                     "run_id": state.get("run_id")}
            info = state["stages"][stage]
            if code == 0:
                break
            if code == GATE_EXIT_CODE:                         # the budget gate of `prepare`: retrying cannot help
                state["stopped"] = f"{stage}: projected time above budget.max_minutes (see the GATE line above)"
                save_state(p, state)
                status(p, "STOPPED: " + state["stopped"])
                return 2
            after = progress_marker(p, state, stage)
            failures = 0 if after > before else failures + 1
            status(p, f"{stage}: attempt {info['attempts']} ended with code {code}; progress {before}->{after}; "
                      f"consecutive failures without progress: {failures}")
            tail = [ln for ln in p.stage_log(stage).read_text(encoding="utf-8", errors="replace").splitlines()
                    if ln.strip() and not ln.startswith("wandb:")][-6:]
            status(p, f"{stage}: last lines of its log:" + "".join("\n    " + ln for ln in tail))
            if failures >= 2 and state["wandb_mode"] == "online" and wandb_failure_in_log(p, stage):
                state["wandb_mode"], failures = "offline", 0
                status(p, "W&B keeps failing -> switching to WANDB_MODE=offline (synced at the end)")
            if failures >= MAX_CONSECUTIVE_FAILURES:
                state["stopped"] = f"{stage} failed {failures} times in a row without progress; see {p.stage_log(stage)}"
                save_state(p, state)
                status(p, "STOPPED: " + state["stopped"])
                return 1
            save_state(p, state)
            time.sleep(RETRY_PAUSE_SECONDS)
        info["done"], info["seconds"] = True, round(time.time() - t0)
        save_state(p, state)
        status(p, f"{stage}: done in {info['seconds'] / 60:.1f} min")
        if stage == "study":
            c = study_counts(study_cfg(p))
            status(p, f"study counts: {c}")
            if c.get("COMPLETE", 0) * 2 < c.get("COMPLETE", 0) + c.get("PRUNED", 0):
                status(p, "FLAG: fewer than half of the answered trials completed (pruning-heavy study); report this")
    if state["wandb_mode"] == "offline":
        wandb_sync(p)
    write_summary(p, state)
    status(p, "ALL STAGES DONE")
    state["finished"] = now()
    save_state(p, state)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="tiny end-to-end check in artifacts/dryrun_t3/ (starts clean: the folder is emptied first)")
    ap.add_argument("--child", choices=STAGES, help=argparse.SUPPRESS)       # internal: run one stage
    ap.add_argument("--tag", default="t3", help="names the log/state folder artifacts/logs/<tag>/")
    ap.add_argument("--device-profile", default="local", choices=["local", "kaggle", "colab"])
    ap.add_argument("--num-workers", type=int, default=None, help="data workers (default: 0 locally, profile value elsewhere)")
    ap.add_argument("--data-root", default=None,
                    help="override the profile's data_root (e.g. a fake nested input/ folder, to rehearse Kaggle paths)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="budget override, repeatable, e.g. --set timeout_minutes=10 --set train.joint_epochs=8")
    ap.add_argument("--expected-sources", default=None, help=argparse.SUPPRESS)   # tests only: JSON with fixture sha256 values
    args = ap.parse_args()
    try:
        p = Paths(args.dry_run, args.tag, args.device_profile, args.num_workers, args.data_root, args.set,
                  args.expected_sources)
    except ValueError as err:
        ap.error(str(err))
    if args.child:
        STAGE_FUNCS[args.child](p, load_state(p))
        return
    if p.dry:
        shutil.rmtree(p.tmp, ignore_errors=True)               # a dry run always starts clean (no stale state from an earlier one)
    sys.exit(supervise(p))


if __name__ == "__main__":
    main()
