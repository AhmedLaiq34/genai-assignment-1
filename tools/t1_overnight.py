"""Unattended Task 1 pipeline: Optuna study -> final config -> final training -> val evaluation
-> ONNX export + parity -> promote. Written for an overnight run on the local RTX 3050.

Usage (from the repository root, with the venv's python):
    python tools/t1_overnight.py               # the real pipeline (resumes wherever it stopped)
    python tools/t1_overnight.py --dry-run     # tiny end-to-end check in artifacts/dryrun_overnight/
    python tools/t1_overnight.py --device-profile kaggle     # on Kaggle (notebooks/kaggle_train.ipynb does this)

Options: --tag (names the log folder artifacts/logs/<tag>/ and keeps the state of different studies apart),
--study-config, --final-config, --device-profile, --num-workers (default: 0 locally, the profile's value elsewhere).

Why a supervisor?  Every stage runs in a CHILD process. The supervisor watches the child's log;
if the child crashes or its log stops moving for STALL_MINUTES it is killed and started again.
Every stage is resumable, so a restart loses at most one trial or a few minutes of training:
  * study:  Optuna SQLite DB with load_if_exists (plus the fixes of decision D19 in task1/tune.py)
  * train:  ckpt_last.pt is written every epoch; the restart passes that exact file as `resume`
  * other stages are short and are simply re-run.
A stage that fails MAX_CONSECUTIVE_FAILURES times in a row WITHOUT any progress stops the pipeline
(state and reason in artifacts/logs/t1_overnight_state.json and t1_overnight_status.log).

Robustness choices (decision D19 in docs/DECISIONS.md):
  * num_workers = 0 for training data: no DataLoader worker processes, so the Windows worker crash
    seen on 2026-10-04 (WinError 1114 under memory pressure) cannot happen, and RAM use is ~1.5 GB.
  * W&B online (TRACKER=wandb). If wandb itself keeps failing, the supervisor switches to
    WANDB_MODE=offline (still W&B, files under wandb/) and runs `wandb sync` at the end.
  * The PC is asked not to go to sleep while the pipeline runs (SetThreadExecutionState).

Never reads the test set: evaluation is on the VAL manifest only (final_test=False everywhere).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

STUDY_CONFIG = "configs/task1_universal.yaml"
FINAL_CONFIG = "configs/task1_final_v2.yaml"     # configs/task1_final.yaml belongs to the first (dense) study
FINAL_EPOCHS = 100                       # approved budget (D17)
STAGES = ["study", "final_config", "train", "evaluate", "export", "promote"]
STALL_MINUTES = 20                       # one epoch + validation takes < 1 min; 20 min of silence = hung
MAX_CONSECUTIVE_FAILURES = 4             # without progress
RETRY_PAUSE_SECONDS = 60                 # let memory settle before a restart
WANDB_ENV = {"TRACKER": "wandb", "WANDB_ENTITY": "ahmedlaiq34", "WANDB_PROJECT": "genai-a1",
             "WANDB_INIT_TIMEOUT": "300", "WANDB_SILENT": "false"}


# ------------------------------------------------------------------------------ small helpers
def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Paths:
    """Every file the pipeline writes. The dry run redirects all of them into a tmp folder."""

    def __init__(self, dry: bool, tag: str = "v2", profile: str = "local", num_workers=None,
                 study_config: str = STUDY_CONFIG, final_config: str = FINAL_CONFIG):
        self.dry, self.tag, self.profile = dry, tag, profile
        self.study_config, self.final_config_name = study_config, final_config
        # 0 data workers locally (Windows worker crash, D19); elsewhere the profile's own value
        self.num_workers = num_workers if num_workers is not None else (0 if profile == "local" else None)
        self.logs = ROOT / "artifacts" / ("dryrun_overnight/logs" if dry else f"logs/{tag}")
        self.state = self.logs / "t1_overnight_state.json"
        self.status = self.logs / "t1_overnight_status.log"
        self.summary = self.logs / "t1_overnight_summary.json"
        self.tmp = ROOT / "artifacts" / "dryrun_overnight"
        self.final_config = (self.tmp / "task1_final.yaml") if dry else (ROOT / final_config)
        self.onnx = (self.tmp / "onnx" / "t1_universal_ae.onnx") if dry else (ROOT / "models/onnx/t1_universal_ae.onnx")
        self.parity_csv = (self.tmp / "onnx_parity.csv") if dry else (ROOT / "report/tables/onnx_parity.csv")
        self.models_root = (self.tmp / "models") if dry else None      # None = the real models/ folder

    def stage_log(self, stage: str) -> Path:
        return self.logs / f"t1_{stage}.log"

    def overrides(self) -> dict:
        return {} if self.num_workers is None else {"num_workers": self.num_workers}

    def cli_args(self) -> list:
        """Arguments a child process needs to rebuild the same Paths."""
        args = ["--tag", self.tag, "--device-profile", self.profile, "--study-config", self.study_config,
                "--final-config", self.final_config_name]
        if self.num_workers is not None:
            args += ["--num-workers", str(self.num_workers)]
        return args + (["--dry-run"] if self.dry else [])


def load_state(p: Paths) -> dict:
    if p.state.exists():
        return json.loads(p.state.read_text(encoding="utf-8"))
    return {"stages": {s: {"done": False, "attempts": 0} for s in STAGES}, "wandb_mode": "online"}


def save_state(p: Paths, state: dict) -> None:
    p.logs.mkdir(parents=True, exist_ok=True)
    tmp = p.state.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, p.state)


def status(p: Paths, msg: str) -> None:
    """One line to the status log (and the console): the Sonnet session reads this file."""
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    p.logs.mkdir(parents=True, exist_ok=True)
    with open(p.status, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def keep_awake() -> None:
    """Ask Windows not to sleep while this process runs (lid close may still sleep: see handoff doc)."""
    if os.name == "nt":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def kill_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.call(["taskkill", "/PID", str(pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        os.kill(pid, 9)


# ------------------------------------------------------------------------------ configs
def study_cfg(p: Paths) -> dict:
    from genai.tasks.task1.config import load_config
    from genai.tasks.task1.tune import dry_run_overrides
    cfg = load_config(p.study_config, p.profile, p.overrides())
    if p.dry:
        cfg = dry_run_overrides(cfg, p.tmp)
        cfg.update(n_trials=3, epochs_per_trial=2, trial_train_subset=256)
        cfg["pruner"] = {"name": "Median", "n_startup_trials": 1, "n_warmup_steps": 1}
        cfg["train"]["val_subset"] = 64
    return cfg


def final_cfg(p: Paths) -> dict:
    from genai.tasks.task1.config import load_config
    cfg = load_config(p.final_config, p.profile, p.overrides())
    if p.dry:
        cfg["output_root"] = cfg["persist_root"] = str(p.tmp)
        cfg["train"]["train_subset"], cfg["train"]["val_subset"] = 256, 64
    return cfg


def open_study(cfg: dict):
    import optuna
    from genai.tasks.task1.config import resolve
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.load_study(study_name=cfg["study"], storage=f"sqlite:///{resolve(cfg['storage']).as_posix()}")


def study_counts(cfg: dict) -> dict:
    """Trial counts by state ({} if the study does not exist yet)."""
    try:
        study = open_study(cfg)
    except Exception:
        return {}
    counts = {}
    for t in study.trials:
        counts[t.state.name] = counts.get(t.state.name, 0) + 1
    return counts


# ------------------------------------------------------------------------------ the stages (child process)
def stage_study(p: Paths, state: dict) -> None:
    from genai.tasks.task1.config import resolve
    from genai.tasks.task1.tune import run_study
    cfg = study_cfg(p)
    out_dir = run_study(cfg)                                   # resumes; exports trials.csv + plots at the end
    shutil.copyfile(resolve(cfg["storage"]), Path(out_dir) / Path(cfg["storage"]).name)   # DB next to the exports


def stage_final_config(p: Paths, state: dict) -> None:
    """configs/task1_final.yaml = the study config + the best trial's parameters + the 100-epoch schedule."""
    import yaml
    from genai.tasks.task1.config import set_dotted
    from genai.tasks.task1.tune import PARAM_TARGETS
    study = open_study(study_cfg(p))
    best = study.best_trial
    raw = yaml.safe_load((ROOT / p.study_config).read_text(encoding="utf-8"))
    for name, value in best.params.items():
        set_dotted(raw, PARAM_TARGETS[name], value)
    raw["train"]["epochs"] = 2 if p.dry else FINAL_EPOCHS
    raw["train"]["sample_every_epochs"] = 5
    raw["run"]["desc"] = "t1_final"
    raw["search_space_status"] = "USED BY STUDY t1_universal (this file holds its best trial)"
    header = (f"# Task 1 FINAL training config, written by tools/t1_overnight.py on {now()}.\n"
              f"# Best trial of study '{study.study_name}': trial {best.number}, val J {best.value:.5f}.\n"
              f"# Parameters copied from that trial: {json.dumps(best.params)}\n"
              f"# Schedule: {raw['train']['epochs']} epochs (approved budget D17). Selection by best val J.\n")
    p.final_config.parent.mkdir(parents=True, exist_ok=True)
    p.final_config.write_text(header + yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def stage_train(p: Paths, state: dict) -> None:
    from genai.tasks.task1.config import runs_dir
    from genai.tasks.task1.train import run_training
    cfg = final_cfg(p)
    cfg["run_id"] = state["final_run_id"]                      # fixed by the supervisor: restarts reuse it
    run_dir = runs_dir(cfg) / cfg["run_id"]
    last = run_dir / "ckpt_last.pt"
    if last.exists():
        resume = str(last)                                     # explicit path: never picks up a trial run
    else:
        resume = None
        if (run_dir / "metrics.jsonl").exists():               # crashed before the first checkpoint
            (run_dir / "metrics.jsonl").unlink()
    run_training(cfg, resume=resume)
    (run_dir / "DONE").write_text(now(), encoding="utf-8")


def _final_run_dir(p: Paths, state: dict) -> Path:
    from genai.tasks.task1.config import runs_dir
    return runs_dir(final_cfg(p)) / state["final_run_id"]


def stage_evaluate(p: Paths, state: dict) -> None:
    from genai.tasks.task1.evaluate import run_evaluation
    out = run_evaluation(final_cfg(p), str(_final_run_dir(p, state) / "ckpt_best.pt"), final_test=False)
    state_note(p, "eval_dir", str(out))


def stage_export(p: Paths, state: dict) -> None:
    from genai.export.onnx_export import export_model
    from genai.export.onnx_verify import verify_parity
    ckpt = _final_run_dir(p, state) / "ckpt_best.pt"
    export_model("t1_universal", ckpt, p.onnx)
    meta = json.loads(Path(str(p.onnx) + ".meta.json").read_text(encoding="utf-8"))
    if meta["smoke"] and not p.dry:
        raise RuntimeError("exported model is flagged smoke; refusing to treat it as final")
    # data_root = this run's config (on Kaggle the data is under /kaggle/input, not in the repository)
    result = verify_parity("t1_universal", ckpt, p.onnx, n=16, tag="dryrun" if p.dry else "final",
                           csv_path=p.parity_csv, data_root=final_cfg(p))
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError(f"ONNX parity failed: max abs diff {result['max_abs_diff']}")
    state_note(p, "parity", result)


def stage_promote(p: Paths, state: dict) -> None:
    from genai.common.checkpoint import promote
    dest = promote(_final_run_dir(p, state) / "ckpt_best.pt", "t1_universal_ae", root=p.models_root)
    state_note(p, "promoted", str(dest))


def state_note(p: Paths, key: str, value) -> None:
    """Child processes record results in the state file (the supervisor re-reads it)."""
    state = load_state(p)
    state.setdefault("results", {})[key] = value
    save_state(p, state)


STAGE_FUNCS = {"study": stage_study, "final_config": stage_final_config, "train": stage_train,
               "evaluate": stage_evaluate, "export": stage_export, "promote": stage_promote}


# ------------------------------------------------------------------------------ progress measures
def progress_marker(p: Paths, state: dict, stage: str):
    """A number that grows when a stage makes progress (used to tell 'stuck' from 'slow but moving')."""
    if stage == "study":
        c = study_counts(study_cfg(p))
        return c.get("COMPLETE", 0) + c.get("PRUNED", 0)
    if stage == "train" and state.get("final_run_id"):
        metrics = _final_run_dir(p, state) / "metrics.jsonl"
        return sum(1 for _ in open(metrics, encoding="utf-8")) if metrics.exists() else 0
    return 0


# ------------------------------------------------------------------------------ supervisor
def run_child(p: Paths, state: dict, stage: str) -> int:
    """Start one stage as a child process; kill it if its log stops moving. Returns the exit code."""
    tmp_dir = ROOT / "artifacts" / "tmp"                 # temp files on D:, not the nearly full C: (D36)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", TMP=str(tmp_dir), TEMP=str(tmp_dir), **WANDB_ENV)
    if state.get("wandb_mode") == "offline":
        env["WANDB_MODE"] = "offline"
    if p.dry:
        env["TRACKER"] = "none"                  # the dry run must not add junk runs to the W&B project
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", stage] + p.cli_args()
    log_path = p.stage_log(stage)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n===== {now()} start {stage} (attempt {state['stages'][stage]['attempts']}) =====\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(30)
            silent = time.time() - log_path.stat().st_mtime
            if silent > STALL_MINUTES * 60:
                status(p, f"{stage}: log silent for {silent / 60:.0f} min -> killing the stuck process")
                kill_tree(proc.pid)
                proc.wait()
                return -9
    return proc.returncode


def wandb_failure_in_log(p: Paths, stage: str) -> bool:
    tail = p.stage_log(stage).read_text(encoding="utf-8", errors="replace")[-20000:]
    return any(k in tail for k in ("wandb.errors", "CommError", "wandb: ERROR", "UsageError"))


def wandb_sync(p: Paths) -> None:
    """Upload runs that were logged offline (only needed if the supervisor had to go offline)."""
    offline_dirs = sorted((ROOT / "wandb").glob("offline-run-*"))
    if offline_dirs:
        status(p, f"wandb sync of {len(offline_dirs)} offline run(s)")
        subprocess.call([sys.executable, "-m", "wandb", "sync", "--include-offline", *map(str, offline_dirs)],
                        cwd=ROOT, env=dict(os.environ, **WANDB_ENV))


def write_summary(p: Paths, state: dict) -> None:
    """Facts for the report (docs/PHASE_T1_TRAINING_REPORT.md is written from this file + the logs)."""
    import re
    cfg = study_cfg(p)
    summary = {"written": now(), "dry_run": p.dry, "wandb_mode": state.get("wandb_mode"),
               "stage_attempts": {s: v["attempts"] for s, v in state["stages"].items()},
               "stage_seconds": {s: v.get("seconds") for s, v in state["stages"].items()},
               "results": state.get("results", {}), "final_run_id": state.get("final_run_id")}
    study = open_study(cfg)
    summary["study"] = {"name": study.study_name, "counts": study_counts(cfg),
                        "best_trial": study.best_trial.number, "best_J": study.best_value,
                        "best_params": study.best_params, "search_space": cfg["tuned_params"],
                        "pruner": cfg["pruner"], "epochs_per_trial": cfg["epochs_per_trial"],
                        "n_trials_budget": cfg["n_trials"]}
    metrics = _final_run_dir(p, state) / "metrics.jsonl"
    if metrics.exists():
        rows = [json.loads(line) for line in open(metrics, encoding="utf-8")]
        best = min(rows, key=lambda r: r["val_J"])
        summary["final_training"] = {"epochs_logged": len(rows), "last": rows[-1], "best_epoch": best["epoch"],
                                     "best_val_J": best["val_J"], "first": rows[0]}
    urls = set()
    for log in p.logs.glob("t1_*.log"):
        urls.update(re.findall(r"https://wandb\.ai/\S+/runs/\w+", log.read_text(encoding="utf-8", errors="replace")))
    summary["wandb_run_urls"] = sorted(urls)
    p.summary.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    status(p, f"summary written to {p.summary}")


def preflight(p: Paths) -> None:
    """Information only: GPU, free memory, W&B login. Nothing here stops the run."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        status(p, f"preflight: GPU util/mem {out}")
    except Exception as err:
        status(p, f"preflight: nvidia-smi failed ({err})")
    try:
        import wandb
        status(p, f"preflight: wandb logged in as entity {wandb.Api().viewer.entity}")
    except Exception as err:
        status(p, f"preflight: wandb check failed ({err}); runs will be retried and may go offline")


def supervise(p: Paths) -> int:
    keep_awake()
    state = load_state(p)
    status(p, f"supervisor start (dry_run={p.dry}); stages done: "
              f"{[s for s in STAGES if state['stages'][s]['done']]}")
    preflight(p)
    for stage in STAGES:
        info = state["stages"][stage]
        if info["done"]:
            continue
        if stage == "train" and not state.get("final_run_id"):
            from genai.tasks.task1.config import make_run_id
            state["final_run_id"] = make_run_id(p.profile, f"t1_final_{p.tag}")
            save_state(p, state)
        status(p, f"{stage}: starting")
        t0, failures = time.time(), 0
        while True:
            before = progress_marker(p, state, stage)
            info["attempts"] += 1
            save_state(p, state)
            code = run_child(p, state, stage)
            state = load_state(p) | {"stages": state["stages"], "wandb_mode": state["wandb_mode"],
                                     "final_run_id": state.get("final_run_id")}
            info = state["stages"][stage]
            if code == 0:
                break
            after = progress_marker(p, state, stage)
            failures = 0 if after > before else failures + 1
            status(p, f"{stage}: attempt {info['attempts']} ended with code {code}; progress {before}->{after}; "
                      f"consecutive failures without progress: {failures}")
            tail = [ln for ln in p.stage_log(stage).read_text(encoding="utf-8", errors="replace").splitlines()
                    if ln.strip() and not ln.startswith("wandb:")][-6:]
            # the error text shows up in the notebook output, so a failure needs no second round trip
            status(p, f"{stage}: last lines of its log:" + "".join("\n    " + ln for ln in tail))
            if failures >= 2 and state["wandb_mode"] == "online" and wandb_failure_in_log(p, stage):
                state["wandb_mode"] = "offline"
                failures = 0
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
            answered = c.get("COMPLETE", 0) + c.get("PRUNED", 0)
            status(p, f"study counts: {c}")
            if c.get("COMPLETE", 0) * 2 < answered:
                status(p, "FLAG: fewer than half of the answered trials completed (pruning-heavy study); "
                          "continuing with the best COMPLETE trial; report this")
    if state["wandb_mode"] == "offline":
        wandb_sync(p)
    write_summary(p, state)
    status(p, "ALL STAGES DONE")
    state["finished"] = now()
    save_state(p, state)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="tiny end-to-end check in artifacts/dryrun_overnight/")
    ap.add_argument("--child", choices=STAGES, help=argparse.SUPPRESS)       # internal: run one stage
    ap.add_argument("--tag", default="v2", help="names the log/state folder artifacts/logs/<tag>/")
    ap.add_argument("--device-profile", default="local", choices=["local", "kaggle", "colab"])
    ap.add_argument("--num-workers", type=int, default=None, help="data workers (default: 0 locally, profile value elsewhere)")
    ap.add_argument("--study-config", default=STUDY_CONFIG)
    ap.add_argument("--final-config", default=FINAL_CONFIG)
    args = ap.parse_args()
    p = Paths(args.dry_run, args.tag, args.device_profile, args.num_workers, args.study_config, args.final_config)
    if args.child:
        STAGE_FUNCS[args.child](p, load_state(p))
        return
    sys.exit(supervise(p))


if __name__ == "__main__":
    main()
