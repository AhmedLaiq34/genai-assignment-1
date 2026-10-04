"""Unattended Task 4 pipeline (Kaggle or local), modelled on tools/t2_pipeline.py. Plan: docs/TASK4_PLAN.md.

Stages, in order (every one resumable; finished stages are skipped on a re-run):
    data_check     the FS2K split and caches are found (prints pair and style counts)
    benchmark      scripts/benchmark.py --model t4 on THIS device (rows appended to docs/BENCHMARKS.md)
    train          final cGAN training from configs/task4_final.yaml (resumes from ckpt_last.pt)
    split_ckpt     ckpt_best.pt -> t4_generator.pt + t4_discriminator.pt (not promoted: the student promotes locally)
    export         t4_generator.onnx + ONNX parity over 16 val photos x 3 style ids
    evaluate       validation evaluation (the test set is never opened here)

The confirmatory Optuna study (plan B.3) is NOT part of this pipeline (student decision: skipped).
`train.epochs` in configs/task4_final.yaml must be set by the student from the benchmark BEFORE the real run
(it is TBD_AFTER_BENCHMARK; the pipeline refuses to start the real training while it is TBD).

Usage (repository root):
    python tools/t4_pipeline.py --device-profile kaggle        # notebooks/kaggle_t4_pipeline.ipynb does this
    python tools/t4_pipeline.py --dry-run                      # tiny end-to-end check in artifacts/dryrun_t4/
    python tools/t4_pipeline.py --dry-run --data-root <folder> # rehearse a Kaggle-like nested data folder locally

Why a supervisor: every stage runs in a CHILD process; if it crashes or its log is silent for STALL_MINUTES it is
killed and started again, and it resumes (ckpt_last.pt every epoch / every checkpoint_every_minutes).
A stage that fails MAX_CONSECUTIVE_FAILURES times in a row without progress stops the pipeline.
The supervisor code repeats tools/t2_pipeline.py (a Task 2 file that is not edited); candidate refactor.

Never reads the test set: evaluation is on the VAL split only (final_test=False everywhere).
"""
from __future__ import annotations

import argparse
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

PIPELINE_VERSION = "t4-v2"   # printed by the notebook (Cell 1) and the supervisor: proves which code Kaggle runs

FINAL_CONFIG = "configs/task4_final.yaml"
STAGES = ["data_check", "benchmark", "train", "split_ckpt", "export", "evaluate"]
STALL_MINUTES = 30                       # one epoch + validation takes well under this
MAX_CONSECUTIVE_FAILURES = 4             # without progress
RETRY_PAUSE_SECONDS = 60
WANDB_ENV = {"TRACKER": "wandb", "WANDB_ENTITY": "ahmedlaiq34", "WANDB_PROJECT": "genai-a1",
             "WANDB_INIT_TIMEOUT": "300", "WANDB_SILENT": "false"}
BENCH_BATCHES = ["8"]                     # only the final config's batch size (keeps the stage to about 1 minute)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------------------------ paths and state
class Paths:
    """Every file the pipeline writes. The dry run redirects all of them into artifacts/dryrun_t4/."""

    def __init__(self, dry: bool, tag: str = "t4", profile: str = "local", num_workers=None, data_root=None):
        self.dry, self.tag, self.profile = dry, tag, profile
        self.data_root = data_root       # optional override of the profile's data_root (tests a Kaggle-like layout locally)
        # 0 data workers locally (Windows worker crash); elsewhere the profile's own value
        self.num_workers = num_workers if num_workers is not None else (0 if profile == "local" else None)
        self.tmp = ROOT / "artifacts" / "dryrun_t4"
        self.logs = (self.tmp / "logs") if dry else (ROOT / "artifacts" / "logs" / tag)
        self.state = self.logs / "t4_state.json"
        self.status = self.logs / "t4_status.log"
        self.summary = self.logs / "t4_summary.json"
        self.onnx_dir = (self.tmp / "onnx") if dry else (ROOT / "models" / "onnx")
        self.promote_dir = (self.tmp / "promote") if dry else (ROOT / "artifacts" / "promote_t4")
        self.parity_csv = (self.tmp / "onnx_parity.csv") if dry else (ROOT / "report" / "tables" / "onnx_parity.csv")

    def stage_log(self, stage: str) -> Path:
        return self.logs / f"t4_{stage}.log"

    def overrides(self) -> dict:
        out = {} if self.num_workers is None else {"num_workers": self.num_workers}
        if self.data_root:
            out["data_root"] = self.data_root
        return out

    def cli_args(self) -> list:
        args = ["--tag", self.tag, "--device-profile", self.profile]
        if self.num_workers is not None:
            args += ["--num-workers", str(self.num_workers)]
        if self.data_root:
            args += ["--data-root", self.data_root]
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


# ------------------------------------------------------------------------------ config
def final_cfg(p: Paths) -> dict:
    """configs/task4_final.yaml + device profile. The dry run shrinks it to a few minutes."""
    from genai.tasks.task4.config import load_config
    cfg = load_config(FINAL_CONFIG, p.profile, p.overrides())
    if p.dry:
        cfg["output_root"] = cfg["persist_root"] = str(p.tmp)
        cfg["train"].update(epochs=2, train_subset=64, val_subset=32, sample_every_epochs=1,
                            snapshot_every_epochs=0)
        cfg["run"]["smoke"] = True
    elif p.profile == "kaggle":
        # a new Kaggle version starts with an empty /kaggle/working; the previous version's output attached
        # as an input holds the checkpoint (plan C19, resume_roots)
        cfg["resume_roots"] = ["/kaggle/input"]
    return cfg


def run_dir(p: Paths, state: dict) -> Path:
    from genai.tasks.task4.config import runs_dir
    return runs_dir(final_cfg(p)) / state["run_id"]


# ------------------------------------------------------------------------------ the stages (child process)
def stage_data_check(p, state):
    from genai.fs2k.dataset import resolve_fs2k_paths
    from genai.fs2k.split import load_split
    cfg = final_cfg(p)
    paths = resolve_fs2k_paths(cfg["data_root"])   # /kaggle/input on Kaggle, data/ locally
    split = load_split(paths["split_file"])             # checks the sha256 of the id lists
    counts = {k: len(split[k]) for k in ("train", "val", "test")}
    print("FS2K paths:", {k: str(v) for k, v in paths.items()})
    print("pairs per split:", counts, "style counts:", split.get("style_counts"))
    if counts["train"] != 899 or counts["val"] != 159:
        raise RuntimeError(f"unexpected split sizes {counts} (expected 899 train / 159 val)")
    state_note(p, "data_check", {"counts": counts, "style_counts": split.get("style_counts")})


def stage_benchmark(p, state):
    """Run scripts/benchmark.py on this device; the rows go to docs/BENCHMARKS.md (in the results zip)."""
    from genai.tasks.task4.config import find_latest_checkpoint
    if p.dry or not os.environ.get("T4_BENCHMARK"):
        print("benchmark skipped (set the environment variable T4_BENCHMARK=1 to run it)")
        return
    if find_latest_checkpoint(final_cfg(p)):
        print("a checkpoint of an earlier session exists (this is a continuation): benchmark skipped")
        return
    cmd = [sys.executable, str(ROOT / "scripts" / "benchmark.py"), "--model", "t4", "--batch", *BENCH_BATCHES,
           "--amp", "both", "--base-channels", "64", "--device-profile", p.profile]
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT)


def stage_train(p, state):
    """Final training, resuming from its own ckpt_last.pt (never from a trial run)."""
    from genai.tasks.task1.config import is_tbd
    from genai.tasks.task4.config import find_latest_checkpoint
    from genai.tasks.task4.train import run_training
    cfg = final_cfg(p)
    if is_tbd(cfg["train"]["epochs"]):
        raise RuntimeError("train.epochs in configs/task4_final.yaml is still TBD_AFTER_BENCHMARK: set it from the "
                           "benchmark (docs/TASK4_PLAN.md C17), rebuild the code zip and run again")
    cfg["run_id"] = state["run_id"]                    # fixed by the supervisor: restarts reuse it
    rd = run_dir(p, state)
    last = rd / "ckpt_last.pt"
    if not last.exists() and not p.dry:
        # a reset or a new Kaggle version: continue the newest earlier run by copying its files into this run folder
        previous = find_latest_checkpoint(cfg)
        if previous is not None and previous.parent.resolve() != rd.resolve():
            print("continuing the earlier session's run:", previous.parent)
            shutil.copytree(previous.parent, rd, dirs_exist_ok=True)
    if last.exists():
        resume = str(last)
    else:
        resume = None
        if rd.exists():                                # crashed before the first checkpoint: start clean
            shutil.rmtree(rd)
    run_training(cfg, resume=resume)
    (rd / "DONE").write_text(now(), encoding="utf-8")


def stage_split_ckpt(p, state):
    from genai.tasks.task4.train import split_checkpoint
    g_path, d_path = split_checkpoint(run_dir(p, state) / "ckpt_best.pt", p.promote_dir)
    print("generator:", g_path, "\ndiscriminator:", d_path)
    state_note(p, "split_ckpt", {"generator": str(g_path), "discriminator": str(d_path)})


def stage_export(p, state):
    from genai.common.constants import ONNX_FILES
    from genai.export.task4_export import export_t4_generator, verify_t4_parity
    cfg = final_cfg(p)
    g_path = p.promote_dir / "t4_generator.pt"
    onnx_path = p.onnx_dir / ONNX_FILES["t4_generator"]
    export_t4_generator("t4_generator", g_path, onnx_path)
    meta = json.loads(Path(str(onnx_path) + ".meta.json").read_text(encoding="utf-8"))
    if meta["smoke"] and not p.dry:
        raise RuntimeError("exported model is flagged smoke; refusing to treat it as final")
    # on Kaggle the data is under /kaggle/input: the config (with its data_root) tells the verifier where
    r = verify_t4_parity("t4_generator", g_path, onnx_path, n=16, tag="dryrun" if p.dry else "final",
                         csv_path=p.parity_csv, data_root=cfg["data_root"])
    print(json.dumps(r))
    if not r["passed"]:
        raise RuntimeError(f"ONNX parity failed: max abs diff {r['max_abs_diff']}")
    state_note(p, "parity", {k: r[k] for k in ("max_abs_diff", "mean_abs_diff", "passed")})


def stage_evaluate(p, state):
    from genai.tasks.task4.evaluate import run_evaluation
    cfg = final_cfg(p)
    out = run_evaluation(cfg, str(p.promote_dir / "t4_generator.pt"), final_test=False)
    state_note(p, "eval_dir", str(out))
    summary = Path(out) / "summary.json"
    if summary.exists():
        state_note(p, "eval_summary", json.loads(summary.read_text(encoding="utf-8")))


STAGE_FUNCS = {name: globals()[f"stage_{name}"] for name in STAGES}


# ------------------------------------------------------------------------------ progress measures
def progress_marker(p: Paths, state: dict, stage: str) -> int:
    """A number that grows when a stage makes progress (tells 'stuck' from 'slow but moving')."""
    if stage == "train" and state.get("run_id"):
        metrics = run_dir(p, state) / "metrics.jsonl"
        return sum(1 for _ in open(metrics, encoding="utf-8")) if metrics.exists() else 0
    return 0


# ------------------------------------------------------------------------------ supervisor
def kill_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.call(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        os.kill(pid, 9)


def run_child(p: Paths, state: dict, stage: str) -> int:
    """Start one stage as a child process; kill it if its log stops moving. Returns the exit code."""
    tmp_dir = ROOT / "artifacts" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", TMP=str(tmp_dir), TEMP=str(tmp_dir), **WANDB_ENV)
    if state.get("wandb_mode") == "offline":
        env["WANDB_MODE"] = "offline"
    if p.dry:
        env["TRACKER"] = "none"                        # the dry run adds no runs to the W&B project
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", stage] + p.cli_args()
    log_path = p.stage_log(stage)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n===== {now()} start {stage} (attempt {state['stages'][stage]['attempts']}) =====\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(10)
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
    offline_dirs = sorted((ROOT / "wandb").glob("offline-run-*"))
    if offline_dirs:
        status(p, f"wandb sync of {len(offline_dirs)} offline run(s)")
        subprocess.call([sys.executable, "-m", "wandb", "sync", "--include-offline", *map(str, offline_dirs)],
                        cwd=ROOT, env=dict(os.environ, **WANDB_ENV))


def write_summary(p: Paths, state: dict) -> None:
    """Facts for the report: stage times, training curve summary, evaluation, parity."""
    summary = {"written": now(), "dry_run": p.dry, "wandb_mode": state.get("wandb_mode"),
               "stage_seconds": {s: v.get("seconds") for s, v in state["stages"].items()},
               "stage_attempts": {s: v["attempts"] for s, v in state["stages"].items()},
               "run_id": state.get("run_id"), "results": state.get("results", {})}
    metrics = run_dir(p, state) / "metrics.jsonl"
    if metrics.exists():
        rows = [json.loads(line) for line in open(metrics, encoding="utf-8")]
        best = min(rows, key=lambda r: r["val/l1"])
        summary["final_training"] = {"epochs_logged": len(rows), "best_epoch": best["epoch"],
                                     "best_val_l1": best["val/l1"], "last": rows[-1]}
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
    try:
        import wandb
        status(p, f"preflight: wandb logged in as entity {wandb.Api().viewer.entity}")
    except Exception as err:
        status(p, f"preflight: wandb check failed ({err}); runs will be retried and may go offline")


def supervise(p: Paths) -> int:
    from genai.tasks.task1.config import make_run_id
    state = load_state(p)
    status(p, f"supervisor start (code version {PIPELINE_VERSION}, dry_run={p.dry}); "
              f"stages done: {[s for s in STAGES if state['stages'][s]['done']]}")
    preflight(p)
    if not state.get("run_id"):
        state["run_id"] = make_run_id(p.profile, f"t4_final_{p.tag}", smoke=p.dry)
        save_state(p, state)
    for stage in STAGES:
        info = state["stages"][stage]
        if info["done"]:
            continue
        status(p, f"{stage}: starting")
        t0, failures = time.time(), 0
        while True:
            before = progress_marker(p, state, stage)
            info["attempts"] += 1
            save_state(p, state)
            code = run_child(p, state, stage)
            state = load_state(p) | {"stages": state["stages"], "wandb_mode": state["wandb_mode"],
                                     "run_id": state["run_id"]}
            info = state["stages"][stage]
            if code == 0:
                break
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
    if state["wandb_mode"] == "offline":
        wandb_sync(p)
    write_summary(p, state)
    status(p, "ALL STAGES DONE")
    state["finished"] = now()
    save_state(p, state)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="tiny end-to-end check in artifacts/dryrun_t4/")
    ap.add_argument("--child", choices=STAGES, help=argparse.SUPPRESS)       # internal: run one stage
    ap.add_argument("--tag", default="t4", help="names the log/state folder artifacts/logs/<tag>/")
    ap.add_argument("--device-profile", default="local", choices=["local", "kaggle", "colab"])
    ap.add_argument("--num-workers", type=int, default=None, help="data workers (default: 0 locally, profile value elsewhere)")
    ap.add_argument("--data-root", default=None,
                    help="override the profile's data_root (e.g. a fake nested input/ folder, to rehearse Kaggle paths)")
    args = ap.parse_args()
    p = Paths(args.dry_run, args.tag, args.device_profile, args.num_workers, args.data_root)
    if args.child:
        STAGE_FUNCS[args.child](p, load_state(p))
        return
    sys.exit(supervise(p))


if __name__ == "__main__":
    main()
