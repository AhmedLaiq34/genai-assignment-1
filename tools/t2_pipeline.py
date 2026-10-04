"""Unattended Task 2 pipeline (Kaggle or local), modelled on tools/t1_overnight.py.

Stages, in order (every one resumable; finished stages are skipped on a re-run):
    cls_study            Optuna study t2_classifier                (configs/task2_classifier.yaml)
    cls_final_config     configs/task2_classifier_final.yaml = study config + best trial + final epochs
    cls_train            final classifier training
    spec_study           shared Optuna study t2_specialist_shared  (configs/task2_specialist.yaml; 3 trainings per trial)
    spec_final_config    configs/task2_specialist_final.yaml = study config + best trial + final epochs
    spec_train_salt / spec_train_blur / spec_train_occlusion      the three final specialist trainings
    evaluate             hard-routed evaluation on the VAL manifest (oracle + predicted routing, input baseline)
    export               the four ONNX files + numeric parity + routing parity
    promote              models/checkpoints/t2_classifier.pt, t2_ae_salt.pt, t2_ae_blur.pt, t2_ae_occlusion.pt
                         (+ sha256 in models/MANIFEST.json)

Usage (repository root):
    python tools/t2_pipeline.py --device-profile kaggle        # notebooks/kaggle_t2_pipeline.ipynb does this
    python tools/t2_pipeline.py --dry-run                      # tiny end-to-end check in artifacts/dryrun_t2/

Budgets: n_trials / epochs_per_trial / train.epochs come from the two YAML files (decisions D44 and D47).
Why a supervisor: every stage runs in a CHILD process; if it crashes or its log is silent for STALL_MINUTES
it is killed and started again, and it resumes (Optuna SQLite with load_if_exists, ckpt_last.pt every epoch).
A stage that fails MAX_CONSECUTIVE_FAILURES times in a row without progress stops the pipeline.
The supervisor code repeats tools/t1_overnight.py (a Task 1 file that is not edited); candidate refactor (D44).

Never reads the test set: evaluation is on the VAL manifest only (final_test=False everywhere).
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

PIPELINE_VERSION = "t2-v3"   # printed by the notebook (Cell 1) and the supervisor: proves which code Kaggle runs

CLS_CONFIG = "configs/task2_classifier.yaml"
SPEC_CONFIG = "configs/task2_specialist.yaml"
CLS_FINAL = "configs/task2_classifier_final.yaml"
SPEC_FINAL = "configs/task2_specialist_final.yaml"
CORRUPTIONS = ("salt", "blur", "occlusion")
STAGES = (["cls_study", "cls_final_config", "cls_train", "spec_study", "spec_final_config"]
          + [f"spec_train_{c}" for c in CORRUPTIONS] + ["evaluate", "export", "promote"])
STALL_MINUTES = 20                       # one epoch + validation takes well under a minute
MAX_CONSECUTIVE_FAILURES = 4             # without progress
RETRY_PAUSE_SECONDS = 60
WANDB_ENV = {"TRACKER": "wandb", "WANDB_ENTITY": "ahmedlaiq34", "WANDB_PROJECT": "genai-a1",
             "WANDB_INIT_TIMEOUT": "300", "WANDB_SILENT": "false"}


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------------------------ paths and state
class Paths:
    """Every file the pipeline writes. The dry run redirects all of them into artifacts/dryrun_t2/."""

    def __init__(self, dry: bool, tag: str = "t2", profile: str = "local", num_workers=None, data_root=None):
        self.dry, self.tag, self.profile = dry, tag, profile
        self.data_root = data_root       # optional override of the profile's data_root (tests a Kaggle-like layout locally)
        # 0 data workers locally (Windows worker crash, D19); elsewhere the profile's own value
        self.num_workers = num_workers if num_workers is not None else (0 if profile == "local" else None)
        self.tmp = ROOT / "artifacts" / "dryrun_t2"
        self.logs = (self.tmp / "logs") if dry else (ROOT / "artifacts" / "logs" / tag)
        self.state = self.logs / "t2_state.json"
        self.status = self.logs / "t2_status.log"
        self.summary = self.logs / "t2_summary.json"
        self.cls_final = (self.tmp / "task2_classifier_final.yaml") if dry else (ROOT / CLS_FINAL)
        self.spec_final = (self.tmp / "task2_specialist_final.yaml") if dry else (ROOT / SPEC_FINAL)
        self.onnx_dir = (self.tmp / "onnx") if dry else (ROOT / "models" / "onnx")
        self.parity_csv = (self.tmp / "onnx_parity.csv") if dry else (ROOT / "report" / "tables" / "onnx_parity.csv")
        self.models_root = (self.tmp / "models") if dry else None      # None = the real models/ folder

    def stage_log(self, stage: str) -> Path:
        return self.logs / f"t2_{stage}.log"

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
    return {"stages": {s: {"done": False, "attempts": 0} for s in STAGES}, "wandb_mode": "online", "run_ids": {}}


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


# ------------------------------------------------------------------------------ configs
def study_cfg(p: Paths, which: str) -> dict:
    """Study config of 'cls' or 'spec'. The dry run shrinks it to a few minutes."""
    from genai.tasks.task1.config import load_config
    from genai.tasks.task1.tune import dry_run_overrides
    cfg = load_config(CLS_CONFIG if which == "cls" else SPEC_CONFIG, p.profile, p.overrides())
    if p.dry:
        cfg = dry_run_overrides(cfg, p.tmp)
        cfg.update(n_trials=2, epochs_per_trial=1, trial_train_subset=256, trial_val_subset=32)
        cfg["pruner"] = {"name": "Median", "n_startup_trials": 1, "n_warmup_steps": 1}
        cfg["train"]["val_subset"] = 64
    return cfg


def final_cfg(p: Paths, which: str, corruption: str | None = None) -> dict:
    from genai.tasks.task1.config import load_config
    cfg = load_config(p.cls_final if which == "cls" else p.spec_final, p.profile, p.overrides())
    if corruption:
        cfg["specialist"]["corruption"] = corruption
    if p.dry:
        cfg["output_root"] = cfg["persist_root"] = str(p.tmp)
        cfg["train"]["train_subset"], cfg["train"]["val_subset"] = 256, 64
        cfg["run"]["smoke"] = True
    return cfg


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


# ------------------------------------------------------------------------------ the stages (child process)
def _run_study(p: Paths, which: str) -> None:
    from genai.tasks.task1.config import resolve
    from genai.tasks.task2.tune import run_study
    cfg = study_cfg(p, which)
    out_dir = run_study(cfg)                                   # resumes; exports trials.csv + plots at the end
    shutil.copyfile(resolve(cfg["storage"]), Path(out_dir) / Path(cfg["storage"]).name)   # DB next to the exports


def stage_cls_study(p, state):
    _run_study(p, "cls")


def stage_spec_study(p, state):
    _run_study(p, "spec")


def _write_final(p: Paths, which: str, raw: dict, best, out: Path) -> None:
    """Final training config = study config + best parameters + the final schedule from the study YAML."""
    import yaml
    study_yaml = yaml.safe_load((ROOT / (CLS_CONFIG if which == "cls" else SPEC_CONFIG)).read_text(encoding="utf-8"))
    raw["train"]["epochs"] = 2 if p.dry else study_yaml["train"]["epochs"]
    raw["train"]["sample_every_epochs"] = 5
    raw["run"]["desc"] = f"{'t2cls' if which == 'cls' else 't2spec'}_final"
    raw["search_space_status"] = f"USED BY STUDY {raw['study']} (this file holds its best trial)"
    header = (f"# Task 2 FINAL {'classifier' if which == 'cls' else 'specialist'} config, written by tools/t2_pipeline.py on {now()}.\n"
              f"# Best trial of study '{raw['study']}': trial {best.number}, value {best.value:.5f}.\n"
              f"# Parameters copied from that trial: {json.dumps(best.params)}\n"
              f"# Schedule: {raw['train']['epochs']} epochs (budget D47).\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def stage_cls_final_config(p, state):
    import yaml
    from genai.tasks.task1.config import set_dotted
    from genai.tasks.task2.classifier import CHANNEL_CHOICES, PARAM_TARGETS
    best = open_study(study_cfg(p, "cls")).best_trial
    raw = yaml.safe_load((ROOT / CLS_CONFIG).read_text(encoding="utf-8"))
    for name, value in best.params.items():
        set_dotted(raw, PARAM_TARGETS[name], CHANNEL_CHOICES[value] if name == "conv_channels" else value)
    _write_final(p, "cls", raw, best, p.cls_final)


def stage_spec_final_config(p, state):
    import yaml
    from genai.tasks.task1.config import set_dotted
    from genai.tasks.task2.specialist import PARAM_TARGETS
    best = open_study(study_cfg(p, "spec")).best_trial
    raw = yaml.safe_load((ROOT / SPEC_CONFIG).read_text(encoding="utf-8"))
    for name, value in best.params.items():
        set_dotted(raw, PARAM_TARGETS[name], value)
    raw["final_config_source"] = {"study": raw["study"], "best_trial": best.number, "best_value": best.value,
                                  "J_per_corruption": {c: best.user_attrs.get(f"J_{c}") for c in CORRUPTIONS}}
    _write_final(p, "spec", raw, best, p.spec_final)


def _component_cfg(p: Paths, component: str) -> dict:
    return final_cfg(p, "cls") if component == "classifier" else final_cfg(p, "spec", component)


def _run_dir(p: Paths, state: dict, component: str) -> Path:
    from genai.tasks.task2.runs import runs_dir
    return runs_dir(_component_cfg(p, component)) / state["run_ids"][component]


def _train_component(p: Paths, state: dict, component: str) -> None:
    """Final training of one component, resuming from its own ckpt_last.pt (never from a trial run)."""
    from genai.tasks.task2.train import run_training
    cfg = _component_cfg(p, component)
    cfg["run_id"] = state["run_ids"][component]                # fixed by the supervisor: restarts reuse it
    run_dir = _run_dir(p, state, component)
    last = run_dir / "ckpt_last.pt"
    if last.exists():
        resume = str(last)
    else:
        resume = None
        if run_dir.exists():                                   # crashed before the first checkpoint: start clean
            shutil.rmtree(run_dir)
    run_training(cfg, resume=resume)
    (run_dir / "DONE").write_text(now(), encoding="utf-8")


def stage_cls_train(p, state):
    _train_component(p, state, "classifier")


def stage_spec_train_salt(p, state):
    _train_component(p, state, "salt")


def stage_spec_train_blur(p, state):
    _train_component(p, state, "blur")


def stage_spec_train_occlusion(p, state):
    _train_component(p, state, "occlusion")


def _best_ckpts(p: Paths, state: dict) -> dict:
    return {c: str(_run_dir(p, state, c) / "ckpt_best.pt") for c in ("classifier",) + CORRUPTIONS}


def stage_evaluate(p, state):
    from genai.tasks.task2.evaluation import evaluate_task2
    out = evaluate_task2(final_cfg(p, "cls"), _best_ckpts(p, state), final_test=False)
    state_note(p, "eval_dir", str(out))
    state_note(p, "eval_summary", json.loads((Path(out) / "summary.json").read_text(encoding="utf-8")))


def stage_export(p, state):
    from genai.common.constants import ONNX_FILES
    from genai.export.task2_export import export_task2_model, verify_routing_parity, verify_task2_parity
    from genai.tasks.task2 import ONNX_KEYS
    data_root = final_cfg(p, "cls")                            # on Kaggle the data is under /kaggle/input
    results = {}
    for component, ckpt in _best_ckpts(p, state).items():
        name = ONNX_KEYS[component]
        onnx_path = p.onnx_dir / ONNX_FILES[name]
        export_task2_model(name, ckpt, onnx_path)
        meta = json.loads(Path(str(onnx_path) + ".meta.json").read_text(encoding="utf-8"))
        if meta["smoke"] and not p.dry:
            raise RuntimeError(f"{name}: exported model is flagged smoke; refusing to treat it as final")
        r = verify_task2_parity(name, ckpt, onnx_path, n=16, tag="dryrun" if p.dry else "final",
                                csv_path=p.parity_csv, data_root=data_root)
        print(name, json.dumps(r))
        if not r["passed"]:
            raise RuntimeError(f"ONNX parity failed for {name}: max abs diff {r['max_abs_diff']}")
        results[name] = {k: r[k] for k in ("max_abs_diff", "mean_abs_diff", "passed")}
    routing = verify_routing_parity(_best_ckpts(p, state)["classifier"], p.onnx_dir / ONNX_FILES["t2_classifier"],
                                    n=16, data_root=data_root)
    print("routing parity", json.dumps(routing, default=str))
    if not routing["passed"]:
        raise RuntimeError("routing parity failed: the ONNX classifier picks a different class than PyTorch")
    results["routing"] = {"n_inputs": routing["n_inputs"], "n_agree": routing["n_agree"]}
    state_note(p, "parity", results)


def stage_promote(p, state):
    from genai.common.checkpoint import promote
    from genai.tasks.task2 import PROMOTED_NAMES
    promoted = {}
    for component, ckpt in _best_ckpts(p, state).items():
        promoted[component] = str(promote(ckpt, PROMOTED_NAMES[component], root=p.models_root))
    state_note(p, "promoted", promoted)


STAGE_FUNCS = {name: globals()[f"stage_{name}"] for name in STAGES}


# ------------------------------------------------------------------------------ progress measures
def progress_marker(p: Paths, state: dict, stage: str) -> int:
    """A number that grows when a stage makes progress (tells 'stuck' from 'slow but moving')."""
    if stage in ("cls_study", "spec_study"):
        c = study_counts(study_cfg(p, stage.split("_")[0]))
        return c.get("COMPLETE", 0) + c.get("PRUNED", 0)
    if stage.endswith("_train") or stage.startswith("spec_train_"):
        component = "classifier" if stage == "cls_train" else stage.rsplit("_", 1)[1]
        if component in state.get("run_ids", {}):
            metrics = _run_dir(p, state, component) / "metrics.jsonl"
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
    offline_dirs = sorted((ROOT / "wandb").glob("offline-run-*"))
    if offline_dirs:
        status(p, f"wandb sync of {len(offline_dirs)} offline run(s)")
        subprocess.call([sys.executable, "-m", "wandb", "sync", "--include-offline", *map(str, offline_dirs)],
                        cwd=ROOT, env=dict(os.environ, **WANDB_ENV))


def write_summary(p: Paths, state: dict) -> None:
    """Facts for the report: study counts and best trials, final-training curves, evaluation, parity."""
    summary = {"written": now(), "dry_run": p.dry, "wandb_mode": state.get("wandb_mode"),
               "stage_seconds": {s: v.get("seconds") for s, v in state["stages"].items()},
               "stage_attempts": {s: v["attempts"] for s, v in state["stages"].items()},
               "run_ids": state.get("run_ids"), "results": state.get("results", {})}
    for which in ("cls", "spec"):
        cfg = study_cfg(p, which)
        study = open_study(cfg)
        summary[f"{which}_study"] = {"name": study.study_name, "counts": study_counts(cfg),
                                     "best_trial": study.best_trial.number, "best_value": study.best_value,
                                     "best_params": study.best_params, "best_user_attrs": study.best_trial.user_attrs,
                                     "n_trials_budget": cfg["n_trials"], "epochs_per_trial": cfg["epochs_per_trial"],
                                     "pruner": cfg["pruner"], "search_space": cfg["tuned_params"]}
    for component in ("classifier",) + CORRUPTIONS:
        metrics = _run_dir(p, state, component) / "metrics.jsonl"
        if metrics.exists():
            rows = [json.loads(line) for line in open(metrics, encoding="utf-8")]
            key, pick = ("val_macro_f1", max) if component == "classifier" else ("val_J", min)
            best = pick(rows, key=lambda r: r[key])
            summary[f"final_{component}"] = {"epochs_logged": len(rows), "best_epoch": best["epoch"],
                                             f"best_{key}": best[key], "last": rows[-1]}
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
        status(p, f"preflight: wandb logged in as entity {wandb.Api().viewer.entity}")
    except Exception as err:
        status(p, f"preflight: wandb check failed ({err}); runs will be retried and may go offline")


def supervise(p: Paths) -> int:
    from genai.tasks.task1.config import make_run_id
    state = load_state(p)
    status(p, f"supervisor start (code version {PIPELINE_VERSION}, dry_run={p.dry}); "
              f"stages done: {[s for s in STAGES if state['stages'][s]['done']]}")
    preflight(p)
    for stage in STAGES:
        info = state["stages"][stage]
        if info["done"]:
            continue
        component = {"cls_train": "classifier"}.get(stage) or (stage.rsplit("_", 1)[1] if stage.startswith("spec_train_") else None)
        if component and component not in state["run_ids"]:
            desc = "t2cls_final" if component == "classifier" else f"t2spec_{component}_final"
            state["run_ids"][component] = make_run_id(p.profile, f"{desc}_{p.tag}", smoke=p.dry)
            save_state(p, state)
        status(p, f"{stage}: starting")
        t0, failures = time.time(), 0
        while True:
            before = progress_marker(p, state, stage)
            info["attempts"] += 1
            save_state(p, state)
            code = run_child(p, state, stage)
            state = load_state(p) | {"stages": state["stages"], "wandb_mode": state["wandb_mode"],
                                     "run_ids": state["run_ids"]}
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
        if stage in ("cls_study", "spec_study"):
            c = study_counts(study_cfg(p, stage.split("_")[0]))
            status(p, f"{stage} counts: {c}")
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
    ap.add_argument("--dry-run", action="store_true", help="tiny end-to-end check in artifacts/dryrun_t2/")
    ap.add_argument("--child", choices=STAGES, help=argparse.SUPPRESS)       # internal: run one stage
    ap.add_argument("--tag", default="t2", help="names the log/state folder artifacts/logs/<tag>/")
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
