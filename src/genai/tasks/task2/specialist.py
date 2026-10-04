"""Task 2 specialists: three denoising autoencoders, one per corruption (PDF page 5).

Public functions
----------------
train_specialist(cfg, resume=None, on_checkpoint=None) -> run directory
    Trains ONE specialist. Which one is chosen with cfg["specialist"]["corruption"]
    (salt | blur | occlusion, command line: --set specialist.corruption=salt).
study_specialists(cfg, on_checkpoint=None) -> study directory
    The shared Optuna search (study t2_specialist_shared). The PDF allows one search for a
    common architecture, then three independent trainings with it.
    ONE TRIAL COSTS THREE TRAININGS: every trial trains a short specialist for each of the
    three corruptions with the same sampled values and scores the MEAN of their three best J.
write_final_config(study_db, out_path, cfg=None) -> path of the final training config
    Copies the best trial's values into configs/task2_specialist_final.yaml.
    Command line:  python -m genai.tasks.task2.specialist write-final-config --study-db ... --out ...

What one specialist run does (same as Task 1, only the data and the validation rows differ)
-------------------------------------------------------------------------------------------
* data: PetsTrainDataset with policy "fixed:<cond_id>": the specialist sees ONLY its own
  corruption (salt 1, blur 2, occlusion 3), freshly sampled on every load; the target is the
  clean image. Validation uses only the val-manifest rows of its own corruption.
* loss:  alpha * L1 + (1 - alpha) * (1 - SSIM)   (task1.train.restoration_loss)
* score: fixed objective J = 0.5*L1 + 0.5*(1-SSIM) on those rows (lower is better); the best
  epoch by J is saved as ckpt_best.pt. PSNR, SSIM, L1 and a per-severity J are logged too.
* files in artifacts/runs/task2_specialist_<corruption>/<run_id>/: config.yaml, ckpt_last.pt,
  ckpt_best.pt, metrics.jsonl, samples/. Resume works exactly as in Task 1.
* every checkpoint's `config` contains: component ("specialist"), specialist {corruption,
  cond_id}, the full `model` section (UniversalAE.from_config(ckpt["config"]["model"]) rebuilds
  the model), `train`, `run_id`, the seed and so on.

This file never reads the locked official test set (only the val manifest).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import time
from pathlib import Path

import optuna
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from genai.common import tracking
from genai.common.checkpoint import (build_checkpoint, capture_rng_state, load_checkpoint,
                                     restore_rng_state, save_checkpoint)
from genai.common.metrics import evaluate_restoration
from genai.common.seed import seed_everything, worker_init_fn
from genai.models.autoencoder import UniversalAE, describe
from genai.pets import samplers
from genai.pets.dataset import PetsManifestDataset, PetsTrainDataset, resolve_data_paths
from genai.pets.split import split_sha256
from genai.tasks.task1.config import deep_copy, is_tbd, load_config, make_run_id, resolve, set_dotted
from genai.tasks.task1.train import (N_SAMPLE_IMAGES, VAL_MANIFEST_NAME, _adopt_run_dir,
                                     _check_resume_compatible, _SkipFirst, _unique_run_dir,
                                     restoration_loss, save_sample_grid)
from genai.tasks.task1.tune import export_study, suggest_params
from genai.tasks.task2 import SPECIALIST_COND_ID
from genai.tasks.task2 import runs

# The three corruptions in the order a trial trains them (salt, blur, occlusion).
CORRUPTIONS = tuple(SPECIALIST_COND_ID)

# Where each tuned parameter goes inside the config (names are the ones in tuned_params).
PARAM_TARGETS = {
    "lr": "train.lr",
    "bottleneck": "model.bottleneck_dim",
    "channels": "model.base_channels",
    "batch": "train.batch_size",
    "l1_ssim_weight": "train.alpha",
}

BASE_CONFIG = "configs/task2_specialist.yaml"
TRIAL_ROOT = "t2_trials"        # trial runs live in <output_root>/t2_trials/runs/..., away from the final runs
DEVICE_KEYS = ("device", "device_profile", "data_root", "dataset_dir", "output_root", "persist_root",
               "num_workers")   # merged in from configs/devices/*.yaml; not part of a final config


# ------------------------------------------------------------------------------------ data
def build_train_loader(cfg: dict, paths: dict, batch_size: int, cond_id: int):
    """Train dataset + loader of ONE specialist. Returns (dataset, sampler_wrapper, loader).

    Policy "fixed:<cond_id>": every item gets the corruption of this specialist, with new random
    parameters each time it is loaded. The batch sampler is the same one Task 1 uses.
    """
    tc = cfg["train"]
    policy = f"fixed:{cond_id}"
    ds = PetsTrainDataset("train", policy, data_root=paths, seed=cfg["seed"])
    if tc.get("train_subset"):                 # smoke / dry runs: keep only the first N images
        ds.ids = ds.ids[: int(tc["train_subset"])]
    sampler = _SkipFirst(samplers.make_batch_sampler(policy, len(ds), batch_size, cfg["seed"]))
    nw = int(cfg.get("num_workers", 0))
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=nw, worker_init_fn=worker_init_fn,
                        persistent_workers=nw > 0, pin_memory=torch.cuda.is_available())
    return ds, sampler, loader


def build_val_loader(cfg: dict, paths: dict, cond_id: int):
    """Val loader with ONLY the manifest rows of this specialist's corruption.

    The rows are filtered first and THEN cut with train.val_subset, so a "small" validation set
    still has N rows of the right corruption. Returns (dataset, row_indices, loader, manifest_sha256).
    """
    tc = cfg["train"]
    manifest = Path(paths["manifests"]) / VAL_MANIFEST_NAME
    ds = PetsManifestDataset(manifest, split="val", final_test=False, data_root=paths)
    rows = [i for i, row in enumerate(ds.rows) if row["cond_id"] == cond_id]
    if tc.get("val_subset"):
        rows = rows[: int(tc["val_subset"])]
    if not rows:
        raise ValueError(f"no validation rows with cond_id {cond_id} in {manifest}")
    nw = int(tc.get("val_workers", 0))   # 0 = load val data in the main process (saves RAM)
    loader = DataLoader(Subset(ds, rows), batch_size=int(tc.get("val_batch_size", 128)), shuffle=False,
                        num_workers=nw, persistent_workers=nw > 0)
    manifest_sha = Path(str(manifest) + ".sha256").read_text(encoding="utf-8").strip()
    return ds, rows, loader, manifest_sha


# ------------------------------------------------------------------------------ resume helpers
def _check_resume_specialist(saved: dict, corruption: str) -> None:
    """A checkpoint of one specialist must never be continued as another one."""
    if saved.get("component") != "specialist":
        raise ValueError(f"resume: the checkpoint is not a specialist checkpoint (component={saved.get('component')!r})")
    saved_corruption = (saved.get("specialist") or {}).get("corruption")
    if saved_corruption != corruption:
        raise ValueError(f"resume: the checkpoint belongs to the {saved_corruption!r} specialist but "
                         f"specialist.corruption is {corruption!r}")


# ------------------------------------------------------------------------------------- main
def _train_specialist(cfg: dict, resume=None, on_checkpoint=None):
    """Does the work of train_specialist; also returns the best J (used by the Optuna study).

    Modelled on genai.tasks.task1.train._train (duplicated on purpose, decision D22): the policy,
    the validation rows, the run folder and the tracker group differ.
    """
    cfg = copy.deepcopy(cfg)
    tc = cfg["train"]

    # ---- which specialist? write it into the config BEFORE it is saved anywhere ----
    if runs.get_component(cfg) != "specialist":
        raise ValueError("train_specialist needs a config with component: specialist")
    corruption = runs.get_corruption(cfg)                  # clear ValueError if missing / invalid
    cond_id = SPECIALIST_COND_ID[corruption]
    cfg["specialist"] = dict(cfg["specialist"], corruption=corruption, cond_id=cond_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(tc.get("amp")) and device.type == "cuda"
    torch.backends.cudnn.benchmark = True   # faster convs for fixed sizes; tiny non-determinism (CONTRACTS 3.11)

    # ---- resume: find the checkpoint first (it decides run_id / run_dir) ----
    state, ckpt_path = None, None
    if resume == "auto":
        ckpt_path = runs.find_latest_checkpoint(cfg)       # newest ckpt_last.pt of THIS corruption
        print("resume=auto ->", ckpt_path or "no checkpoint found, starting a fresh run")
    elif resume:
        ckpt_path = Path(resume)
    if ckpt_path is not None:
        state = load_checkpoint(ckpt_path)
        _check_resume_specialist(state["config"], corruption)
        _check_resume_compatible(state["config"], cfg)
        cfg["run_id"] = state["config"]["run_id"]
        run_dir = runs.runs_dir(cfg) / cfg["run_id"]
        _adopt_run_dir(Path(ckpt_path), run_dir)
    else:
        base = runs.runs_dir(cfg)
        wanted = cfg.get("run_id")                          # chosen by the caller (the Optuna study does this)
        wanted = wanted or make_run_id(cfg["device"], runs.run_desc(cfg), bool(cfg["run"].get("smoke")))
        rid, run_dir = _unique_run_dir(base, wanted)
        cfg["run_id"] = rid
        run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "samples").mkdir(exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"run dir: {run_dir}")

    seed_everything(cfg["seed"])

    # ---- data ----
    paths = resolve_data_paths(cfg)
    batch_size = int(tc["batch_size"])
    train_ds, sampler, train_loader = build_train_loader(cfg, paths, batch_size, cond_id)
    val_ds, val_rows, val_loader, manifest_sha = build_val_loader(cfg, paths, cond_id)
    split_sha = split_sha256(train_ds.split)
    if state is not None and (state["split_sha256"] != split_sha or state["manifest_sha256"] != manifest_sha):
        raise ValueError("resume: split / manifest hashes differ from the checkpoint (different data)")
    sample_idx = val_rows[:N_SAMPLE_IMAGES]     # the same fixed rows of this corruption at every sample step

    # ---- schedule length (epochs may be TBD; then max_steps is required) ----
    steps_per_epoch = len(sampler.inner)
    if steps_per_epoch == 0:
        raise ValueError(f"train_subset {tc.get('train_subset')} gives no full batch of {batch_size}")
    max_steps = tc.get("max_steps")
    if is_tbd(tc["epochs"]):
        if not max_steps:
            raise ValueError("train.epochs is TBD_AFTER_BENCHMARK: set a real epoch count or train.max_steps")
        epochs = math.ceil(int(max_steps) / steps_per_epoch)
    else:
        epochs = int(tc["epochs"])
    total_steps = min(epochs * steps_per_epoch, int(max_steps)) if max_steps else epochs * steps_per_epoch

    # ---- model, optimizer, scheduler, scaler ----
    model = UniversalAE.from_config(cfg["model"]).to(device)
    print(describe(model))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tc["lr"]), weight_decay=float(tc["weight_decay"]))
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
                 if tc.get("scheduler", "cosine") == "cosine" else None)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epoch, step_in_epoch, global_step, best_J, loss_sum = 0, 0, 0, None, 0.0
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        epoch, step_in_epoch = state["epoch"], state["step_in_epoch"]
        global_step, best_J, loss_sum = state["global_step"], state["best_metric"], state["epoch_loss_sum"]
        restore_rng_state(state["rng"])          # python / numpy / torch / cuda generators
        print(f"resumed from {ckpt_path}: epoch {epoch}, step_in_epoch {step_in_epoch}, global_step {global_step}")

    def save(name: str) -> Path:
        """Write ckpt_<name>.pt atomically and tell the caller (e.g. copy to Drive)."""
        ckpt = build_checkpoint(model, optimizer, scheduler, scaler, epoch=epoch, global_step=global_step,
                                best_metric=best_J, config=cfg, seed=cfg["seed"],
                                split_sha256=split_sha, manifest_sha256=manifest_sha)
        ckpt["step_in_epoch"], ckpt["epoch_loss_sum"], ckpt["rng"] = step_in_epoch, loss_sum, capture_rng_state()
        path = run_dir / f"ckpt_{name}.pt"
        save_checkpoint(path, ckpt)
        if on_checkpoint is not None:
            on_checkpoint(path)
        return path

    tracking.init_run(cfg, cfg["run_id"], project=cfg["run"].get("project", "genai-a1"), group=runs.TRACKER_GROUP)
    ckpt_every = float(tc.get("checkpoint_every_minutes", 10)) * 60
    log_every = int(tc.get("log_every_steps", 50))
    sample_every = int(tc.get("sample_every_epochs", 1))
    pause_after = tc.get("pause_after_steps")
    last_ckpt_time, paused = time.monotonic(), False

    try:
        model.train()
        while epoch < epochs and global_step < total_steps and not paused:
            sampler.inner.epoch = epoch              # fixes the batch order of this epoch
            sampler.skip = step_in_epoch             # already-trained batches (mid-epoch resume)
            t_epoch = time.monotonic()
            for corrupted, clean, _cond, _sev in train_loader:    # _cond is always cond_id here
                corrupted = corrupted.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    restored = model(corrupted)
                loss = restoration_loss(restored.float(), clean, float(tc["alpha"]))   # loss in float32
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if tc.get("grad_clip"):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc["grad_clip"]))
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
                step_in_epoch += 1
                loss_sum += loss.item()
                if global_step % log_every == 0:
                    tracking.log({"train/loss": loss.item(), "train/lr": optimizer.param_groups[0]["lr"]}, global_step)
                if time.monotonic() - last_ckpt_time >= ckpt_every:
                    save("last")
                    last_ckpt_time = time.monotonic()
                if pause_after and global_step >= int(pause_after):
                    paused = True                    # simulated interruption: checkpoint, no validation
                    break
                if global_step >= total_steps:
                    break
            sampler.skip = 0

            if paused:
                save("last")
                print(f"paused at step {global_step} (train.pause_after_steps); resume to continue")
                break

            # ---- epoch finished (or step cap reached): validate, record, checkpoint ----
            epoch_done = step_in_epoch >= steps_per_epoch
            res = evaluate_restoration(model, val_loader, device)   # only this corruption's rows
            val_J = res["overall"]["J"]
            row = {
                "epoch": epoch + 1, "global_step": global_step, "partial_epoch": not epoch_done,
                "corruption": corruption,
                "train_loss": loss_sum / max(step_in_epoch, 1), "lr": optimizer.param_groups[0]["lr"],
                "val_J": val_J, "val_l1": res["overall"]["l1"], "val_ssim": res["overall"]["ssim"],
                "val_psnr": res["overall"]["psnr"], "val_rows": res["overall"]["count"],
                "per_condition": {k: {m: v[m] for m in ("J", "l1", "ssim", "psnr")}
                                  for k, v in res.items() if k not in ("overall", "by_severity")},
                "by_severity": {k: v["J"] for k, v in res["by_severity"].items()},
                "epoch_seconds": time.monotonic() - t_epoch,
            }
            with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            flat = {"val/J": val_J, "val/l1": row["val_l1"], "val/ssim": row["val_ssim"],
                    "val/psnr": row["val_psnr"], "train/epoch_loss": row["train_loss"]}
            flat.update({f"val/J_{k}": v for k, v in row["by_severity"].items()})
            tracking.log(flat, global_step)
            print(f"[{corruption}] epoch {epoch + 1} step {global_step}: train_loss {row['train_loss']:.4f}  "
                  f"val_J {val_J:.4f}  ssim {row['val_ssim']:.3f}  psnr {row['val_psnr']:.2f}")

            if (epoch + 1) % sample_every == 0 or global_step >= total_steps:
                grid = save_sample_grid(model, val_ds, sample_idx, device,
                                        run_dir / "samples" / f"epoch{epoch + 1:03d}_step{global_step}.png")
                tracking.log_images("val/samples (clean, input, restored, error)", grid, global_step)

            if epoch_done:                           # advance the counters BEFORE saving
                epoch += 1
                step_in_epoch, loss_sum = 0, 0.0
            if best_J is None or val_J < best_J:     # lower J is better
                best_J = val_J
                save("best")
            save("last")
            last_ckpt_time = time.monotonic()
            if global_step >= total_steps:
                break
    finally:
        tracking.finish()
    return run_dir, best_J


def train_specialist(cfg: dict, resume: str | None = None, on_checkpoint=None) -> Path:
    """Train ONE specialist (see module docstring). Returns the run directory."""
    run_dir, _best = _train_specialist(cfg, resume=resume, on_checkpoint=on_checkpoint)
    return run_dir


# ------------------------------------------------------------------------------------ study
def make_trial_config(cfg: dict, values: dict, trial_number: int, corruption: str) -> dict:
    """Copy of the base config for ONE of the three trainings of a trial.

    The three trainings of a trial get the same `values`; only the corruption differs.
    """
    tcfg = deep_copy(cfg)
    for name, value in values.items():
        set_dotted(tcfg, PARAM_TARGETS[name], value)
    tcfg["specialist"] = {"corruption": corruption}
    tcfg["train"]["epochs"] = cfg["epochs_per_trial"]
    if cfg.get("trial_train_subset"):
        tcfg["train"]["train_subset"] = cfg["trial_train_subset"]
    # validation rows PER corruption: trial_val_subset wins; otherwise train.val_subset (the dry run sets it)
    if cfg.get("trial_val_subset"):
        tcfg["train"]["val_subset"] = cfg["trial_val_subset"]
    tcfg["train"]["checkpoint_every_minutes"] = 1e9      # trial runs only save at epoch ends
    tcfg["train"]["sample_every_epochs"] = 10 ** 6       # no sample grids during trials (except the last epoch)
    desc = f"{cfg['study']}-trial{trial_number}-{corruption}"
    tcfg["run_id"] = make_run_id(cfg["device"], desc, smoke=bool(cfg["run"].get("smoke")))
    tcfg["run"] = dict(cfg["run"], desc=desc)
    # trial runs go to <output_root>/t2_trials/runs/task2_specialist_<corruption>/ so that resume="auto"
    # of a real specialist run can never pick up the unfinished checkpoint of a trial
    tcfg["output_root"] = tcfg["persist_root"] = f"{str(cfg['output_root']).rstrip('/')}/{TRIAL_ROOT}"
    return tcfg


def study_specialists(cfg: dict, on_checkpoint=None) -> Path:
    """Run (or continue) the shared specialist study; returns the folder holding trials.csv and plots."""
    cfg = deep_copy(cfg)
    n_trials, epochs = cfg["n_trials"], cfg["epochs_per_trial"]
    if is_tbd(n_trials) or is_tbd(epochs):
        raise ValueError("n_trials / epochs_per_trial are TBD_AFTER_BENCHMARK: measure first "
                         "(scripts/benchmark.py) or pass explicit values (--n-trials / --epochs)")

    db_path = resolve(cfg["storage"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # mean J: lower is better. open_study adds the Task 1 fixes: sampler seed + number of existing
    # trials, and trials left RUNNING by a killed process are marked FAIL (runs.py, D42).
    study = runs.open_study(cfg, "minimize", f"sqlite:///{db_path.as_posix()}")

    def objective(trial: optuna.Trial) -> float:
        values = suggest_params(trial, cfg["tuned_params"])
        best_js = {}                                          # corruption -> best val J of its training
        for step, corruption in enumerate(CORRUPTIONS, start=1):
            tcfg = make_trial_config(cfg, values, trial.number, corruption)
            run_dir, best_J = _train_specialist(tcfg)
            best_js[corruption] = best_J
            trial.set_user_attr(f"J_{corruption}", best_J)
            if not cfg.get("trial_keep_checkpoints"):         # the trial only needs its J: save disk space
                for ckpt in run_dir.glob("ckpt_*.pt"):
                    ckpt.unlink()

            # Pruning: report the RUNNING MEAN after each corruption (steps 1, 2, 3) and let the
            # MedianPruner decide. This is the simplest choice: each corruption's training is already
            # short, and every trial reports the same quantity at the same step (step 1 = J of salt,
            # step 2 = mean of salt and blur, step 3 = the final value), so the comparison between
            # trials is fair. A bad hyperparameter set is therefore stopped after one or two of the
            # three trainings instead of all three.
            trial.report(sum(best_js.values()) / len(best_js), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return sum(best_js.values()) / len(best_js)

    def after_trial(study_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Copy the DB to the persistent folder (Drive on Colab) and notify the caller."""
        persist = resolve(cfg["persist_root"]) / "optuna" / db_path.name
        if persist.resolve() != db_path.resolve():
            persist.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(db_path, persist)
        if on_checkpoint is not None:
            on_checkpoint(db_path)

    # the budget counts trials that gave an answer (COMPLETE or PRUNED); failed trials are retried
    already = runs.n_answered(study)
    study.optimize(objective, n_trials=max(int(n_trials) - already, 0), callbacks=[after_trial])

    out_dir = resolve(cfg["study_dir"])
    export_study(study, out_dir)                               # trials.csv has the J_<corruption> columns too
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"study {cfg['study']}: {len(done)} complete / {len(study.trials)} total trials; exported to {out_dir}")
    if done:
        print(f"best mean J {study.best_value:.5f} with {study.best_params}")
    return out_dir


# ------------------------------------------------------------------------------ final config
def write_final_config(study_db, out_path="configs/task2_specialist_final.yaml", cfg=None) -> Path:
    """Write a FULL specialist training config that uses the best trial of the study.

    study_db : the SQLite file of the finished study (artifacts/optuna/t2_specialist_shared.db)
    out_path : where to write the YAML (relative paths are relative to the repository root)
    cfg      : the base config (default: configs/task2_specialist.yaml); its `study` is the study name

    The file keeps everything of the base config, replaces the tuned values with the best ones
    (same name -> config map as the study), leaves specialist.corruption null (choose it with
    --set specialist.corruption=salt|blur|occlusion) and leaves train.epochs as TBD_AFTER_BENCHMARK
    until the student sets it. Block `final_config_source` records where the values came from.
    """
    final = deep_copy(cfg) if cfg is not None else load_config(BASE_CONFIG)
    for key in DEVICE_KEYS:                       # device values are merged in again by load_config
        final.pop(key, None)

    db_path = resolve(study_db)
    if not db_path.exists():                      # sqlite would silently create an empty database
        raise FileNotFoundError(f"study database not found: {db_path}")
    study = optuna.load_study(study_name=final["study"], storage=f"sqlite:///{db_path.as_posix()}")
    try:
        best = study.best_trial
    except ValueError:
        raise ValueError(f"study {final['study']!r} in {db_path} has no completed trial yet") from None

    for name, value in best.params.items():       # same mapping the study used
        set_dotted(final, PARAM_TARGETS[name], value)
    final["specialist"] = dict(final.get("specialist") or {}, corruption=None)
    final["train"]["epochs"] = "TBD_AFTER_BENCHMARK"
    final["final_config_source"] = {
        "study": final["study"],
        "best_trial": best.number,
        "best_value": best.value,
        "best_params": dict(best.params),
        "J_per_corruption": {c: best.user_attrs.get(f"J_{c}") for c in CORRUPTIONS},
    }

    out = resolve(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ("# Final specialist config written by genai.tasks.task2.specialist.write_final_config.\n"
              "# Train each specialist with:  --config <this file> --set specialist.corruption=salt|blur|occlusion\n"
              "# train.epochs is still TBD_AFTER_BENCHMARK: the student sets it.\n")
    out.write_text(header + yaml.safe_dump(final, sort_keys=False), encoding="utf-8")
    print(f"final config written to {out} (best trial {best.number}, mean J {best.value:.5f})")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Task 2 specialist helpers")
    sub = ap.add_subparsers(dest="command", required=True)
    wf = sub.add_parser("write-final-config", help="copy the best trial of the shared study into a training config")
    wf.add_argument("--study-db", default=None, help="SQLite file (default: `storage` of the base config)")
    wf.add_argument("--out", default="configs/task2_specialist_final.yaml")
    wf.add_argument("--config", default=BASE_CONFIG, help="base config (its `study` is the study name)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    write_final_config(args.study_db or cfg["storage"], args.out, cfg=cfg)


if __name__ == "__main__":
    main()
