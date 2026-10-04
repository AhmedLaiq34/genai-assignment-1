"""Task 2 corruption classifier: training, validation metrics, evaluation and the Optuna study.

Implements CONTRACTS 3.5 (balanced batches), 3.6 (macro-F1), 3.8 (checkpoints), 3.13 (study) and
plan 10.1 / 10.3. The loop is modelled on genai/tasks/task1/train.py (decision D22).

Public functions (called by the dispatchers task2/train.py, tune.py, evaluate.py):

    train_classifier(cfg, resume=None, on_checkpoint=None) -> run directory
    study_classifier(cfg, on_checkpoint=None)              -> study directory
    evaluate_classifier(cfg, checkpoint, final_test=False) -> evaluation directory
    classification_metrics(y_true, y_pred)                 -> dict (accuracy, macro P/R/F1, per class, confusion)
    save_confusion_matrix(cm_normalised, csv_path, png_path)
    load_classifier(ckpt_path)                             -> (model in eval mode, checkpoint dict)

What one training run does
--------------------------
* data: PetsTrainDataset with policy "balanced_batch" (every batch holds exactly batch_size/4 images
  of each class: 0 clean, 1 salt_pepper, 2 gaussian_blur, 3 occlusion). The label is the cond_id that
  the runtime corruption pipeline used, so no manual labelling is needed.
* loss: multiclass cross-entropy on the 4 logits.
* validation after every epoch: the fixed VAL MANIFEST (all 4 conditions). Reported: accuracy, macro
  precision / recall / F1, per-class numbers and the row-normalised 4x4 confusion matrix.
* best checkpoint = highest validation macro-F1 (CONTRACTS 3.6).
* files in artifacts/runs/task2_classifier/<run_id>/: config.yaml, ckpt_last.pt, ckpt_best.pt,
  metrics.jsonl, confusion_val_best.csv / .png (matrix of the best epoch), samples/ (one small
  confusion PNG per epoch).
* resilience as in Task 1: atomic checkpoints, full resume (also RNG and a mid-epoch skip), on_checkpoint hook.

The test split is never opened by the training / study code in this file. Only evaluate_classifier can
read it, and only when it is called with its final_test argument switched on.
"""
from __future__ import annotations

import copy
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from genai.common import tracking
from genai.common.checkpoint import (build_checkpoint, capture_rng_state, load_checkpoint,
                                     restore_rng_state, save_checkpoint)
from genai.common.constants import CLASS_NAMES, NUM_CLASSES
from genai.common.seed import seed_everything, worker_init_fn
from genai.models.classifier import CorruptionClassifier, describe
from genai.pets import samplers
from genai.pets.dataset import PetsManifestDataset, PetsTrainDataset, resolve_data_paths
from genai.pets.split import split_sha256
from genai.tasks.task1.config import deep_copy, is_tbd, make_run_id, resolve, set_dotted
from genai.tasks.task1.evaluate import TEST_MANIFEST_NAME
from genai.tasks.task1.train import (VAL_MANIFEST_NAME, _adopt_run_dir, _check_resume_compatible,
                                     _SkipFirst, _unique_run_dir, build_val_loader)
from genai.tasks.task1.tune import export_study, suggest_params
from genai.tasks.task2 import runs
from genai.tasks.task2.runs import TRACKER_GROUP, find_latest_checkpoint, run_desc, runs_dir


# ============================================================================== metrics
def classification_metrics(y_true, y_pred) -> dict:
    """Accuracy, macro precision / recall / F1, per-class numbers and the confusion matrices.

    y_true, y_pred: integer arrays of class ids (0 clean, 1 salt_pepper, 2 gaussian_blur, 3 occlusion).
    Returns
      accuracy, macro_precision, macro_recall, macro_f1       floats
      per_class            {class_name: {precision, recall, f1, support}}
      confusion_counts     4x4 list, row = true class, column = predicted class
      confusion_normalised 4x4 list, every row divided by its sum (a row with no samples stays zeros)

    Macro average = the plain mean of the four per-class values, so every class counts equally.
    """
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    labels = list(range(NUM_CLASSES))
    # zero_division=0: a class that is never predicted (or never present) gets 0 instead of a warning.
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    counts = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = counts.sum(axis=1, keepdims=True)
    normalised = np.divide(counts, row_sums, out=np.zeros(counts.shape, dtype=float), where=row_sums > 0)
    return {
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "macro_precision": float(prec.mean()),
        "macro_recall": float(rec.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {name: {"precision": float(prec[k]), "recall": float(rec[k]),
                             "f1": float(f1[k]), "support": int(support[k])}
                      for k, name in enumerate(CLASS_NAMES)},
        "confusion_counts": counts.tolist(),
        "confusion_normalised": normalised.tolist(),
    }


def _plot_confusion(cm, png_path: Path, title: str = "Normalised confusion matrix (rows = true class)") -> None:
    """Heat map of a 4x4 matrix with the number written in every cell."""
    import matplotlib
    matplotlib.use("Agg")                          # no window; we only save files
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    image = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(NUM_CLASSES), CLASS_NAMES)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(title, fontsize=9)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black")
    fig.colorbar(image, ax=ax, fraction=0.046)
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(cm_normalised, csv_path, png_path) -> None:
    """Write the row-normalised confusion matrix as CSV (class names as header and index) and as PNG."""
    cm = np.asarray(cm_normalised, dtype=float)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cm, index=list(CLASS_NAMES), columns=list(CLASS_NAMES)).to_csv(csv_path, index_label="true_class")
    _plot_confusion(cm, png_path)


@torch.no_grad()
def predict_loader(model, loader, device) -> tuple:
    """Run the classifier over a loader of (corrupted, clean, cond_id, severity_id) batches.

    Returns (y_true, y_pred, probs, mean_cross_entropy): numpy arrays of true classes, predicted
    classes, softmax probabilities (N x 4) and the mean cross-entropy. The model mode is restored.
    """
    was_training = model.training
    model.eval()                                    # dropout off, BatchNorm uses running statistics
    ys, preds, probs, loss_sum, n = [], [], [], 0.0, 0
    for corrupted, _clean, cond, _sev in loader:
        logits = model(corrupted.to(device)).float().cpu()
        loss_sum += F.cross_entropy(logits, cond, reduction="sum").item()
        n += len(cond)
        ys.append(cond.numpy())
        preds.append(logits.argmax(dim=1).numpy())
        probs.append(logits.softmax(dim=1).numpy())
    model.train(was_training)
    return np.concatenate(ys), np.concatenate(preds), np.concatenate(probs), loss_sum / max(n, 1)


# ============================================================================== data
def build_train_loader(cfg: dict, paths: dict, batch_size: int):
    """Train dataset + balanced loader. Returns (dataset, sampler_wrapper, loader)."""
    tc = cfg["train"]
    if batch_size % NUM_CLASSES != 0:
        raise ValueError(f"train.batch_size must be a multiple of {NUM_CLASSES} "
                         f"(balanced batches hold batch_size/{NUM_CLASSES} images per class), got {batch_size}")
    ds = PetsTrainDataset("train", "balanced_batch", data_root=paths, seed=cfg["seed"])
    if tc.get("train_subset"):                     # smoke / dry runs: keep only the first N images
        ds.ids = ds.ids[: int(tc["train_subset"])]
    # BalancedBatchSampler yields lists of (image_index, class_id): exactly B/4 of each class per batch.
    sampler = _SkipFirst(samplers.make_batch_sampler("balanced_batch", len(ds), batch_size, cfg["seed"]))
    nw = int(cfg.get("num_workers", 0))
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=nw, worker_init_fn=worker_init_fn,
                        persistent_workers=nw > 0, pin_memory=torch.cuda.is_available())
    return ds, sampler, loader


# ============================================================================== training
def _train(cfg: dict, resume=None, on_checkpoint=None, report_fn=None):
    """Does the work of train_classifier; returns (run_dir, best macro-F1). The Optuna study uses it too.

    report_fn(epoch, val_macro_f1): optional callback after each COMPLETED epoch's validation
    (Optuna uses it to report and prune; it may raise an exception to stop the run).
    """
    cfg = copy.deepcopy(cfg)
    cfg["component"] = "classifier"
    tc = cfg["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(tc.get("amp")) and device.type == "cuda"
    torch.backends.cudnn.benchmark = True   # faster convs for fixed sizes; tiny non-determinism (CONTRACTS 3.11)

    # ---- cheap checks first, so a wrong config fails before any folder is created ----
    batch_size = int(tc["batch_size"])
    if batch_size % NUM_CLASSES != 0:
        raise ValueError(f"train.batch_size must be a multiple of {NUM_CLASSES}, got {batch_size}")
    max_steps = tc.get("max_steps")
    if is_tbd(tc["epochs"]) and not max_steps:
        raise ValueError("train.epochs is TBD_AFTER_BENCHMARK: set a real epoch count or train.max_steps")

    # ---- resume: find the checkpoint first (it decides run_id / run_dir) ----
    state, ckpt_path = None, None
    if resume == "auto":
        ckpt_path = find_latest_checkpoint(cfg)
        print("resume=auto ->", ckpt_path or "no checkpoint found, starting a fresh run")
    elif resume:
        ckpt_path = Path(resume)
    if ckpt_path is not None:
        state = load_checkpoint(ckpt_path)
        _check_resume_compatible(state["config"], cfg)
        cfg["run_id"] = state["config"]["run_id"]
        run_dir = runs_dir(cfg) / cfg["run_id"]
        _adopt_run_dir(Path(ckpt_path), run_dir)
    else:
        base = runs_dir(cfg)
        if cfg.get("run_id"):                     # chosen by the caller (the Optuna study does this)
            rid = cfg["run_id"]
            run_dir = base / rid
        else:
            smoke = bool(cfg["run"].get("smoke"))
            rid, run_dir = _unique_run_dir(base, make_run_id(cfg["device"], run_desc(cfg), smoke))
        cfg["run_id"] = rid
        run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "samples").mkdir(exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"run dir: {run_dir}")

    seed_everything(cfg["seed"])

    # ---- data ----
    paths = resolve_data_paths(cfg)
    train_ds, sampler, train_loader = build_train_loader(cfg, paths, batch_size)
    _val_ds, _n_val, val_loader, manifest_sha = build_val_loader(cfg, paths)
    split_sha = split_sha256(train_ds.split)
    if state is not None and (state["split_sha256"] != split_sha or state["manifest_sha256"] != manifest_sha):
        raise ValueError("resume: split / manifest hashes differ from the checkpoint (different data)")

    # ---- schedule length (epochs may be TBD; then max_steps is required) ----
    steps_per_epoch = len(sampler.inner)
    if is_tbd(tc["epochs"]):
        epochs = math.ceil(int(max_steps) / steps_per_epoch)
    else:
        epochs = int(tc["epochs"])
    total_steps = min(epochs * steps_per_epoch, int(max_steps)) if max_steps else epochs * steps_per_epoch

    # ---- model, optimizer, scheduler, scaler ----
    model = CorruptionClassifier.from_config(cfg["model"]).to(device)
    print(describe(model))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tc["lr"]), weight_decay=float(tc["weight_decay"]))
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
                 if tc.get("scheduler", "cosine") == "cosine" else None)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epoch, step_in_epoch, global_step, best_f1 = 0, 0, 0, None
    loss_sum, correct_sum = 0.0, 0                 # running sums over the current epoch
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        epoch, step_in_epoch = state["epoch"], state["step_in_epoch"]
        global_step, best_f1, loss_sum = state["global_step"], state["best_metric"], state["epoch_loss_sum"]
        correct_sum = state.get("epoch_correct_sum", 0)
        restore_rng_state(state["rng"])          # python / numpy / torch / cuda generators
        print(f"resumed from {ckpt_path}: epoch {epoch}, step_in_epoch {step_in_epoch}, global_step {global_step}")

    def save(name: str) -> Path:
        """Write ckpt_<name>.pt atomically and tell the caller (e.g. copy to Drive)."""
        ckpt = build_checkpoint(model, optimizer, scheduler, scaler, epoch=epoch, global_step=global_step,
                                best_metric=best_f1, config=cfg, seed=cfg["seed"],     # best_metric = macro-F1
                                split_sha256=split_sha, manifest_sha256=manifest_sha)
        ckpt["step_in_epoch"], ckpt["epoch_loss_sum"], ckpt["rng"] = step_in_epoch, loss_sum, capture_rng_state()
        ckpt["epoch_correct_sum"] = correct_sum
        path = run_dir / f"ckpt_{name}.pt"
        save_checkpoint(path, ckpt)
        if on_checkpoint is not None:
            on_checkpoint(path)
        return path

    tracking.init_run(cfg, cfg["run_id"], project=cfg["run"].get("project", "genai-a1"), group=TRACKER_GROUP)
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
            for corrupted, _clean, cond, _sev in train_loader:
                corrupted = corrupted.to(device, non_blocking=True)
                labels = cond.to(device, non_blocking=True)          # the class the corruption pipeline applied
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    logits = model(corrupted)
                loss = F.cross_entropy(logits.float(), labels)       # loss in float32
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
                correct_sum += (logits.argmax(dim=1) == labels).sum().item()
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
            y_true, y_pred, _probs, val_loss = predict_loader(model, val_loader, device)
            m = classification_metrics(y_true, y_pred)
            val_f1 = m["macro_f1"]
            seen = max(step_in_epoch, 1) * batch_size
            row = {
                "epoch": epoch + 1, "global_step": global_step, "partial_epoch": not epoch_done,
                "train_loss": loss_sum / max(step_in_epoch, 1), "train_accuracy": correct_sum / seen,
                "lr": optimizer.param_groups[0]["lr"],
                "val_loss": val_loss, "val_accuracy": m["accuracy"], "val_macro_precision": m["macro_precision"],
                "val_macro_recall": m["macro_recall"], "val_macro_f1": val_f1, "per_class": m["per_class"],
                "epoch_seconds": time.monotonic() - t_epoch,
            }
            with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            flat = {"val/macro_f1": val_f1, "val/accuracy": m["accuracy"], "val/macro_precision": m["macro_precision"],
                    "val/macro_recall": m["macro_recall"], "val/loss": val_loss, "train/epoch_loss": row["train_loss"],
                    "train/epoch_accuracy": row["train_accuracy"]}
            flat.update({f"val/f1_{name}": v["f1"] for name, v in m["per_class"].items()})
            tracking.log(flat, global_step)
            print(f"epoch {epoch + 1} step {global_step}: train_loss {row['train_loss']:.4f}  "
                  f"val_loss {val_loss:.4f}  val_acc {m['accuracy']:.3f}  val_macro_f1 {val_f1:.4f}")

            if (epoch + 1) % sample_every == 0 or global_step >= total_steps:
                _plot_confusion(m["confusion_normalised"], run_dir / "samples" / f"confusion_epoch{epoch + 1:03d}_step{global_step}.png")

            if epoch_done:                           # advance the counters BEFORE saving
                epoch += 1
                step_in_epoch, loss_sum, correct_sum = 0, 0.0, 0
            improved = best_f1 is None or val_f1 > best_f1       # macro-F1: higher is better
            if improved:
                best_f1 = val_f1
                save_confusion_matrix(m["confusion_normalised"], run_dir / "confusion_val_best.csv",
                                      run_dir / "confusion_val_best.png")
                save("best")
            save("last")
            last_ckpt_time = time.monotonic()
            if report_fn is not None and epoch_done:
                report_fn(epoch, val_f1)             # Optuna: may raise TrialPruned
            if global_step >= total_steps:
                break
    finally:
        tracking.finish()
    return run_dir, best_f1


def train_classifier(cfg: dict, resume: str | None = None, on_checkpoint=None) -> Path:
    """Train the corruption classifier (see module docstring). Returns the run directory."""
    run_dir, _best = _train(cfg, resume=resume, on_checkpoint=on_checkpoint)
    return run_dir


# ============================================================================== loading + evaluation
def load_classifier(ckpt_path) -> tuple:
    """Rebuild the classifier from a checkpoint. Returns (model in eval mode, checkpoint dict)."""
    ckpt = load_checkpoint(ckpt_path)
    model = CorruptionClassifier.from_config(ckpt["config"]["model"])
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt          # eval(): dropout off, BatchNorm uses running statistics


def evaluate_classifier(cfg: dict, checkpoint, final_test: bool = False) -> Path:
    """Evaluate a classifier checkpoint on the val manifest (or the locked test manifest).

    Writes to <output_root>/eval/task2/classifier_<run_id>_<val|test>/:
      classification_report.json   accuracy, macro P/R/F1, per class, both confusion matrices, checkpoint info
      confusion_normalised.csv/.png   row-normalised 4x4 matrix
      confusion_counts.csv         raw counts
      predictions.csv              one row per manifest row: true / predicted class, the 4 probabilities
    Returns that directory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_classifier(checkpoint)
    model = model.to(device)

    paths = resolve_data_paths(cfg)
    split_name = "test" if final_test else "val"
    manifest = Path(paths["manifests"]) / (TEST_MANIFEST_NAME if final_test else VAL_MANIFEST_NAME)
    ds = PetsManifestDataset(manifest, split=split_name, final_test=final_test, data_root=paths)

    run_id = ckpt["config"].get("run_id", Path(checkpoint).parent.name)
    out_dir = resolve(cfg["output_root"]) / "eval" / "task2" / f"classifier_{run_id}_{split_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int((cfg.get("train") or {}).get("val_batch_size", 128))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    y_true, y_pred, probs, loss = predict_loader(model, loader, device)
    m = classification_metrics(y_true, y_pred)

    report = {"checkpoint": str(checkpoint), "run_id": run_id, "split": split_name, "n_rows": int(len(y_true)),
              "global_step": ckpt["global_step"], "val_cross_entropy": loss, **m}
    (out_dir / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_confusion_matrix(m["confusion_normalised"], out_dir / "confusion_normalised.csv", out_dir / "confusion_normalised.png")
    pd.DataFrame(m["confusion_counts"], index=list(CLASS_NAMES), columns=list(CLASS_NAMES)).to_csv(
        out_dir / "confusion_counts.csv", index_label="true_class")

    rows = []
    for i in range(len(y_true)):
        r = ds.rows[i]
        rows.append({"row": i, "image_id": r["image_id"], "cond": r["cond_name"], "severity": r["severity"] or "none",
                     "true_class": int(y_true[i]), "predicted_class": int(y_pred[i]),
                     "correct": bool(y_true[i] == y_pred[i]),
                     **{f"p_{name}": float(probs[i, k]) for k, name in enumerate(CLASS_NAMES)}})
    pd.DataFrame(rows).to_csv(out_dir / "predictions.csv", index=False)
    print(f"evaluated {len(y_true)} rows on {split_name}: accuracy {m['accuracy']:.4f} "
          f"macro-F1 {m['macro_f1']:.4f}; results in {out_dir}")
    return out_dir


# ============================================================================== Optuna study
# Where each tuned parameter goes inside the config (the names are the ones in tuned_params).
PARAM_TARGETS = {
    "lr": "train.lr",
    "batch": "train.batch_size",             # every choice must be a multiple of 4 (balanced batches)
    "conv_channels": "model.channels",       # a NAME from CHANNEL_CHOICES, converted to a list below
    "dropout": "model.dropout",
    "weight_decay": "train.weight_decay",
}

# Optuna can only store simple values, so the channel configuration is tuned as a string
# and turned into the list of block widths here. PROVISIONAL list of options.
CHANNEL_CHOICES = {
    "16-32-64": [16, 32, 64],
    "32-64-128": [32, 64, 128],
    "16-32-64-128": [16, 32, 64, 128],
    "32-64-128-256": [32, 64, 128, 256],
}


def make_trial_config(cfg: dict, values: dict, trial_number: int) -> dict:
    """Copy of the base config with the sampled values and the trial's epoch budget inserted."""
    tcfg = deep_copy(cfg)
    for name, value in values.items():
        if name == "conv_channels":
            value = CHANNEL_CHOICES[value]                   # "16-32-64" -> [16, 32, 64]
        set_dotted(tcfg, PARAM_TARGETS[name], value)
    tcfg["train"]["epochs"] = cfg["epochs_per_trial"]
    if cfg.get("trial_train_subset"):
        tcfg["train"]["train_subset"] = cfg["trial_train_subset"]
    tcfg["train"]["checkpoint_every_minutes"] = 1e9          # trial runs only save at epoch ends
    tcfg["train"]["sample_every_epochs"] = 10 ** 6           # no confusion PNGs during trials
    tcfg["run_id"] = make_run_id(cfg["device"], f"{cfg['study']}-trial{trial_number}",
                                  smoke=bool(cfg["run"].get("smoke")))
    tcfg["run"] = dict(cfg["run"], desc=f"{cfg['study']}-trial{trial_number}")
    # Trial runs go to <output_root>/trials/runs/..., so resume="auto" of the final training
    # (which looks in <output_root>/runs/task2_classifier) never picks up a trial checkpoint.
    tcfg["output_root"] = str(resolve(cfg["output_root"]) / "trials")
    return tcfg


def _check_search_space(space: list) -> None:
    """Fail before the first trial if a tuned parameter would break training."""
    for spec in space:
        name = spec["name"]
        if name not in PARAM_TARGETS:
            raise ValueError(f"unknown tuned parameter {name!r}; known: {sorted(PARAM_TARGETS)}")
        if name == "batch":
            bad = [b for b in spec["choices"] if int(b) % NUM_CLASSES != 0]
            if bad:
                raise ValueError(f"batch choices must be multiples of {NUM_CLASSES}, got {bad}")
        if name == "conv_channels":
            bad = [c for c in spec["choices"] if c not in CHANNEL_CHOICES]
            if bad:
                raise ValueError(f"unknown conv_channels names {bad}; known: {sorted(CHANNEL_CHOICES)}")


def study_classifier(cfg: dict, on_checkpoint=None) -> Path:
    """Run (or continue) the Optuna study t2_classifier; returns the folder holding trials.csv and plots.

    Objective: the BEST validation macro-F1 of a trial (maximise). MedianPruner stops unpromising
    trials after an epoch (useful when a trial has more than one epoch). SQLite storage with
    load_if_exists=True, so a restarted session continues the same study; after every trial the DB is
    copied to the persistent folder and on_checkpoint(db_path) is called.
    """
    cfg = deep_copy(cfg)
    n_trials, epochs = cfg["n_trials"], cfg["epochs_per_trial"]
    if is_tbd(n_trials) or is_tbd(epochs):
        raise ValueError("n_trials / epochs_per_trial are TBD_AFTER_BENCHMARK: measure first "
                         "(scripts/benchmark.py) or pass explicit values (--n-trials / --epochs)")
    _check_search_space(cfg["tuned_params"])

    db_path = resolve(cfg["storage"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # macro-F1: higher is better. open_study adds the Task 1 fixes: sampler seed + number of existing
    # trials, and trials left RUNNING by a killed process are marked FAIL (runs.py, D42).
    study = runs.open_study(cfg, "maximize", f"sqlite:///{db_path.as_posix()}")

    def objective(trial: optuna.Trial) -> float:
        values = suggest_params(trial, cfg["tuned_params"])
        tcfg = make_trial_config(cfg, values, trial.number)
        trial.set_user_attr("run_id", tcfg["run_id"])

        def report(epoch: int, val_f1: float) -> None:       # called after each finished epoch
            trial.report(val_f1, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        _run_dir, best_f1 = _train(tcfg, report_fn=report)
        return best_f1

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
    export_study(study, out_dir)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"study {cfg['study']}: {len(done)} complete / {len(study.trials)} total trials; exported to {out_dir}")
    if done:
        print(f"best macro-F1 {study.best_value:.5f} with {study.best_params}")
    return out_dir
