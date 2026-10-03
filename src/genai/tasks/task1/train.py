"""Task 1 training: universal denoising autoencoder. Implements CONTRACTS 3.6, 3.8, 3.12 and plan 10.1/10.3.

Entry point:  run_training(cfg, resume=None, on_checkpoint=None) -> run directory (Path)

What one run does
-----------------
* data: PetsTrainDataset with policy "iid_uniform" (a NEW random corruption on every load),
  validation on the fixed VAL MANIFEST (never the test set: this file never opens it).
* loss:  alpha * L1 + (1 - alpha) * (1 - SSIM)        (PDF page 4)
* score: fixed objective J = 0.5*L1 + 0.5*(1-SSIM) on the val manifest (independent of alpha)
* files in artifacts/runs/task1/<run_id>/: config.yaml, ckpt_last.pt, ckpt_best.pt, metrics.jsonl, samples/
* ckpt_last.pt is written atomically every `checkpoint_every_minutes` and at the end of every epoch.
* resume: model, optimizer, scheduler, AMP scaler, epoch, global step, best J and all RNG states
  are restored (resume="auto" finds the newest ckpt_last.pt, resume=<path> uses that file).
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import shutil
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid, save_image

from genai.common import tracking
from genai.common.checkpoint import (build_checkpoint, capture_rng_state, load_checkpoint,
                                     restore_rng_state, save_checkpoint)
from genai.common.metrics import evaluate_restoration, l1, ssim
from genai.common.seed import seed_everything, worker_init_fn
from genai.models.autoencoder import UniversalAE, describe
from genai.pets import samplers
from genai.pets.dataset import PetsManifestDataset, PetsTrainDataset, resolve_data_paths
from genai.pets.split import split_sha256
from genai.tasks.task1.config import find_latest_checkpoint, is_tbd, make_run_id, runs_dir

VAL_MANIFEST_NAME = "pets_val_manifest.jsonl"
N_SAMPLE_IMAGES = 12          # the same 12 fixed val images are shown at every sample step


# ------------------------------------------------------------------------------------ loss
def restoration_loss(restored: torch.Tensor, clean: torch.Tensor, alpha: float) -> torch.Tensor:
    """L_UDA = alpha * L1 + (1 - alpha) * (1 - SSIM), averaged over the batch."""
    return alpha * l1(restored, clean).mean() + (1.0 - alpha) * (1.0 - ssim(restored, clean).mean())


# ------------------------------------------------------------------------------------ data
class _SkipFirst:
    """Batch-sampler wrapper: skip the first `skip` batches (used once after a mid-epoch resume).

    The inner sampler's order depends only on (seed, epoch), so skipping the batches that were
    already trained before the interruption continues the same epoch without repeating data.
    """

    def __init__(self, inner):
        self.inner, self.skip = inner, 0

    def __len__(self):
        return len(self.inner) - self.skip

    def __iter__(self):
        return itertools.islice(iter(self.inner), self.skip, None)


def build_train_loader(cfg: dict, paths: dict, batch_size: int):
    """Train dataset + loader. Returns (dataset, sampler_wrapper, loader)."""
    tc = cfg["train"]
    ds = PetsTrainDataset("train", "iid_uniform", data_root=paths, seed=cfg["seed"])
    if tc.get("train_subset"):                 # smoke / dry runs: keep only the first N images
        ds.ids = ds.ids[: int(tc["train_subset"])]
    sampler = _SkipFirst(samplers.make_batch_sampler("iid_uniform", len(ds), batch_size, cfg["seed"]))
    nw = int(cfg.get("num_workers", 0))
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=nw, worker_init_fn=worker_init_fn,
                        persistent_workers=nw > 0, pin_memory=torch.cuda.is_available())
    return ds, sampler, loader


def build_val_loader(cfg: dict, paths: dict):
    """Val manifest dataset + loader. Returns (dataset, loader, manifest_sha256)."""
    tc = cfg["train"]
    manifest = Path(paths["manifests"]) / VAL_MANIFEST_NAME
    ds = PetsManifestDataset(manifest, split="val", final_test=False, data_root=paths)
    n = int(tc["val_subset"]) if tc.get("val_subset") else len(ds)
    data = Subset(ds, range(n)) if n < len(ds) else ds
    nw = int(tc.get("val_workers", 0))   # 0 = load val data in the main process (saves RAM: every worker re-imports torch)
    loader = DataLoader(data, batch_size=int(tc.get("val_batch_size", 128)), shuffle=False,
                        num_workers=nw, persistent_workers=nw > 0)
    manifest_sha = Path(str(manifest) + ".sha256").read_text(encoding="utf-8").strip()
    return ds, n, loader, manifest_sha


def fixed_sample_indices(ds: PetsManifestDataset, limit: int, n: int = N_SAMPLE_IMAGES) -> list:
    """n/4 manifest rows of each condition (the first ones), among the first `limit` rows.
    Same rows every time, so the sample grids of different epochs are comparable."""
    per_cond = max(1, n // 4)
    picks = []
    for cond in range(4):
        rows = [i for i in range(limit) if ds.rows[i]["cond_id"] == cond]
        picks += rows[:per_cond]
    return picks


@torch.no_grad()
def save_sample_grid(model, ds, indices, device, path: Path):
    """PNG with 4 rows: clean, corrupted, restored, |error| (x4 gain so it is visible).
    Returns the image grid tensor (for the tracker)."""
    was_training = model.training
    model.eval()
    items = [ds[i] for i in indices]
    corrupted = torch.stack([it[0] for it in items]).to(device)
    clean = torch.stack([it[1] for it in items]).to(device)
    restored = model(corrupted)
    error = ((restored - clean).abs() * 4).clamp(0, 1)
    rows = torch.cat([clean, corrupted, restored, error]).cpu()
    grid = make_grid(rows, nrow=len(indices), padding=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, path)
    model.train(was_training)
    return grid


# ------------------------------------------------------------------------------ resume helpers
_MUST_MATCH = ("lr", "batch_size", "alpha", "weight_decay", "scheduler", "amp")


def _check_resume_compatible(saved: dict, cfg: dict) -> None:
    """A resumed run must keep the same model and optimisation settings, otherwise the
    restored optimizer / scheduler state would not belong to the run being continued."""
    if saved["model"] != cfg["model"]:
        raise ValueError(f"resume: model config differs: ckpt {saved['model']} vs now {cfg['model']}")
    for key in _MUST_MATCH:
        if saved["train"].get(key) != cfg["train"].get(key):
            raise ValueError(f"resume: train.{key} differs (ckpt {saved['train'].get(key)!r} vs "
                             f"now {cfg['train'].get(key)!r}); use the same config to resume")


def _adopt_run_dir(ckpt_path: Path, run_dir: Path) -> None:
    """If the checkpoint lives somewhere else (e.g. Google Drive), copy the run files into the
    local run directory so training continues there."""
    if ckpt_path.parent.resolve() == run_dir.resolve():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("ckpt_last.pt", "ckpt_best.pt", "metrics.jsonl", "config.yaml"):
        src = ckpt_path.parent / name
        if src.exists():
            shutil.copyfile(src, run_dir / name)


def _unique_run_dir(base: Path, run_id: str) -> tuple:
    """Run ids only have minute resolution; add -2, -3 ... if the folder already exists."""
    rid, n = run_id, 1
    while (base / rid).exists():
        n += 1
        rid = f"{run_id}-{n}"
    return rid, base / rid


# ------------------------------------------------------------------------------------- main
def _train(cfg: dict, resume=None, on_checkpoint=None, report_fn=None):
    """Does the work of run_training; also returns the best J (used by the Optuna study).

    report_fn(epoch, val_J): optional callback after each COMPLETED epoch's validation
    (Optuna uses it to report and prune; it may raise an exception to stop the run).
    """
    cfg = copy.deepcopy(cfg)
    tc = cfg["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(tc.get("amp")) and device.type == "cuda"
    torch.backends.cudnn.benchmark = True   # faster convs for fixed sizes; tiny non-determinism (CONTRACTS 3.11)

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
            rid, run_dir = _unique_run_dir(base, make_run_id(cfg["device"], cfg["run"]["desc"], smoke))
        cfg["run_id"] = rid
        run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "samples").mkdir(exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"run dir: {run_dir}")

    seed_everything(cfg["seed"])

    # ---- data ----
    paths = resolve_data_paths(cfg)
    batch_size = int(tc["batch_size"])
    train_ds, sampler, train_loader = build_train_loader(cfg, paths, batch_size)
    val_ds, n_val, val_loader, manifest_sha = build_val_loader(cfg, paths)
    split_sha = split_sha256(train_ds.split)
    if state is not None and (state["split_sha256"] != split_sha or state["manifest_sha256"] != manifest_sha):
        raise ValueError("resume: split / manifest hashes differ from the checkpoint (different data)")
    sample_idx = fixed_sample_indices(val_ds, n_val)

    # ---- schedule length (epochs may be TBD; then max_steps is required) ----
    steps_per_epoch = len(sampler.inner)
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

    tracking.init_run(cfg, cfg["run_id"], project=cfg["run"].get("project", "genai-a1"), group="task1")
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
            for corrupted, clean, _cond, _sev in train_loader:
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
            res = evaluate_restoration(model, val_loader, device)
            val_J = res["overall"]["J"]
            row = {
                "epoch": epoch + 1, "global_step": global_step, "partial_epoch": not epoch_done,
                "train_loss": loss_sum / max(step_in_epoch, 1), "lr": optimizer.param_groups[0]["lr"],
                "val_J": val_J, "val_l1": res["overall"]["l1"], "val_ssim": res["overall"]["ssim"],
                "val_psnr": res["overall"]["psnr"],
                "per_condition": {k: {m: v[m] for m in ("J", "l1", "ssim", "psnr")}
                                  for k, v in res.items() if k not in ("overall", "by_severity")},
                "by_severity": {k: v["J"] for k, v in res["by_severity"].items()},
                "epoch_seconds": time.monotonic() - t_epoch,
            }
            with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            flat = {"val/J": val_J, "val/l1": row["val_l1"], "val/ssim": row["val_ssim"],
                    "val/psnr": row["val_psnr"], "train/epoch_loss": row["train_loss"]}
            flat.update({f"val/J_{k}": v["J"] for k, v in row["per_condition"].items()})
            tracking.log(flat, global_step)
            print(f"epoch {epoch + 1} step {global_step}: train_loss {row['train_loss']:.4f}  val_J {val_J:.4f}  "
                  f"ssim {row['val_ssim']:.3f}  psnr {row['val_psnr']:.2f}")

            if (epoch + 1) % sample_every == 0 or global_step >= total_steps:
                grid = save_sample_grid(model, val_ds, sample_idx, device,
                                        run_dir / "samples" / f"epoch{epoch + 1:03d}_step{global_step}.png")
                tracking.log_images("val/samples (clean, input, restored, error)", grid, global_step)

            if epoch_done:                           # advance the counters BEFORE saving
                epoch += 1
                step_in_epoch, loss_sum = 0, 0.0
            improved = best_J is None or val_J < best_J
            if improved:
                best_J = val_J
                save("best")
            save("last")
            last_ckpt_time = time.monotonic()
            if report_fn is not None and epoch_done:
                report_fn(epoch, val_J)              # Optuna: may raise TrialPruned
            if global_step >= total_steps:
                break
    finally:
        tracking.finish()
    return run_dir, best_J


def run_training(cfg: dict, resume: str | None = None, on_checkpoint=None) -> Path:
    """Train Task 1 (see module docstring). Returns the run directory."""
    run_dir, _best = _train(cfg, resume=resume, on_checkpoint=on_checkpoint)
    return run_dir
