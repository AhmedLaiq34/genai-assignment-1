"""Task 4 training: style-conditioned face-to-sketch cGAN (pix2pix style).
Implements CONTRACTS 3.6 (val L1 on [0,1] sketches), 3.8 (checkpoints), 3.12 (tracking) and plan 10.1/10.3.
Plan: docs/TASK4_PLAN.md C9 to C11, C13 to C18, C20.

Entry point:  run_training(cfg, resume=None, on_checkpoint=None) -> run directory (Path)

What one run does
-----------------
* data: FS2KDataset train split (paired augmentation) and the validation split. The test split is never
  opened here (`final_test` is never passed).
* per batch: ONE discriminator step, then ONE generator step (the same fake sketch is used by both).
    L_D = 0.5 * (BCE(D(x,y,s), 1) + BCE(D(x,G(x,s).detach(),s), 0))
    L_G = BCE(D(x,G(x,s),s), 1) + lambda_l1 * mean|y - G(x,s)|            (sketches in [-1,1])
* score: val L1 of the generator on sketches rescaled to [0,1]  (SSIM and PSNR are logged as well)
* files in artifacts/runs/task4/<run_id>/: config.yaml, ckpt_last.pt, ckpt_best.pt, metrics.jsonl, samples/
  (optional ckpt_epochNNN.pt snapshots). Optuna trials use artifacts/runs/task4_trials/ instead (config
  key run.trial: true), so resume="auto" can never pick up a trial.
* ckpt_last.pt is written atomically every `checkpoint_every_minutes` and at the end of every epoch.
* resume: both networks, both optimisers, both schedulers, both scalers, epoch, step in the epoch, global step,
  best val L1, the fixed sample photos and all RNG states are restored.
"""
from __future__ import annotations

import copy
import json
import math
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from genai.common import tracking
from genai.common.checkpoint import (build_checkpoint, capture_rng_state, load_checkpoint,
                                     restore_rng_state, save_checkpoint, sha256_file)
from genai.common.metrics import l1, psnr, ssim
from genai.common.seed import seed_everything, worker_init_fn
from genai.fs2k import dataset as fs2k_dataset      # module import: FS2KDataset, resolve_fs2k_paths
from genai.models.cgan import Discriminator, Generator, describe
from genai.tasks.task4.config import find_latest_checkpoint, is_tbd, make_run_id, runs_dir

NUM_STYLES = 3
SAMPLES_PER_STYLE = (3, 3, 2)      # 8 fixed val photos: 3 of style 0, 3 of style 1, 2 of style 2 (plan C18)
SUM_KEYS = ("d_real", "d_fake", "g_adv", "g_l1", "d_real_prob", "d_fake_prob")   # per-batch values averaged per epoch
_MUST_MATCH = ("lr_g", "lr_d", "batch_size", "lambda_l1", "amp", "schedule")


# ----------------------------------------------------------------------------- one training step
def _bce(logits: torch.Tensor, target: float) -> torch.Tensor:
    """BCEWithLogits against a constant target (1 = "real", 0 = "fake"), computed in float32."""
    logits = logits.float()
    return F.binary_cross_entropy_with_logits(logits, torch.full_like(logits, target))


def d_step(G, D, opt_d, batch, scaler, fake=None) -> dict:
    """One discriminator update. batch = (photo, sketch, style), all already on the device.

    fake: optional generator output for this batch (the training loop computes it once and shares it with
    g_step); if None it is generated here without gradients. Either way D never sends gradients into G.
    Returns detached 0-dim tensors: d_real, d_fake, d_loss and the mean sigmoid(D) on real / fake patches.
    """
    photo, sketch, style = batch
    with torch.autocast(device_type=photo.device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
        if fake is None:
            with torch.no_grad():
                fake = G(photo, style)
        real_logits = D(photo, sketch, style)
        fake_logits = D(photo, fake.detach(), style)          # detach: only D learns in this step
    d_real = _bce(real_logits, 1.0)                           # D should say "real" for the true sketch
    d_fake = _bce(fake_logits, 0.0)                           # ... and "fake" for G's sketch
    loss = 0.5 * (d_real + d_fake)                            # L_D (the 0.5 slows D down relative to G)
    opt_d.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(opt_d)
    scaler.update()
    return {"d_real": d_real.detach(), "d_fake": d_fake.detach(), "d_loss": loss.detach(),
            "d_real_prob": torch.sigmoid(real_logits.detach().float()).mean(),
            "d_fake_prob": torch.sigmoid(fake_logits.detach().float()).mean()}


def g_step(G, D, opt_g, batch, lambda_l1: float, scaler, fake=None) -> dict:
    """One generator update: fool D (adversarial term) and stay close to the real sketch (L1 term).

    L_G = BCE(D(x, G(x,s), s), 1) + lambda_l1 * mean|y - G(x,s)|.  Returns detached 0-dim tensors g_adv, g_l1, g_loss.
    D's parameters are frozen here (no weight gradients are wasted on them); D itself is not updated.
    """
    photo, sketch, style = batch
    D.requires_grad_(False)
    try:
        with torch.autocast(device_type=photo.device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
            if fake is None:
                fake = G(photo, style)
            fake_logits = D(photo, fake, style)               # D sees the fake WITH gradient path back into G
        g_adv = _bce(fake_logits, 1.0)                        # G wants D to answer "real"
        g_l1 = (fake.float() - sketch).abs().mean()           # L1 on [-1,1], the "reconstruction" term
        loss = g_adv + float(lambda_l1) * g_l1
        opt_g.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt_g)
        scaler.update()
    finally:
        D.requires_grad_(True)
    return {"g_adv": g_adv.detach(), "g_l1": g_l1.detach(), "g_loss": loss.detach()}


# -------------------------------------------------------------------------------------- validation
def _to01(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] -> [0,1] (the scale of all reported metrics, plan 3.6)."""
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


@torch.no_grad()
def validate(G, loader, device) -> dict:
    """Mean L1 / SSIM / PSNR of G(photo, true style) against the true sketch, on [0,1] sketches (1 channel).

    Returns {l1, ssim, psnr, l1_style0, l1_style1, l1_style2}; a style missing from the loader gives nan.
    The model's mode is restored afterwards.
    """
    was_training = G.training
    G.eval()
    l1s, ssims, psnrs, styles = [], [], [], []
    for photo, sketch, style, _pid in loader:
        photo, sketch, style = photo.to(device), sketch.to(device), style.to(device)
        fake = _to01(G(photo, style))
        real = _to01(sketch)
        l1s.append(l1(fake, real).cpu())
        ssims.append(ssim(fake, real).cpu())
        psnrs.append(psnr(fake, real).cpu())
        styles.append(style.cpu())
    G.train(was_training)
    l1s, ssims, psnrs, styles = torch.cat(l1s), torch.cat(ssims), torch.cat(psnrs), torch.cat(styles)
    out = {"l1": l1s.mean().item(), "ssim": ssims.mean().item(), "psnr": psnrs.mean().item()}
    for k in range(NUM_STYLES):
        mask = styles == k
        out[f"l1_style{k}"] = l1s[mask].mean().item() if mask.any() else float("nan")
    return out


# ------------------------------------------------------------------------------------ sample grid
def _scan_val(val_ds) -> tuple:
    """(pair ids, styles) of every item of the validation dataset, in dataset order."""
    pids, styles = [], []
    for i in range(len(val_ds)):
        _photo, _sketch, style, pid = val_ds[i]
        pids.append(pid)
        styles.append(int(style))
    return pids, styles


def _pick_fixed(pids: list, styles: list) -> list:
    """The first 3, 3 and 2 pair ids (sorted by id) of styles 0, 1 and 2."""
    picks = []
    for k, n in enumerate(SAMPLES_PER_STYLE):
        picks += sorted(p for p, s in zip(pids, styles) if s == k)[:n]
    return picks


def fixed_sample_ids(val_ds) -> list:
    """Pair ids of the 8 fixed validation photos shown in every sample grid (plan C18).

    If a style has fewer validation pairs than requested, all of its pairs are used."""
    return _pick_fixed(*_scan_val(val_ds))


def _fetch_items(val_ds, ids: list, pids: list | None = None) -> list:
    """The dataset items (photo, sketch, style, pair_id) for the given pair ids."""
    if pids is None:
        pids, _ = _scan_val(val_ds)
    return [val_ds[pids.index(i)] for i in ids]


@torch.no_grad()
def save_sample_grid(G, val_ds, ids, device, path, items=None) -> torch.Tensor:
    """PNG grid, one row per fixed photo: photo | real sketch | G(x,0) | G(x,1) | G(x,2).

    items: optional pre-loaded dataset items for `ids` (saves re-reading the images at every sample step).
    Returns the grid tensor in [0,1] (for tracking.log_images)."""
    was_training = G.training
    G.eval()
    if items is None:
        items = _fetch_items(val_ds, ids)
    photo = torch.stack([it[0] for it in items]).to(device)
    sketch = torch.stack([it[1] for it in items]).to(device)
    tiles = []                                                    # row-major: 5 tiles per photo
    fakes = [G(photo, torch.full((len(items),), k, dtype=torch.long, device=device)) for k in range(NUM_STYLES)]
    for i in range(len(items)):
        tiles += [_to01(photo[i]), _to01(sketch[i]).expand(3, -1, -1)]
        tiles += [_to01(f[i]).expand(3, -1, -1) for f in fakes]
    grid = make_grid(torch.stack(tiles).cpu(), nrow=2 + NUM_STYLES, padding=2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, path)
    G.train(was_training)
    return grid


# ------------------------------------------------------------------------------------ schedules
def make_scheduler(optimizer, schedule: str, total_steps: int):
    """Learning-rate factor per optimiser step (plan C10).

    constant:          factor 1 always.
    linear_decay_half: factor 1 for the first half of the steps, then a straight line down to 0 at the end.
    """
    if schedule == "constant":
        def factor(step):
            return 1.0
    elif schedule == "linear_decay_half":
        half = total_steps / 2

        def factor(step):
            return 1.0 if step < half else max(0.0, (total_steps - step) / half)
    else:
        raise ValueError(f"train.schedule must be 'constant' or 'linear_decay_half', not {schedule!r}")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# ------------------------------------------------------------------------------------------ data
class _EpochBatchSampler:
    """Batches of one epoch: a shuffle that depends only on (seed, epoch), so a mid-epoch resume can
    continue the same epoch by skipping the batches that were already trained (`skip`). Partial last batch dropped."""

    def __init__(self, n: int, batch_size: int, seed: int):
        self.n, self.batch_size, self.seed = n, batch_size, seed
        self.steps = n // batch_size
        self.epoch, self.skip = 0, 0

    def __len__(self):
        return self.steps - self.skip

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(self.n, generator=g).tolist()
        batches = [perm[i * self.batch_size:(i + 1) * self.batch_size] for i in range(self.steps)]
        return iter(batches[self.skip:])


def _check_item(item, num_styles: int) -> None:
    """Fail early (with a clear message) if the dataset does not follow the FS2K tensor contract."""
    photo, sketch, style, _pid = item
    if tuple(photo.shape) != (3, 128, 128) or tuple(sketch.shape) != (1, 128, 128):
        raise ValueError(f"dataset item shapes {tuple(photo.shape)} / {tuple(sketch.shape)}, expected (3,128,128) / (1,128,128)")
    if not (0 <= int(style) < num_styles):
        raise ValueError(f"dataset style {int(style)} is outside 0..{num_styles - 1}")


# ------------------------------------------------------------------------------ resume helpers
def _check_resume_compatible(saved: dict, cfg: dict) -> None:
    """A resumed run must keep the same networks and optimisation settings, otherwise the
    restored optimiser / scheduler states would not belong to the run being continued."""
    if saved["model"] != cfg["model"]:
        raise ValueError(f"resume: model config differs: ckpt {saved['model']} vs now {cfg['model']}")
    for key in _MUST_MATCH:
        a, b = saved["train"].get(key), cfg["train"].get(key)
        if key == "schedule":                                  # default when the key is missing
            a, b = a or "constant", b or "constant"
        if a != b:
            raise ValueError(f"resume: train.{key} differs (ckpt {a!r} vs now {b!r}); use the same config to resume")


def _adopt_run_dir(ckpt_path: Path, run_dir: Path) -> None:
    """If the checkpoint lives somewhere else (a previous Kaggle output, Google Drive), copy the run files
    into the local run directory so training continues there."""
    src_dir = ckpt_path.parent
    if src_dir.resolve() == run_dir.resolve():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    names = [p.name for p in src_dir.glob("ckpt_*.pt")] + ["metrics.jsonl", "config.yaml"]
    for name in names:
        if (src_dir / name).exists():
            shutil.copyfile(src_dir / name, run_dir / name)
    if (src_dir / "samples").is_dir():
        shutil.copytree(src_dir / "samples", run_dir / "samples", dirs_exist_ok=True)


def _unique_run_dir(base: Path, run_id: str) -> tuple:
    """Run ids only have minute resolution; add -2, -3 ... if the folder already exists."""
    rid, n = run_id, 1
    while (base / rid).exists():
        n += 1
        rid = f"{run_id}-{n}"
    return rid, base / rid


def _finite_or_none(x):
    """JSON has no nan: a missing value (a style absent from a tiny val subset) is written as null."""
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else x


# ----------------------------------------------------------------------------------- split files
def split_checkpoint(ckpt_path, out_dir) -> tuple:
    """Split a run checkpoint into the two promotable files (plan C14, CONTRACTS 3.8).

    out_dir/t4_generator.pt      model = generator weights
    out_dir/t4_discriminator.pt  model = discriminator weights (training only, never exported)
    Both keep the metadata (config incl. model {base_channels, style_dim, dropout, num_styles}, epoch, seed,
    hashes, ...) so each rebuilds its network with <Class>.from_config(ckpt["config"]["model"]). The optimiser,
    scheduler, scaler and RNG entries are set to None / dropped. Returns (generator_path, discriminator_path).
    """
    ckpt = load_checkpoint(ckpt_path)
    out_dir = Path(out_dir)
    keep = ("epoch", "global_step", "best_metric", "config", "seed", "split_sha256", "manifest_sha256",
            "git_commit", "torch_version", "created_utc")
    meta = {k: ckpt.get(k) for k in keep}
    stripped = {"optimizer": None, "scheduler": None, "scaler": None}
    g_state = {"model": ckpt["model"], **stripped, **meta, "role": "generator"}
    d_state = {"model": ckpt["discriminator"], **stripped, **meta, "role": "discriminator"}
    g_path, d_path = out_dir / "t4_generator.pt", out_dir / "t4_discriminator.pt"
    save_checkpoint(g_path, g_state)
    save_checkpoint(d_path, d_state)
    return g_path, d_path


# ------------------------------------------------------------------------------------------ main
def _make_loaders(cfg: dict, dataset_factory):
    """Datasets and loaders. Returns (train_ds, sampler, train_loader, val_ds, val_loader)."""
    tc = cfg["train"]
    factory = dataset_factory or fs2k_dataset.FS2KDataset
    # final_test is NEVER passed: this module cannot open the official test split
    train_ds = factory(split="train", data_root=cfg["data_root"], augment=bool(tc.get("augment", True)),
                       subset=tc.get("train_subset"))
    val_ds = factory(split="val", data_root=cfg["data_root"], augment=False, subset=tc.get("val_subset"))
    batch_size = int(tc["batch_size"])
    if len(train_ds) < batch_size:
        raise ValueError(f"train set has {len(train_ds)} pairs, fewer than batch_size {batch_size}")
    if len(val_ds) == 0:
        raise ValueError("validation set is empty")
    _check_item(train_ds[0], int(cfg["model"].get("num_styles", NUM_STYLES)))
    _check_item(val_ds[0], int(cfg["model"].get("num_styles", NUM_STYLES)))

    sampler = _EpochBatchSampler(len(train_ds), batch_size, cfg["seed"])
    nw = int(cfg.get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=nw, worker_init_fn=worker_init_fn,
                              persistent_workers=nw > 0, pin_memory=torch.cuda.is_available())
    vw = int(tc.get("val_workers", 0))      # 0 = load val data in the main process (each worker costs RAM)
    val_loader = DataLoader(val_ds, batch_size=int(tc.get("val_batch_size", 32)), shuffle=False,
                            num_workers=vw, persistent_workers=vw > 0)
    return train_ds, sampler, train_loader, val_ds, val_loader


def _train(cfg: dict, resume=None, on_checkpoint=None, report_fn=None, dataset_factory=None):
    """Does the work of run_training; also returns the best val L1 (used by the Optuna study).

    report_fn(epoch, val_l1): optional callback after each COMPLETED epoch's validation
    (Optuna uses it to report and prune; it may raise an exception to stop the run).
    dataset_factory: optional replacement for FS2KDataset, called as factory(split=, data_root=, augment=,
    subset=) (tests use it with a tiny in-memory dataset).
    """
    cfg = copy.deepcopy(cfg)
    tc = cfg["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(tc.get("amp")) and device.type == "cuda"
    torch.backends.cudnn.benchmark = True   # faster convs for fixed sizes; tiny non-determinism (CONTRACTS 3.11)
    kind = "task4_trials" if cfg["run"].get("trial") else "task4"

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
        run_dir = runs_dir(cfg, kind=kind) / cfg["run_id"]
        _adopt_run_dir(Path(ckpt_path), run_dir)
    else:
        base = runs_dir(cfg, kind=kind)
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
    train_ds, sampler, train_loader, val_ds, val_loader = _make_loaders(cfg, dataset_factory)
    split_sha = sha256_file(fs2k_dataset.resolve_fs2k_paths(cfg["data_root"])["split_file"])
    if state is not None and state["split_sha256"] != split_sha:
        raise ValueError("resume: the split file differs from the checkpoint's (different data)")
    val_ids, val_styles = _scan_val(val_ds)
    sample_ids = list(state["fixed_sample_ids"]) if state is not None else _pick_fixed(val_ids, val_styles)
    sample_items = _fetch_items(val_ds, sample_ids, val_ids)       # loaded once, reused at every sample step

    # ---- schedule length (epochs may be TBD; then max_steps is required) ----
    steps_per_epoch = sampler.steps
    max_steps = tc.get("max_steps")
    if is_tbd(tc["epochs"]):
        if not max_steps:
            raise ValueError("train.epochs is TBD_AFTER_BENCHMARK: set a real epoch count or train.max_steps")
        epochs = math.ceil(int(max_steps) / steps_per_epoch)
    else:
        epochs = int(tc["epochs"])
    total_steps = min(epochs * steps_per_epoch, int(max_steps)) if max_steps else epochs * steps_per_epoch

    # ---- networks, optimisers, schedulers, scalers (one of each for G and for D) ----
    G = Generator.from_config(cfg["model"]).to(device)
    D = Discriminator.from_config(cfg["model"]).to(device)
    print(describe(G, D))
    betas = tuple(float(b) for b in tc.get("betas", (0.5, 0.999)))
    opt_g = torch.optim.Adam(G.parameters(), lr=float(tc["lr_g"]), betas=betas)
    opt_d = torch.optim.Adam(D.parameters(), lr=float(tc["lr_d"]), betas=betas)
    schedule = tc.get("schedule", "constant")
    sched_g = make_scheduler(opt_g, schedule, total_steps)
    sched_d = make_scheduler(opt_d, schedule, total_steps)
    scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp)
    lambda_l1 = float(tc["lambda_l1"])

    epoch, step_in_epoch, global_step, best_l1 = 0, 0, 0, None
    sums = {k: torch.zeros((), device=device) for k in SUM_KEYS}      # running sums over the current epoch
    if state is not None:
        G.load_state_dict(state["model"])
        D.load_state_dict(state["discriminator"])
        opt_g.load_state_dict(state["optimizer"])
        opt_d.load_state_dict(state["optimizer_d"])
        sched_g.load_state_dict(state["scheduler"])
        sched_d.load_state_dict(state["scheduler_d"])
        scaler_g.load_state_dict(state["scaler"])
        scaler_d.load_state_dict(state["scaler_d"])
        epoch, step_in_epoch = state["epoch"], state["step_in_epoch"]
        global_step, best_l1 = state["global_step"], state["best_metric"]
        for k in SUM_KEYS:
            sums[k].fill_(state["epoch_sums"][k])
        restore_rng_state(state["rng"])          # python / numpy / torch / cuda generators
        print(f"resumed from {ckpt_path}: epoch {epoch}, step_in_epoch {step_in_epoch}, global_step {global_step}")

    def save(name: str) -> Path:
        """Write ckpt_<name>.pt atomically and tell the caller (e.g. copy to Drive)."""
        ckpt = build_checkpoint(G, opt_g, sched_g, scaler_g, epoch=epoch, global_step=global_step,
                                best_metric=best_l1, config=cfg, seed=cfg["seed"],
                                split_sha256=split_sha, manifest_sha256=None)    # FS2K has no manifest
        ckpt.update(discriminator=D.state_dict(), optimizer_d=opt_d.state_dict(),
                    scheduler_d=sched_d.state_dict(), scaler_d=scaler_d.state_dict(),
                    rng=capture_rng_state(), step_in_epoch=step_in_epoch,
                    epoch_sums={k: float(v) for k, v in sums.items()}, fixed_sample_ids=sample_ids)
        path = run_dir / f"ckpt_{name}.pt"
        save_checkpoint(path, ckpt)
        if on_checkpoint is not None:
            on_checkpoint(path)
        return path

    tracking.init_run(cfg, cfg["run_id"], project=cfg["run"].get("project", "genai-a1"), group="task4")
    ckpt_every = float(tc.get("checkpoint_every_minutes", 10)) * 60
    log_every = int(tc.get("log_every_steps", 50))
    sample_every = int(tc.get("sample_every_epochs") or 0)       # 0 / null = no sample grids (Optuna trials)
    snapshot_every = int(tc.get("snapshot_every_epochs") or 0)   # 0 / null = no ckpt_epochNNN.pt files
    pause_after = tc.get("pause_after_steps")
    last_ckpt_time, paused = time.monotonic(), False

    try:
        G.train()
        D.train()
        while epoch < epochs and global_step < total_steps and not paused:
            sampler.epoch = epoch                    # fixes the batch order of this epoch
            sampler.skip = step_in_epoch             # already-trained batches (mid-epoch resume)
            t_epoch = time.monotonic()
            for photo, sketch, style, _pid in train_loader:
                batch = (photo.to(device, non_blocking=True), sketch.to(device, non_blocking=True),
                         style.to(device, non_blocking=True))
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    fake = G(batch[0], batch[2])     # one generator forward, shared by the D step and the G step
                d_out = d_step(G, D, opt_d, batch, scaler_d, fake=fake)
                g_out = g_step(G, D, opt_g, batch, lambda_l1, scaler_g, fake=fake)
                sched_g.step()
                sched_d.step()
                out = {**d_out, **g_out}
                for k in SUM_KEYS:
                    sums[k] += out[k]
                global_step += 1
                step_in_epoch += 1
                if global_step % log_every == 0:
                    tracking.log({f"step/{k}": float(out[k]) for k in SUM_KEYS}, global_step)
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
            res = validate(G, val_loader, device)
            val_l1 = res["l1"]
            n_steps = max(step_in_epoch, 1)
            row = {"epoch": epoch + 1, "global_step": global_step, "partial_epoch": not epoch_done,
                   "lr_g": opt_g.param_groups[0]["lr"], "lr_d": opt_d.param_groups[0]["lr"]}
            row.update({f"train/{k}": float(sums[k]) / n_steps for k in SUM_KEYS})
            row.update({f"val/{k}": v for k, v in res.items()})
            row["epoch_seconds"] = time.monotonic() - t_epoch
            with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({k: _finite_or_none(v) for k, v in row.items()}) + "\n")
            tracking.log({k: v for k, v in row.items()
                          if k.startswith(("train/", "val/")) and _finite_or_none(v) is not None}, global_step)
            print(f"epoch {epoch + 1} step {global_step}: d_real {row['train/d_real']:.3f}  d_fake {row['train/d_fake']:.3f}  "
                  f"g_adv {row['train/g_adv']:.3f}  g_l1 {row['train/g_l1']:.4f}  | val l1 {val_l1:.4f}  "
                  f"ssim {res['ssim']:.3f}  psnr {res['psnr']:.2f}")

            this_epoch = epoch + 1                   # 1-based number of the epoch just finished
            if sample_every and (this_epoch == 1 or this_epoch % sample_every == 0 or global_step >= total_steps):
                grid = save_sample_grid(G, val_ds, sample_ids, device,
                                        run_dir / "samples" / f"epoch{this_epoch:03d}.png", items=sample_items)
                tracking.log_images("val/samples (photo, real, style 0, 1, 2)", grid, global_step)

            if epoch_done:                           # advance the counters BEFORE saving
                epoch += 1
                step_in_epoch = 0
                for k in SUM_KEYS:
                    sums[k].zero_()
            if best_l1 is None or val_l1 < best_l1:
                best_l1 = val_l1
                save("best")
            save("last")
            if epoch_done and snapshot_every and epoch % snapshot_every == 0:
                save(f"epoch{epoch:03d}")
            last_ckpt_time = time.monotonic()
            if report_fn is not None and epoch_done:
                report_fn(epoch, val_l1)             # Optuna: may raise TrialPruned
            if global_step >= total_steps:
                break
    finally:
        tracking.finish()
    return run_dir, best_l1


def run_training(cfg: dict, resume: str | None = None, on_checkpoint=None) -> Path:
    """Train Task 4 (see module docstring). Returns the run directory."""
    run_dir, _best = _train(cfg, resume=resume, on_checkpoint=on_checkpoint)
    return run_dir
