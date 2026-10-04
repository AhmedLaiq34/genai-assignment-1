"""Task 3 training: the soft mixture of experts, warm-up then joint fine-tuning.

Implements the PDF's Task 3 training (pages 6 and 7) and CONTRACTS 3.5, 3.6, 3.8, 3.12; design in
docs/TASK3_PLAN.md (B2 to B10, B13 to B16, D3). The loop is modelled on task2/classifier.py::_train.

Entry point:  run_training(cfg, resume=None, on_checkpoint=None) -> run directory (Path)

What one run does
-----------------
* model: SoftMoE (gate + identity branch + three experts), built from the four Task 2 checkpoints.
  Their sha256 is checked first (verify_sources); the Task 2 files are only read, never written.
* data: PetsTrainDataset with policy "balanced_batch" (every batch has batch_size/4 images of each
  class, the label is the class the corruption pipeline applied); validation on the fixed VAL MANIFEST.
* two stages, ONE epoch counter over both (warm-up epochs first):
    warmup  experts frozen (no gradients, not updated), AdamW over the GATE only at the constant warmup_lr
    joint   ONE AdamW over gate + experts at joint_lr, cosine decay over the joint steps
* loss (float32): lambda_1 * L1 + lambda_s * (1 - SSIM)  +  lambda_c * CE(raw gate logits)  +  lambda_b * balance
  with lambda_1 = r and lambda_s = 1 - r (decision B8).
* best checkpoint = the lowest fixed J = 0.5*L1 + 0.5*(1-SSIM) among JOINT-stage validations that are
  not flagged as routing collapse (B14). Epoch 0 (the untrained copy of Task 2) is logged, never best.
* files in <output_root>/runs/task3/<run_id>/: config.yaml, ckpt_last.pt, ckpt_best.pt, metrics.jsonl, samples/
* resilience as in Task 2: atomic checkpoints, full resume (stage, optimizer, RNG, mid-epoch skip), on_checkpoint hook.

Also here (used by the study, the benchmark and the tests): moe_loss, balance_loss, validate,
collapse_flags, measure_step_times.

The test split is never opened by this file.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torchvision.utils import make_grid, save_image

from genai.common import tracking
from genai.common.checkpoint import (build_checkpoint, capture_rng_state, load_checkpoint,
                                     restore_rng_state, save_checkpoint)
from genai.common.constants import CLASS_NAMES, NUM_CLASSES
from genai.common.metrics import evaluate_restoration
from genai.common.seed import seed_everything
from genai.models.autoencoder import count_parameters
from genai.models.moe import SoftMoE, describe
from genai.pets.dataset import resolve_data_paths
from genai.pets.split import split_sha256
from genai.tasks.task1.config import is_tbd, make_run_id, resolve
from genai.tasks.task1.train import (_adopt_run_dir, _unique_run_dir, build_val_loader,
                                     fixed_sample_indices, restoration_loss)
from genai.tasks.task2.classifier import build_train_loader
from genai.tasks.task2.routing import describe_checkpoints
from genai.tasks.task3 import BRANCH_NAMES, RUN_GROUP, TASK2_SHA256, TRACKER_GROUP
from genai.tasks.task3.sources import find_sources, source_record, verify_sources

N_SAMPLE_ROWS = 8                       # the sample grid shows 8 fixed validation rows, 2 per class
LAMBDA_KEYS = ("lambda_1", "lambda_s", "lambda_c", "lambda_b")

# A resumed run must keep these settings, otherwise the restored optimizer / scheduler state would
# not belong to the run being continued (the Task 1 / Task 2 rule).
_MUST_MATCH = ("batch_size", "warmup_epochs", "joint_epochs", "warmup_lr", "joint_lr", "weight_decay",
               "lambda_1", "lambda_s", "lambda_c", "lambda_b", "scheduler", "amp")


# ============================================================================== run folders
def runs_dir(cfg: dict, root_key: str = "output_root") -> Path:
    """<output_root>/runs/task3  (root_key='persist_root' gives the persistent copy)."""
    return resolve(cfg[root_key]) / "runs" / RUN_GROUP


def find_latest_checkpoint(cfg: dict) -> Path | None:
    """Newest ckpt_last.pt of Task 3 (what resume='auto' uses); smoke runs only count for smoke configs."""
    candidates = []
    for key in ("persist_root", "output_root"):
        if key in cfg:
            candidates += list(runs_dir(cfg, key).glob("*/ckpt_last.pt"))
    candidates = [p for p in candidates if not p.parent.name.endswith("_smoke") or cfg.get("run", {}).get("smoke")]
    return max(candidates, key=lambda p: p.stat().st_mtime, default=None)


def expected_hashes(cfg: dict) -> dict:
    """sha256 values the source files must have: the real Task 2 hashes unless cfg['sources']['expected_sha256']
    says otherwise (the tests use fixture checkpoints and pass their own hashes)."""
    return (cfg.get("sources") or {}).get("expected_sha256") or TASK2_SHA256


# ============================================================================== loss
def balance_loss(w: torch.Tensor) -> torch.Tensor:
    """The PDF's balance term: sum_k (mean over the batch of w_k  -  1/4)^2.

    w is N x 4. On a balanced batch (the same number of images of every class) perfect routing gives
    mean weights of exactly 1/4 each, so the term is 0 and does not fight correct routing (plan B9).
    """
    return ((w.mean(dim=0) - 1.0 / w.shape[1]) ** 2).sum()


def moe_loss(x_hat, clean, logits, w, labels, lam: dict) -> tuple:
    """Total loss = restoration + lambda_c * CE + lambda_b * balance (PDF page 6). Everything in float32.

    x_hat, clean  N x 3 x 128 x 128 in [0,1]     the mixture and the target
    logits        N x 4                          the gate's RAW logits (CE does not see tau, decision B2)
    w             N x 4                          softmax(logits / tau), used by the balance term
    labels        N                              true class ids (0 clean, 1 salt, 2 blur, 3 occlusion)
    lam           {lambda_1, lambda_s, lambda_c, lambda_b}

    The restoration part is task1.train.restoration_loss(alpha=lambda_1), i.e. lambda_1 * L1 +
    (1 - lambda_1) * (1 - SSIM); decision B8 defines lambda_s = 1 - lambda_1, so the two must add up to 1.
    Returns (total, {"recon", "ce", "balance"}) where the dict holds plain floats for logging.
    """
    if abs(float(lam["lambda_1"]) + float(lam["lambda_s"]) - 1.0) > 1e-6:
        raise ValueError(f"lambda_1 + lambda_s must be 1 (decision B8: r and 1 - r), got "
                         f"{lam['lambda_1']} + {lam['lambda_s']}")
    recon = restoration_loss(x_hat.float(), clean, alpha=float(lam["lambda_1"]))
    ce = F.cross_entropy(logits.float(), labels)
    balance = balance_loss(w.float())
    total = recon + float(lam["lambda_c"]) * ce + float(lam["lambda_b"]) * balance
    return total, {"recon": recon.item(), "ce": ce.item(), "balance": balance.item()}


# ============================================================================== collapse flags + validation
def collapse_flags(mean_w, mean_w_by_class, cfg) -> dict:
    """Routing-collapse check of one validation (decision B16, CONTRACTS 3.6).

    mean_w           4 numbers: mean weight of every branch over all validation rows
    mean_w_by_class  4 x 4: row c = mean weights over the rows whose TRUE class is c
    cfg              the whole config or just its `collapse` section {min_mean_weight, max_foreign_weight}

    (a) a branch whose mean weight is below min_mean_weight is dead;
    (b) for true class c, a branch k != c whose mean weight on those rows is above max_foreign_weight
        takes over images that belong to another class.
    Returns {"collapsed": bool, "reasons": [text, ...]}.
    """
    section = cfg.get("collapse", cfg)
    min_w = float(section.get("min_mean_weight", 0.02))
    max_foreign = float(section.get("max_foreign_weight", 0.9))
    reasons = []
    for k, value in enumerate(mean_w):
        if value < min_w:
            reasons.append(f"branch {k} ({BRANCH_NAMES[k]}) has mean weight {value:.4f} < {min_w}")
    for c, row in enumerate(mean_w_by_class):
        for k, value in enumerate(row):
            if k != c and value > max_foreign:
                reasons.append(f"on true class {c} ({CLASS_NAMES[c]}) branch {k} ({BRANCH_NAMES[k]}) "
                               f"has mean weight {value:.4f} > {max_foreign}")
    return {"collapsed": bool(reasons), "reasons": reasons}


class _Recorder(nn.Module):
    """Wraps the SoftMoE so that evaluate_restoration (which only wants x_hat) also leaves us the
    weights and logits of every batch it evaluates."""

    def __init__(self, model: SoftMoE):
        super().__init__()
        self.model = model
        self.weights, self.logits = [], []

    def forward(self, x):
        x_hat, w, logits = self.model(x)
        self.weights.append(w.detach().cpu())
        self.logits.append(logits.detach().cpu())
        return x_hat


def _tap(loader, store: list):
    """Pass the batches of `loader` through unchanged and remember each batch's class ids."""
    for batch in loader:
        store.append(batch[2])
        yield batch


@torch.no_grad()
def validate(model: SoftMoE, loader, device, collapse_cfg=None) -> dict:
    """One pass over a validation loader (batches of corrupted, clean, cond_id, severity_id).

    Returns the numbers of task1's evaluate_restoration (J, l1, ssim, psnr, per condition, per severity)
    plus what Task 3 needs: mean_w (4), mean_w_by_class (4 x 4), gate_accuracy (argmax of the raw logits
    against the true class) and flags (collapse_flags with `collapse_cfg`, default thresholds 0.02 / 0.9).
    The model is evaluated in eval mode and put back into the mode it had.
    """
    was_training = model.training
    recorder, cond_batches = _Recorder(model), []
    res = evaluate_restoration(recorder, _tap(loader, cond_batches), device)   # sets model.eval() inside
    model.train(was_training)

    w = torch.cat(recorder.weights)                       # N x 4
    logits = torch.cat(recorder.logits)                   # N x 4
    cond = torch.cat(cond_batches).long()                 # N
    mean_w = w.mean(dim=0).tolist()
    mean_w_by_class = []
    for c in range(NUM_CLASSES):
        rows = cond == c
        mean_w_by_class.append(w[rows].mean(dim=0).tolist() if rows.any() else [0.0] * NUM_CLASSES)
    return {
        **res["overall"],                                 # l1, ssim, psnr, J, count
        "per_condition": {k: {m: v[m] for m in ("J", "l1", "ssim", "psnr")}
                          for k, v in res.items() if k not in ("overall", "by_severity")},
        "by_severity": {k: v["J"] for k, v in res["by_severity"].items()},
        "mean_w": mean_w,
        "mean_w_by_class": mean_w_by_class,
        "gate_accuracy": float((logits.argmax(dim=1) == cond).float().mean()),
        "flags": collapse_flags(mean_w, mean_w_by_class, collapse_cfg or {}),
    }


# ============================================================================== sample grid
@torch.no_grad()
def save_sample_grid(model: SoftMoE, ds, indices: list, device, path: Path):
    """PNG with 4 rows over the fixed validation rows: target, input, x_hat, |error| (x4 so it is visible)."""
    was_training = model.training
    model.eval()
    items = [ds[i] for i in indices]
    corrupted = torch.stack([it[0] for it in items]).to(device)
    clean = torch.stack([it[1] for it in items]).to(device)
    x_hat, _w, _logits = model(corrupted)
    error = ((x_hat - clean).abs() * 4).clamp(0, 1)
    grid = make_grid(torch.cat([clean, corrupted, x_hat, error]).cpu(), nrow=len(indices), padding=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, path)
    model.train(was_training)
    return grid


# ============================================================================== stages and one step
def stage_of(epoch: int, warmup_epochs: int) -> str:
    """Epoch numbers are counted from 0 over BOTH stages: the first warmup_epochs are warm-up."""
    return "warmup" if epoch < warmup_epochs else "joint"


def make_stage(model: SoftMoE, stage: str, tc: dict, joint_steps: int) -> tuple:
    """(optimizer, scheduler) of a stage; also freezes / unfreezes the experts (decision B6).

    warmup: experts frozen, AdamW over the gate's parameters only, constant warmup_lr, no scheduler.
    joint:  experts trainable, ONE AdamW over all parameters at joint_lr, cosine decay over joint_steps.
    """
    wd = float(tc["weight_decay"])
    if stage == "warmup":
        model.freeze_experts()
        optimizer = torch.optim.AdamW(model.gate.parameters(), lr=float(tc["warmup_lr"]), weight_decay=wd)
        return optimizer, None
    model.unfreeze_experts()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tc["joint_lr"]), weight_decay=wd)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(joint_steps), 1))
                 if tc.get("scheduler", "cosine") == "cosine" else None)
    return optimizer, scheduler


def _train_step(model, optimizer, scaler, corrupted, clean, labels, lam, use_amp, grad_clip) -> tuple:
    """One optimisation step. Returns (loss float, {"recon","ce","balance"}, mean weights of the batch (4)).

    The four networks run under fp16 autocast (GPU only); logits, branches, mixture and every loss
    term are float32 (decision B7), so moe_loss is called OUTSIDE the autocast block.
    """
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        x_hat, w, logits = model(corrupted)
    loss, parts = moe_loss(x_hat, clean, logits, w, labels, lam)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    if grad_clip:
        scaler.unscale_(optimizer)
        params = [p for group in optimizer.param_groups for p in group["params"]]
        torch.nn.utils.clip_grad_norm_(params, float(grad_clip))
    scaler.step(optimizer)
    scaler.update()
    return loss.item(), parts, w.detach().mean(dim=0)


def _check_resume_compatible(saved: dict, cfg: dict) -> None:
    """A checkpoint of another component, tau or optimisation setting cannot be continued."""
    if saved.get("component") != "soft_moe":
        raise ValueError(f"resume: the checkpoint is not a Task 3 checkpoint (component={saved.get('component')!r})")
    if float(saved["model"]["tau"]) != float(cfg["model"]["tau"]):
        raise ValueError(f"resume: model.tau differs (ckpt {saved['model']['tau']} vs now {cfg['model']['tau']})")
    for key in _MUST_MATCH:
        if saved["train"].get(key) != cfg["train"].get(key):
            raise ValueError(f"resume: train.{key} differs (ckpt {saved['train'].get(key)!r} vs "
                             f"now {cfg['train'].get(key)!r}); use the same config to resume")


def _val_columns(res: dict) -> dict:
    """The val_* columns of a metrics.jsonl row."""
    return {"val_J": res["J"], "val_l1": res["l1"], "val_ssim": res["ssim"], "val_psnr": res["psnr"],
            "val_mean_w": res["mean_w"], "val_mean_w_by_class": res["mean_w_by_class"],
            "val_gate_accuracy": res["gate_accuracy"],
            "collapsed": res["flags"]["collapsed"], "collapse_reasons": res["flags"]["reasons"]}


# ============================================================================== training
def _train(cfg: dict, resume=None, on_checkpoint=None, report_fn=None, sources: dict | None = None):
    """Does the work of run_training; returns (run_dir, best J). The Optuna study uses it too.

    sources:   {"classifier", "salt", "blur", "occlusion"[, "t1"]} -> Path of the Task 2 checkpoints;
               None = find_sources(cfg). Always verified (sha256) before anything is loaded.
    report_fn: report_fn(step, val_J, flags) after every validation of a COMPLETED epoch (step = epoch
               counter over warm-up + joint, from 1). flags = {"collapsed", "reasons", "mean_w",
               "mean_w_by_class"}. Optuna uses it to report and prune; it may raise an exception to stop the run.
    best J:    lowest val J among non-collapsed joint-stage validations (None if there was none).
    """
    cfg = copy.deepcopy(cfg)
    tc = cfg["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(tc.get("amp")) and device.type == "cuda"
    torch.backends.cudnn.benchmark = True   # faster convs for fixed sizes; tiny non-determinism (CONTRACTS 3.11)

    # ---- cheap checks first, so a wrong config fails before any folder is created ----
    batch_size = int(tc["batch_size"])
    if batch_size % NUM_CLASSES != 0:
        raise ValueError(f"train.batch_size must be a multiple of {NUM_CLASSES}, got {batch_size}")
    for key in ("warmup_epochs", "joint_epochs"):
        if is_tbd(tc[key]):
            raise ValueError(f"train.{key} is TBD_AFTER_BENCHMARK: set a real epoch count")
    warmup_epochs, joint_epochs = int(tc["warmup_epochs"]), int(tc["joint_epochs"])
    if warmup_epochs < 0 or joint_epochs < 0 or warmup_epochs + joint_epochs < 1:
        raise ValueError(f"need at least one epoch, got warmup {warmup_epochs} + joint {joint_epochs}")
    if tc.get("val_subset") and int(tc["val_subset"]) % NUM_CLASSES != 0:
        raise ValueError(f"train.val_subset must be a multiple of {NUM_CLASSES} (whole images), got {tc['val_subset']}")
    lam = {key: float(tc[key]) for key in LAMBDA_KEYS}
    if abs(lam["lambda_1"] + lam["lambda_s"] - 1.0) > 1e-6:
        raise ValueError(f"train.lambda_1 + train.lambda_s must be 1, got {lam['lambda_1']} + {lam['lambda_s']}")
    tau = float(cfg["model"]["tau"])
    if not tau > 0:
        raise ValueError(f"model.tau must be positive, got {tau}")

    # ---- the Task 2 checkpoints: found, hashed, never written. Their smoke flag becomes the run's. ----
    src_paths = sources if sources is not None else find_sources(cfg)
    verify_sources(src_paths, expected_hashes(cfg))                 # ValueError names the file and both hashes
    src_record = source_record(src_paths)
    src_smoke = any(info["smoke"] for info in describe_checkpoints(src_paths).values())
    cfg["run"] = dict(cfg["run"], smoke=bool(cfg["run"].get("smoke")) or src_smoke)

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
        saved_src = state["config"].get("source_checkpoints") or {}
        if any(saved_src.get(name) != src_record.get(name) for name in TASK2_SHA256):
            raise ValueError("resume: the source checkpoints differ from the ones this run started from")

    # ---- model: fresh from the Task 2 files, or rebuilt from the checkpoint when resuming ----
    if state is None:
        model = SoftMoE.load_from_task2(src_paths, tau, expected_sha256=expected_hashes(cfg))
    else:
        model = SoftMoE.from_config(state["config"]["model"])
        model.load_state_dict(state["model"])
    cfg["component"] = "soft_moe"
    cfg["model"] = model.model_config()                             # gate, experts, tau, branches (decision B11)
    cfg["source_checkpoints"] = src_record

    # ---- run folder ----
    if state is not None:
        cfg["run_id"] = state["config"]["run_id"]
        run_dir = runs_dir(cfg) / cfg["run_id"]
        _adopt_run_dir(Path(ckpt_path), run_dir)
    else:
        base = runs_dir(cfg)
        wanted = cfg.get("run_id") or make_run_id(cfg["device"], cfg["run"]["desc"], bool(cfg["run"].get("smoke")))
        cfg["run_id"], run_dir = _unique_run_dir(base, wanted)
        run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "samples").mkdir(exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"run dir: {run_dir}")

    seed_everything(cfg["seed"])

    # ---- data ----
    data_paths = resolve_data_paths(cfg)
    train_ds, sampler, train_loader = build_train_loader(cfg, data_paths, batch_size)   # balanced_batch
    val_ds, n_val, val_loader, manifest_sha = build_val_loader(cfg, data_paths)
    split_sha = split_sha256(train_ds.split)
    if state is not None and (state["split_sha256"] != split_sha or state["manifest_sha256"] != manifest_sha):
        raise ValueError("resume: split / manifest hashes differ from the checkpoint (different data)")
    sample_idx = fixed_sample_indices(val_ds, n_val, N_SAMPLE_ROWS)

    # ---- schedule: one epoch counter over both stages ----
    steps_per_epoch = len(sampler.inner)
    epochs_total = warmup_epochs + joint_epochs
    max_steps = tc.get("max_steps")
    total_steps = min(epochs_total * steps_per_epoch, int(max_steps)) if max_steps else epochs_total * steps_per_epoch
    joint_steps = max(total_steps - warmup_epochs * steps_per_epoch, 1)     # the cosine runs over these

    # ---- optimizer of the current stage, scaler ----
    model = model.to(device)
    print(describe(model))
    stage = state["stage"] if state is not None else stage_of(0, warmup_epochs)
    optimizer, scheduler = make_stage(model, stage, tc, joint_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epoch, step_in_epoch, global_step, best_J, n_joint_validated = 0, 0, 0, None, 0
    sums = {"loss": 0.0, "recon": 0.0, "ce": 0.0, "balance": 0.0}          # running sums over the current epoch
    w_sum = torch.zeros(NUM_CLASSES, device=device)                        # running sum of the batch mean weights
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        epoch, step_in_epoch = state["epoch"], state["step_in_epoch"]
        global_step, best_J = state["global_step"], state["best_metric"]
        n_joint_validated = state.get("n_joint_validated", 0)
        sums = {k: state["epoch_sums"][k] for k in sums}
        w_sum = torch.tensor(state["epoch_sums"]["w"], device=device)
        restore_rng_state(state["rng"])          # python / numpy / torch / cuda generators
        print(f"resumed from {ckpt_path}: stage {stage}, epoch {epoch}, step_in_epoch {step_in_epoch}, "
              f"global_step {global_step}")

    def save(name: str) -> Path:
        """Write ckpt_<name>.pt atomically and tell the caller (e.g. copy to Drive)."""
        ckpt = build_checkpoint(model, optimizer, scheduler, scaler, epoch=epoch, global_step=global_step,
                                best_metric=best_J, config=cfg, seed=cfg["seed"],
                                split_sha256=split_sha, manifest_sha256=manifest_sha)
        ckpt["step_in_epoch"], ckpt["rng"], ckpt["stage"] = step_in_epoch, capture_rng_state(), stage
        ckpt["epoch_sums"] = {**sums, "w": w_sum.tolist()}
        ckpt["n_joint_validated"] = n_joint_validated
        path = run_dir / f"ckpt_{name}.pt"
        save_checkpoint(path, ckpt)
        if on_checkpoint is not None:
            on_checkpoint(path)
        return path

    def write_row(row: dict) -> None:
        with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    tracking.init_run(cfg, cfg["run_id"], project=cfg["run"].get("project", "genai-a1"), group=TRACKER_GROUP)
    ckpt_every = float(tc.get("checkpoint_every_minutes", 10)) * 60
    log_every = int(tc.get("log_every_steps", 20))
    sample_every = int(tc.get("sample_every_epochs", 1))
    val_every = max(int(tc.get("val_every_epochs", 1)), 1)
    pause_after = tc.get("pause_after_steps")
    last_ckpt_time, paused = time.monotonic(), False

    try:
        model.train()                            # the gate in train mode, the experts stay in eval (B4, B5)

        # ---- epoch 0: the untrained soft copy of Task 2 (final runs only; never eligible as best) ----
        if state is None and tc.get("epoch0_validation", True):
            res0 = validate(model, val_loader, device, cfg.get("collapse"))
            write_row({"epoch": 0, "stage": "init", "global_step": 0, "partial_epoch": False,
                       "train_loss": None, "train_recon": None, "train_ce": None, "train_balance": None,
                       "train_mean_w": None, "lr": None, **_val_columns(res0), "epoch_seconds": None})
            tracking.log({"val/J": res0["J"], "val/gate_accuracy": res0["gate_accuracy"]}, 0)
            print(f"epoch 0 (Task 2 copy, untrained): val_J {res0['J']:.4f}  gate_acc {res0['gate_accuracy']:.3f}  "
                  f"mean_w {[round(v, 3) for v in res0['mean_w']]}")

        while epoch < epochs_total and global_step < total_steps and not paused:
            if stage_of(epoch, warmup_epochs) != stage:      # warm-up is over: a new optimizer over everything
                stage = stage_of(epoch, warmup_epochs)
                optimizer, scheduler = make_stage(model, stage, tc, joint_steps)
                print(f"epoch {epoch + 1}: stage {stage} (joint_lr {tc['joint_lr']}, experts trainable)")
            sampler.inner.epoch = epoch              # fixes the batch order of this epoch
            sampler.skip = step_in_epoch             # already-trained batches (mid-epoch resume)
            t_epoch = time.monotonic()
            for corrupted, clean, cond, _sev in train_loader:
                corrupted = corrupted.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)
                labels = cond.to(device, non_blocking=True)          # the class the corruption pipeline applied
                loss, parts, w_mean = _train_step(model, optimizer, scaler, corrupted, clean, labels, lam,
                                                  use_amp, tc.get("grad_clip"))
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
                step_in_epoch += 1
                sums["loss"] += loss
                for key in ("recon", "ce", "balance"):
                    sums[key] += parts[key]
                w_sum += w_mean
                if global_step % log_every == 0:
                    tracking.log({"train/loss": loss, "train/recon": parts["recon"], "train/ce": parts["ce"],
                                  "train/balance": parts["balance"], "train/lr": optimizer.param_groups[0]["lr"]},
                                 global_step)
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
            do_val = (epoch + 1) % val_every == 0 or epoch + 1 >= epochs_total or global_step >= total_steps
            improved, report_args = False, None
            if do_val:
                res = validate(model, val_loader, device, cfg.get("collapse"))
                val_J, flags = res["J"], res["flags"]
                n = max(step_in_epoch, 1)
                row = {"epoch": epoch + 1, "stage": stage, "global_step": global_step, "partial_epoch": not epoch_done,
                       "train_loss": sums["loss"] / n, "train_recon": sums["recon"] / n, "train_ce": sums["ce"] / n,
                       "train_balance": sums["balance"] / n, "train_mean_w": (w_sum / n).tolist(),
                       "lr": optimizer.param_groups[0]["lr"], **_val_columns(res),
                       "epoch_seconds": time.monotonic() - t_epoch}
                write_row(row)
                flat = {"val/J": val_J, "val/l1": res["l1"], "val/ssim": res["ssim"], "val/psnr": res["psnr"],
                        "val/gate_accuracy": res["gate_accuracy"], "val/collapsed": float(flags["collapsed"]),
                        "train/epoch_loss": row["train_loss"], "train/epoch_recon": row["train_recon"],
                        "train/epoch_ce": row["train_ce"], "train/epoch_balance": row["train_balance"]}
                flat.update({f"val/mean_w_{name}": res["mean_w"][k] for k, name in enumerate(BRANCH_NAMES)})
                tracking.log(flat, global_step)
                print(f"epoch {epoch + 1}/{epochs_total} [{stage}] step {global_step}: train_loss {row['train_loss']:.4f}  "
                      f"val_J {val_J:.4f}  ssim {res['ssim']:.3f}  gate_acc {res['gate_accuracy']:.3f}  "
                      f"mean_w {[round(v, 3) for v in res['mean_w']]}" + ("  COLLAPSED" if flags["collapsed"] else ""))

                if (epoch + 1) % sample_every == 0 or global_step >= total_steps:
                    grid = save_sample_grid(model, val_ds, sample_idx, device,
                                            run_dir / "samples" / f"epoch{epoch + 1:03d}_step{global_step}.png")
                    tracking.log_images("val/samples (target, input, x_hat, error)", grid, global_step)

                # best = lowest J among joint-stage validations that are not collapsed (decision B14)
                if stage == "joint":
                    n_joint_validated += 1
                    improved = (not flags["collapsed"]) and (best_J is None or val_J < best_J)
                report_args = (val_J, {**flags, "mean_w": res["mean_w"], "mean_w_by_class": res["mean_w_by_class"]})

            if epoch_done:                           # advance the counters BEFORE saving
                epoch += 1
                step_in_epoch = 0
                sums = {k: 0.0 for k in sums}
                w_sum = torch.zeros(NUM_CLASSES, device=device)
            if improved:
                best_J = val_J
                save("best")
            save("last")
            last_ckpt_time = time.monotonic()
            if report_fn is not None and epoch_done and report_args is not None:
                report_fn(epoch, *report_args)       # Optuna: may raise TrialPruned
            if global_step >= total_steps:
                break
    finally:
        tracking.finish()

    if not paused and n_joint_validated > 0 and best_J is None:
        raise ValueError("every joint-stage validation was flagged as routing collapse, so there is no "
                         "eligible best checkpoint (see collapse_reasons in metrics.jsonl)")
    return run_dir, best_J


def run_training(cfg: dict, resume: str | None = None, on_checkpoint=None) -> Path:
    """Train the soft MoE (see the module docstring). Returns the run directory."""
    run_dir, _best = _train(cfg, resume=resume, on_checkpoint=on_checkpoint)
    return run_dir


# ============================================================================== benchmark
def _endless(loader):
    """Batches forever: start over when the loader is used up (a new epoch of the balanced sampler)."""
    while True:
        yield from loader


def _sync(device) -> None:
    """Wait for the GPU, so that a timer measures finished work (no-op on the CPU)."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure_step_times(cfg: dict, batch: int, amp: bool, steps: int, warmup: int, sources: dict | None = None) -> dict:
    """Seconds per training step of both stages, validation times and peak memory (decision C, plan D8).

    Runs `warmup` untimed + `steps` timed steps of the warm-up stage, then of the joint stage, on the real
    balanced training data (so loading is included, as in a real run), then one full validation pass and one
    pass over cfg['trial_val_subset'] rows. `sources` = {name: Path} of the Task 2 checkpoints (None =
    find_sources(cfg)). Works on the CPU (peak_mb 0.0); with CUDA the timers wait for the GPU.
    Returns {warmup_s_per_step, joint_s_per_step, val_full_s, val_subset_s, peak_mb, params}.
    The model starts from the Task 2 files and is thrown away; nothing is written.
    """
    cfg = copy.deepcopy(cfg)
    tc = cfg["train"]
    steps = max(int(steps), 1)
    tc["batch_size"], tc["amp"], tc["val_subset"] = int(batch), bool(amp), None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(amp) and device.type == "cuda"
    torch.backends.cudnn.benchmark = True
    lam = {key: float(tc[key]) for key in LAMBDA_KEYS}

    src_paths = sources if sources is not None else find_sources(cfg)
    model = SoftMoE.load_from_task2(src_paths, float(cfg["model"]["tau"]),
                                    expected_sha256=expected_hashes(cfg)).to(device)
    data_paths = resolve_data_paths(cfg)
    _ds, _sampler, train_loader = build_train_loader(cfg, data_paths, int(batch))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    batches = _endless(train_loader)
    model.train()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def time_stage(stage: str) -> float:
        optimizer, _sched = make_stage(model, stage, tc, joint_steps=steps)
        for i in range(int(warmup) + steps):
            if i == int(warmup):
                _sync(device)
                t0 = time.monotonic()
            corrupted, clean, cond, _sev = next(batches)
            _train_step(model, optimizer, scaler, corrupted.to(device), clean.to(device), cond.to(device),
                        lam, use_amp, None)
        _sync(device)
        return (time.monotonic() - t0) / steps

    warmup_s = time_stage("warmup")
    joint_s = time_stage("joint")
    peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else 0.0

    def time_validation(rows) -> float:
        vcfg = copy.deepcopy(cfg)
        vcfg["train"]["val_subset"] = rows
        _vds, _n, loader, _sha = build_val_loader(vcfg, data_paths)
        _sync(device)
        t0 = time.monotonic()
        validate(model, loader, device)
        _sync(device)
        return time.monotonic() - t0

    full_s = time_validation(None)
    subset_rows = cfg.get("trial_val_subset")
    subset_s = time_validation(int(subset_rows)) if subset_rows and not is_tbd(subset_rows) else full_s
    return {"warmup_s_per_step": warmup_s, "joint_s_per_step": joint_s, "val_full_s": full_s,
            "val_subset_s": subset_s, "peak_mb": float(peak_mb), "params": int(count_parameters(model, trainable_only=False))}
