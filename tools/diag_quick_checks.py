"""Quick sanity checks for an autoencoder BEFORE spending hours on an Optuna study (see docs/LESSONS_FROM_TASK1.md).

All checks use the TRAIN cache and the VAL manifest only (never the test set). Typical run time: seconds to a few minutes.

    python tools/diag_quick_checks.py baseline                 # "do nothing" J / SSIM of each corruption (the number to beat)
    python tools/diag_quick_checks.py calibrate                # J of an image replaced by its NxN thumbnail (what J 0.31 "looks like")
    python tools/diag_quick_checks.py overfit                  # can the model memorise 64 images? (if not: bug / optimisation problem)
    python tools/diag_quick_checks.py train --latent conv --depth 3 --latent-values 4096 --epochs 30 --cond all
    python tools/diag_quick_checks.py train --latent dense --depth 4 --latent-values 512 --epochs 30 --cond 1   # one corruption only

`train` trains a UniversalAE for N epochs on the real train loader and prints the validation J / SSIM / PSNR every 5 epochs,
per condition at the end. --cond all = Task 1 style (iid_uniform); --cond 1|2|3 = a specialist (fixed:k). Use the same
learning rate / batch / alpha / base channels as the study will use, and change ONE thing per run.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genai.common.constants import CLASS_NAMES  # noqa: E402
from genai.common.metrics import evaluate_restoration, l1, objective_J_batch, psnr, ssim  # noqa: E402
from genai.common.seed import seed_everything  # noqa: E402
from genai.models.autoencoder import UniversalAE, count_parameters  # noqa: E402
from genai.pets.dataset import resolve_data_paths  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def first_train_images(n: int) -> torch.Tensor:
    paths = resolve_data_paths("local")
    imgs = np.load(Path(paths["cache"]) / "trainval_images.npy", mmap_mode="r")
    x = torch.from_numpy(np.ascontiguousarray(imgs[:n])).permute(0, 3, 1, 2).float().div(255)
    return x.to(DEVICE)


def cmd_baseline(args):
    """J / SSIM of the corrupted VAL input against its clean target: a model must beat this to be restoring anything."""
    from genai.pets.dataset import PetsManifestDataset
    paths = resolve_data_paths("local")
    ds = PetsManifestDataset(Path(paths["manifests"]) / "pets_val_manifest.jsonl", "val", False, paths)
    sums = {}
    for i in range(len(ds)):
        corrupted, clean, cond, _sev = ds[i]
        j = objective_J_batch(corrupted[None], clean[None]).item()
        s = ssim(corrupted[None], clean[None]).item()
        a = sums.setdefault(CLASS_NAMES[cond], [0.0, 0.0, 0])
        a[0], a[1], a[2] = a[0] + j, a[1] + s, a[2] + 1
    for name, (j, s, n) in sums.items():
        print(f"{name:14s} input-as-output J {j / n:.4f}  SSIM {s / n:.3f}  ({n} rows)")


def cmd_calibrate(args):
    x = first_train_images(64)
    print("J of a clean image replaced by its low-resolution copy (smaller J = more detail kept):")
    for size in (8, 16, 32, 64):
        low = F.interpolate(F.interpolate(x, size=size, mode="area"), size=128, mode="bilinear", align_corners=False)
        print(f"  {size:3d}x{size:<3d}  J {objective_J_batch(low, x).mean():.4f}  SSIM {ssim(low, x).mean():.3f}  L1 {l1(low, x).mean():.4f}")


def build_model(args) -> UniversalAE:
    return UniversalAE(base_channels=args.channels, depth=args.depth, bottleneck_dim=args.latent_values,
                       dropout=args.dropout, latent=args.latent).to(DEVICE)


def cmd_overfit(args):
    from genai.tasks.task1.train import restoration_loss
    x = first_train_images(64)
    seed_everything(0)
    model = build_model(args).train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"overfit test on 64 clean images, {count_parameters(model):,} parameters")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randperm(64, device=DEVICE)[:32]
        loss = restoration_loss(model(x[idx]), x[idx], args.alpha)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % (args.steps // 3) == 0:
            model.eval()
            with torch.no_grad():
                y = model(x)
            model.train()
            print(f"  step {step}: J {objective_J_batch(y, x).mean():.4f}  SSIM {ssim(y, x).mean():.3f}  ({time.time() - t0:.0f}s)")
    print("expected for a healthy model: J well below 0.15 and SSIM above 0.7 (calibrate shows what 0.31 looks like)")


def cmd_train(args):
    from genai.tasks.task1.config import load_config
    from genai.tasks.task1.train import build_train_loader, build_val_loader, restoration_loss
    cfg = load_config("configs/task1_universal.yaml", "local", {"num_workers": 0})
    paths = resolve_data_paths(cfg)
    policy = "iid_uniform" if args.cond == "all" else f"fixed:{args.cond}"
    seed_everything(42)
    _ds, sampler, loader = build_train_loader(cfg, paths, args.batch)
    if policy != "iid_uniform":                       # specialist: only corruption k
        from genai.pets.dataset import PetsTrainDataset
        from genai.pets import samplers
        ds = PetsTrainDataset("train", policy, data_root=paths, seed=42)
        sampler = type(sampler)(samplers.make_batch_sampler(policy, len(ds), args.batch, 42))
        loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=0)
    _vds, _n, val_loader, _sha = build_val_loader(cfg, paths)
    model = build_model(args)
    steps = args.epochs * len(sampler.inner)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    print(f"{args.latent} latent, depth {args.depth}, {args.latent_values} latent values "
          f"({3 * 128 * 128 / args.latent_values:.0f}:1), {count_parameters(model):,} parameters, policy {policy}")
    for epoch in range(args.epochs):
        sampler.inner.epoch = epoch
        model.train()
        t0 = time.time()
        for corrupted, clean, _c, _s in loader:
            loss = restoration_loss(model(corrupted.to(DEVICE)), clean.to(DEVICE), args.alpha)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            r = evaluate_restoration(model, val_loader, DEVICE)["overall"]
            print(f"  epoch {epoch + 1:3d}  val J {r['J']:.4f}  SSIM {r['ssim']:.3f}  PSNR {r['psnr']:.2f}  ({time.time() - t0:.0f}s/epoch)", flush=True)
    res = evaluate_restoration(model.eval(), val_loader, DEVICE)
    print("per condition J:", {k: round(v["J"], 4) for k, v in res.items() if k not in ("overall", "by_severity")})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=["baseline", "calibrate", "overfit", "train"])
    ap.add_argument("--latent", choices=["dense", "conv"], default="conv")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--latent-values", type=int, default=4096, help="total number of latent values (bottleneck_dim)")
    ap.add_argument("--channels", type=int, default=64, help="base encoder channels")
    ap.add_argument("--dropout", type=float, default=0.03)
    ap.add_argument("--lr", type=float, default=1.7e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.86)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=600, help="overfit: optimiser steps")
    ap.add_argument("--cond", default="all", help="train: 'all' or 1|2|3 (one corruption, like a specialist)")
    args = ap.parse_args()
    {"baseline": cmd_baseline, "calibrate": cmd_calibrate, "overfit": cmd_overfit, "train": cmd_train}[args.check](args)


if __name__ == "__main__":
    main()
