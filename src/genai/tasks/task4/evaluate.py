"""Task 4 evaluation of the style-conditioned generator. Implements CONTRACTS 3.4 (test lock), 3.6 and plan D.

Entry point:  run_evaluation(cfg, checkpoint, final_test=False) -> output directory (Path)

* final_test=False (default): evaluates the VAL split.
* final_test=True: evaluates the locked official TEST split (FS2KDataset logs the access to
  artifacts/test_access.log). Only the final evaluation step may use this.

`checkpoint` is a run checkpoint (ckpt_best.pt / ckpt_last.pt) or a promoted t4_generator.pt.
Every image is generated with its OWN style (the style id of the pair) and compared with its real sketch,
on sketches rescaled to [0,1] with 1 channel (L1, SSIM, PSNR from genai.common.metrics).

Outputs (<output_root>/eval/task4/<YYYYmmdd-HHMMSS>_<val|test>/):
  per_image.csv         pair_id, style, l1, ssim, psnr   (one row per image)
  by_style.csv          style, count, l1, ssim, psnr     (mean per style)
  summary.json          overall means, per-style means, split, count, checkpoint
  results_grid.png      8 photos per style, 3 style blocks side by side; each block: photo | real | generated
  failures_grid.png     the 8 worst images by L1; rows of: photo | real | generated
  style_variations.png  8 photos x the three styles; rows of: photo | style 0 | style 1 | style 2
                        (qualitative only: there is no real sketch for the other two styles)

Optional config key: cfg["eval"]["batch_size"] (default 32).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from genai.common.metrics import l1, psnr, ssim
from genai.export.task4_export import load_t4_generator
from genai.fs2k.dataset import FS2KDataset
from genai.tasks.task4.config import resolve

NUM_STYLES = 3
N_SHOWN = 8          # photos per style in results_grid, worst images in failures_grid, photos in style_variations


def _to01(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] -> [0,1] (the scale of all reported metrics and of all drawings)."""
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def _rgb(sketch01: torch.Tensor) -> torch.Tensor:
    """A 1-channel [1,H,W] sketch -> 3 channels, so it can sit next to colour photos in one grid."""
    return sketch01.expand(3, -1, -1)


@torch.no_grad()
def _per_image_metrics(G, ds, device, batch_size: int) -> pd.DataFrame:
    """Run G over the whole dataset (each image with its own style); one row per image."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    rows = []
    for photo, sketch, style, pair_ids in loader:
        fake = _to01(G(photo.to(device), style.to(device))).cpu()
        real = _to01(sketch)
        a, b, c = l1(fake, real), ssim(fake, real), psnr(fake, real)       # each (batch,)
        for k in range(len(a)):
            rows.append({"pair_id": pair_ids[k], "style": int(style[k]),
                         "l1": a[k].item(), "ssim": b[k].item(), "psnr": c[k].item()})
    return pd.DataFrame(rows)


@torch.no_grad()
def _generate(G, ds, indices, device, style=None) -> tuple:
    """Photos, real sketches and generated sketches (all [0,1]) for dataset items `indices`.

    style=None: every photo gets its own style. style=k: every photo gets style k."""
    items = [ds[int(i)] for i in indices]
    photo = torch.stack([it[0] for it in items])
    real = torch.stack([_to01(it[1]) for it in items])
    styles = torch.stack([it[2] for it in items]) if style is None else torch.full((len(items),), style, dtype=torch.int64)
    fake = _to01(G(photo.to(device), styles.to(device))).cpu()
    return _to01(photo), real, fake


def _draw_results(G, ds, df: pd.DataFrame, device, path: Path) -> None:
    """results_grid.png: row i holds the i-th image of style 0, 1 and 2, each as photo | real | generated.
    A style with fewer than 8 images leaves white tiles."""
    blank = torch.ones(3, 128, 128)
    blocks = []                                           # blocks[k] = list of rows, a row = 3 tiles
    for k in range(NUM_STYLES):
        idx = df.index[df["style"] == k][:N_SHOWN]        # row number in df == dataset index
        rows = []
        if len(idx):
            photo, real, fake = _generate(G, ds, idx, device)
            rows = [[photo[j], _rgb(real[j]), _rgb(fake[j])] for j in range(len(idx))]
        rows += [[blank, blank, blank]] * (N_SHOWN - len(rows))
        blocks.append(rows)
    tiles = []
    for r in range(N_SHOWN):                              # row-major: 9 tiles per row (3 styles x 3)
        for k in range(NUM_STYLES):
            tiles += blocks[k][r]
    save_image(make_grid(torch.stack(tiles), nrow=3 * NUM_STYLES, padding=2), path)


def _draw_failures(G, ds, df: pd.DataFrame, device, path: Path) -> None:
    """failures_grid.png: the 8 images with the highest L1, rows of photo | real | generated."""
    worst = df.sort_values("l1", ascending=False).head(N_SHOWN).index
    photo, real, fake = _generate(G, ds, worst, device)
    tiles = []
    for j in range(len(worst)):
        tiles += [photo[j], _rgb(real[j]), _rgb(fake[j])]
    save_image(make_grid(torch.stack(tiles), nrow=3, padding=2), path)


def _draw_style_variations(G, ds, device, path: Path) -> None:
    """style_variations.png: 8 photos spread evenly over the split, rows of photo | style 0 | style 1 | style 2."""
    idx = np.unique(np.linspace(0, len(ds) - 1, N_SHOWN).round().astype(int))
    photo = None
    fakes = []
    for k in range(NUM_STYLES):
        photo, _real, fake = _generate(G, ds, idx, device, style=k)
        fakes.append(fake)
    tiles = []
    for j in range(len(idx)):
        tiles += [photo[j]] + [_rgb(fakes[k][j]) for k in range(NUM_STYLES)]
    save_image(make_grid(torch.stack(tiles), nrow=1 + NUM_STYLES, padding=2), path)


def run_evaluation(cfg: dict, checkpoint, final_test: bool = False) -> Path:
    """Evaluate a Task 4 generator checkpoint (see module docstring). Returns the output directory."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G, ckpt = load_t4_generator(checkpoint)
    G = G.to(device)                                       # load_t4_generator already set eval() (dropout off)

    split_name = "test" if final_test else "val"
    # final_test=True makes FS2KDataset log the access to artifacts/test_access.log (CONTRACTS 3.4)
    ds = FS2KDataset(split_name, cfg["data_root"], augment=False, final_test=final_test)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = resolve(cfg["output_root"]) / "eval" / "task4" / f"{stamp}_{split_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int((cfg.get("eval") or {}).get("batch_size", 32))
    df = _per_image_metrics(G, ds, device, batch_size)
    df.to_csv(out_dir / "per_image.csv", index=False)

    metrics = ["l1", "ssim", "psnr"]
    by_style = df.groupby("style")[metrics].mean().assign(count=df.groupby("style").size()).reset_index()
    by_style[["style", "count"] + metrics].to_csv(out_dir / "by_style.csv", index=False)

    summary = {
        "checkpoint": str(checkpoint), "run_id": ckpt["config"].get("run_id", ""), "split": split_name,
        "count": len(df), "global_step": ckpt["global_step"],
        "overall": {k: float(df[k].mean()) for k in metrics},
        "by_style": {str(int(r.style)): {"count": int(r.count), **{k: float(getattr(r, k)) for k in metrics}}
                     for r in by_style.itertuples()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _draw_results(G, ds, df, device, out_dir / "results_grid.png")
    _draw_failures(G, ds, df, device, out_dir / "failures_grid.png")
    _draw_style_variations(G, ds, device, out_dir / "style_variations.png")

    o = summary["overall"]
    print(f"evaluated {len(df)} images on {split_name}: L1 {o['l1']:.4f} SSIM {o['ssim']:.4f} "
          f"PSNR {o['psnr']:.2f}; results in {out_dir}")
    return out_dir
