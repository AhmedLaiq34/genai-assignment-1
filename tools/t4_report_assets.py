"""Task 4 report assets: tables and figures built only from files that already exist (plan G).

    python tools/t4_report_assets.py                       # newest local final run + newest *_val evaluation
    python tools/t4_report_assets.py --run-dir artifacts/runs/task4/<run_id> --eval-dir artifacts/eval/task4/<stamp>_val

Reads : the final run folder (metrics.jsonl, samples/), the evaluation folder (per_image.csv, grids),
        data/splits/fs2k_split.json, studies/task4_cgan/ (the study rebuilt from the Colab console log).
Writes: report/figures/task4/*.png and report/tables/task4/*.csv
No training, no test set (the evaluation folder is the validation one unless you pass a *_test folder yourself).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

FIG = ROOT / "report" / "figures" / "task4"
TAB = ROOT / "report" / "tables" / "task4"


def newest(pattern_dir: Path, glob: str) -> Path:
    found = sorted(pattern_dir.glob(glob), key=lambda p: p.stat().st_mtime)
    if not found:
        raise SystemExit(f"nothing matches {pattern_dir / glob}")
    return found[-1]


def loss_curves(rows: pd.DataFrame) -> None:
    """The four separate losses required by the PDF (D real, D fake, G adversarial, G L1) + mean sigmoid(D)."""
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    ax[0].plot(rows["epoch"], rows["train/d_real"], label="D real")
    ax[0].plot(rows["epoch"], rows["train/d_fake"], label="D fake")
    ax[0].set_title("Discriminator losses")
    ax[1].plot(rows["epoch"], rows["train/g_adv"], label="G adversarial", color="tab:green")
    ax[1].set_title("Generator adversarial loss")
    ax[2].plot(rows["epoch"], rows["train/g_l1"], label="G reconstruction (L1)", color="tab:red")
    ax[2].set_title("Generator L1 (on [-1,1])")
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(FIG / "losses.png", dpi=150)
    plt.close(fig)

    fig, a = plt.subplots(figsize=(5.5, 3.6))
    a.plot(rows["epoch"], rows["train/d_real_prob"], label="mean sigmoid(D) on real")
    a.plot(rows["epoch"], rows["train/d_fake_prob"], label="mean sigmoid(D) on fake")
    a.set_xlabel("epoch")
    a.grid(alpha=0.3)
    a.legend()
    fig.tight_layout()
    fig.savefig(FIG / "d_probabilities.png", dpi=150)
    plt.close(fig)


def validation_curves(rows: pd.DataFrame) -> None:
    best = rows.loc[rows["val/l1"].idxmin()]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    ax[0].plot(rows["epoch"], rows["val/l1"], label="all")
    for s in range(3):
        ax[0].plot(rows["epoch"], rows[f"val/l1_style{s}"], label=f"style {s + 1}", alpha=0.7)
    ax[0].axvline(best["epoch"], color="k", ls="--", lw=0.8, label=f"best epoch {int(best['epoch'])}")
    ax[0].set_title("Validation L1 ([0,1] sketches)")
    ax[1].plot(rows["epoch"], rows["val/ssim"], color="tab:purple")
    ax[1].axvline(best["epoch"], color="k", ls="--", lw=0.8)
    ax[1].set_title("Validation SSIM")
    ax[2].plot(rows["epoch"], rows["val/psnr"], color="tab:orange")
    ax[2].axvline(best["epoch"], color="k", ls="--", lw=0.8)
    ax[2].set_title("Validation PSNR (dB)")
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "val_metrics.png", dpi=150)
    plt.close(fig)


def sample_timeline(run_dir: Path) -> None:
    """The fixed validation photos at a few epochs: stack the saved sample grids vertically (same photos each time)."""
    from PIL import Image, ImageDraw
    files = sorted((run_dir / "samples").glob("epoch*.png"))
    if not files:
        return
    wanted = [1, 10, 25, 50, 100]
    picks = [f for f in files if int(f.stem.replace("epoch", "")) in wanted] or files[:: max(1, len(files) // 5)]
    imgs = [Image.open(f).convert("RGB") for f in picks]
    w = max(i.width for i in imgs)
    canvas = Image.new("RGB", (w, sum(i.height + 18 for i in imgs)), "white")
    y = 0
    for f, im in zip(picks, imgs):
        ImageDraw.Draw(canvas).text((4, y + 2), f"{f.stem}  (columns: photo | real sketch | style 1 | style 2 | style 3)", fill="black")
        canvas.paste(im, (0, y + 18))
        y += im.height + 18
    canvas.save(FIG / "sample_timeline.png")


def tables(eval_dir: Path) -> None:
    by_style = pd.read_csv(eval_dir / "by_style.csv")
    by_style.to_csv(TAB / "val_by_style.csv", index=False)
    split = json.loads((ROOT / "data" / "splits" / "fs2k_split.json").read_text(encoding="utf-8"))
    counts = pd.DataFrame({k: {f"style {s} (Style {int(s) + 1})": v for s, v in split["style_counts"][k].items()}
                           for k in ("train", "val", "test")})
    counts.loc["total"] = counts.sum()
    counts.to_csv(TAB / "fs2k_split_counts.csv")
    study = ROOT / "studies" / "task4_cgan"
    if (study / "trials.csv").exists():
        pd.read_csv(study / "trials.csv").to_csv(TAB / "study_log_trials.csv", index=False)
        for src, dst in (("optimization_history.png", "study_log_history.png"),
                         ("param_importances.png", "study_log_importances.png"),
                         ("parallel_coordinate.png", "study_log_parallel.png")):
            if (study / src).exists():
                shutil.copyfile(study / src, FIG / dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--eval-dir", default=None)
    args = ap.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else newest(ROOT / "artifacts" / "runs" / "task4", "*_t4_final*")
    eval_dir = Path(args.eval_dir) if args.eval_dir else newest(ROOT / "artifacts" / "eval" / "task4", "*_val")
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame([json.loads(line) for line in open(run_dir / "metrics.jsonl", encoding="utf-8")])
    rows = rows[~rows["partial_epoch"]] if "partial_epoch" in rows else rows
    loss_curves(rows)
    validation_curves(rows)
    sample_timeline(run_dir)
    tables(eval_dir)
    for name in ("results_grid.png", "failures_grid.png", "style_variations.png"):
        if (eval_dir / name).exists():
            shutil.copyfile(eval_dir / name, FIG / name.replace("_grid", ""))
    audit = ROOT / "report" / "figures" / "task4" / "fs2k_pair_audit.png"
    print("run:", run_dir.name, "| eval:", eval_dir.name)
    print("written:", sorted(p.name for p in FIG.glob("*.png")), sorted(p.name for p in TAB.glob("*.csv")))
    print("pair audit grid present:", audit.exists())


if __name__ == "__main__":
    main()
