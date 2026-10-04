"""Build the Task 1 figures and tables for the report from the finished runs (no training, no test-set access).

    python tools/make_t1_report_assets.py

Reads : artifacts/runs/task1/<FINAL_RUN>/metrics.jsonl, artifacts/eval/task1/<FINAL_RUN>_{val,test}/, studies/t1_universal_v2/
Writes: report/figures/t1_*.png, report/tables/t1_*.csv
The test folder is the output of the single approved --final-test evaluation; this script only reads its result files.
"""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FINAL_RUN = "20261004-1514_kaggle_t1_final_v2"
RUN_DIR = ROOT / "artifacts" / "runs" / "task1" / FINAL_RUN
VAL_DIR = ROOT / "artifacts" / "eval" / "task1" / f"{FINAL_RUN}_val"
TEST_DIR = ROOT / "artifacts" / "eval" / "task1" / f"{FINAL_RUN}_test"
STUDY_DIR = ROOT / "studies" / "t1_universal_v2"
FIGS, TABLES = ROOT / "report" / "figures", ROOT / "report" / "tables"

BLUE, ORANGE, GREY = "#2a6fb0", "#d9822b", "#8a8a8a"      # model, do-nothing baseline, neutral
CONDS = ["clean", "salt_pepper", "gaussian_blur", "occlusion"]
SEVS = ["low", "medium", "high"]


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def training_curves():
    rows = [json.loads(line) for line in open(RUN_DIR / "metrics.jsonl", encoding="utf-8")]
    epochs = [r["epoch"] for r in rows]
    best = min(rows, key=lambda r: r["val_J"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    a.plot(epochs, [r["train_loss"] for r in rows], color=BLUE)
    a.set(title="Training loss (alpha*L1 + (1-alpha)*(1-SSIM), alpha 0.50)", xlabel="epoch", ylabel="loss")
    b.plot(epochs, [r["val_J"] for r in rows], color=ORANGE)
    b.axvline(best["epoch"], color=GREY, ls="--", lw=1)
    b.annotate(f"best epoch {best['epoch']}\nJ = {best['val_J']:.4f}", (best["epoch"], best["val_J"]),
               xytext=(best["epoch"] - 45, best["val_J"] + 0.05), arrowprops=dict(arrowstyle="->", color=GREY))
    b.set(title="Validation objective J (lower is better)", xlabel="epoch", ylabel="J")
    for ax in (a, b):
        style(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "t1_training_curves.png", dpi=170)
    plt.close(fig)


def test_vs_input():
    t = pd.read_csv(TEST_DIR / "table_cond_severity.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, cond in zip(axes, CONDS[1:]):
        sub = t[(t["cond"] == cond) & (t["severity"].isin(SEVS))].set_index("severity").loc[SEVS]
        x = range(len(SEVS))
        ax.bar([i - 0.2 for i in x], sub["J_input"], 0.4, color=ORANGE, label="corrupted input (do nothing)")
        ax.bar([i + 0.2 for i in x], sub["J"], 0.4, color=BLUE, label="model output")
        ax.set_xticks(list(x), SEVS)
        ax.set_title(cond.replace("_", " "))
        style(ax)
    axes[0].set_ylabel("J on the test set (lower is better)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "t1_test_vs_input.png", dpi=170)
    plt.close(fig)


def architecture():
    """Block diagram of the final model (depth 3, 64 base channels, conv latent of 8 x 16 x 16 = 2048 values)."""
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axis("off")
    blocks = [("input", "128x128x3", GREY), ("conv s2\nBN ReLU", "64x64x64", BLUE), ("conv s2\nBN ReLU", "32x32x128", BLUE),
              ("conv s2\nBN ReLU", "16x16x256", BLUE), ("1x1 conv\n(latent z)", "16x16x8\n= 2048 values", ORANGE),
              ("1x1 conv\nReLU", "16x16x256", BLUE), ("deconv s2\nBN ReLU", "32x32x128", BLUE),
              ("deconv s2\nBN ReLU", "64x64x64", BLUE), ("deconv s2\nsigmoid", "128x128x3", GREY)]
    for i, (name, shape, colour) in enumerate(blocks):
        x = i * 1.2
        ax.add_patch(FancyBboxPatch((x, 0.9), 1.0, 1.0, boxstyle="round,pad=0.03", fc=colour, ec="none", alpha=0.9))
        ax.text(x + 0.5, 1.4, name, ha="center", va="center", color="white", fontsize=7.5)
        ax.text(x + 0.5, 0.55, shape, ha="center", va="center", fontsize=7.5)
        if i:
            ax.annotate("", (x - 0.02, 1.4), (x - 0.18, 1.4), arrowprops=dict(arrowstyle="->", color="#444"))
    ax.text(0.5 + 4 * 1.2, 2.25, "bottleneck: 24:1 compression, no skip connections", ha="center", fontsize=8.5, color=ORANGE)
    ax.text(2.6, 2.25, "encoder", ha="center", fontsize=9)
    ax.text(0.5 + 7 * 1.2, 2.25, "decoder", ha="center", fontsize=9)
    ax.set(xlim=(-0.3, 11.1), ylim=(0.2, 2.6))
    fig.savefig(FIGS / "t1_architecture.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def copy_files():
    for src, dst in [(STUDY_DIR / "optimization_history.png", "t1_optuna_history.png"),
                     (STUDY_DIR / "param_importances.png", "t1_optuna_importance.png"),
                     (STUDY_DIR / "parallel_coordinate.png", "t1_optuna_parallel.png"),
                     (TEST_DIR / "representative_12.png", "t1_test_representative_12.png"),
                     (TEST_DIR / "worst_4.png", "t1_test_worst_4.png"),
                     (VAL_DIR / "representative_12.png", "t1_val_representative_12.png")]:
        shutil.copyfile(src, FIGS / dst)
    shutil.copyfile(TEST_DIR / "table_cond_severity.csv", TABLES / "t1_test_results.csv")
    shutil.copyfile(VAL_DIR / "table_cond_severity.csv", TABLES / "t1_val_results.csv")
    shutil.copyfile(STUDY_DIR / "trials.csv", TABLES / "t1_optuna_trials.csv")


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    training_curves()
    test_vs_input()
    architecture()
    copy_files()
    print("wrote", sorted(p.name for p in FIGS.glob("t1_*")), sorted(p.name for p in TABLES.glob("t1_*")))


if __name__ == "__main__":
    main()
