"""Task 2 report assets: tables (CSV + LaTeX) and figures, built only from files that already exist.

    python tools/t2_report_assets.py                  # uses the newest local *_val evaluation folder
    python tools/t2_report_assets.py --eval-dir artifacts/eval/task2/<stamp>_test   # test tables/figures go to report/*/task2/test/
    python tools/t2_report_assets.py --eval-dir artifacts/eval/task2/20261004-213747_val

Reads : the evaluation folder (tables, classifier report, misroute table, plots), studies/t2_*/trials.csv,
        the final runs' metrics.jsonl, and the promoted checkpoints (for the example restorations and parameter counts).
Writes: report/tables/task2/*.csv|*.tex and report/figures/task2/*.png
No training, no test set (the evaluation folder is the validation one).
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
import torch  # noqa: E402
from torchvision.utils import make_grid, save_image  # noqa: E402

from genai.common.constants import CLASS_NAMES  # noqa: E402
from genai.models.autoencoder import count_parameters  # noqa: E402
from genai.pets.dataset import PetsManifestDataset, resolve_data_paths  # noqa: E402
from genai.tasks.task2.routing import HardRoutedSystem, load_task2_models  # noqa: E402

TABLES, FIGURES = ROOT / "report" / "tables" / "task2", ROOT / "report" / "figures" / "task2"
CKPTS = {"classifier": "t2_classifier", "salt": "t2_ae_salt", "blur": "t2_ae_blur", "occlusion": "t2_ae_occlusion"}
RUN_GLOBS = {"classifier": "task2_classifier", "salt": "task2_specialist_salt", "blur": "task2_specialist_blur",
             "occlusion": "task2_specialist_occlusion"}
COND_OF = {"salt": "salt_pepper", "blur": "gaussian_blur", "occlusion": "occlusion"}
N_EXAMPLES = 4                      # example rows per corruption in the restoration grids
ERROR_GAIN = 4.0                    # |error| x4 so small errors are visible (same as the evaluations)


def to_tex(df: pd.DataFrame, path: Path, caption: str = "") -> None:
    """A plain LaTeX tabular (pandas' to_latex needs jinja2, which is not installed)."""
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v).replace("_", r"\_")
    cols = [str(c).replace("_", r"\_") for c in df.columns]
    lines = [r"\begin{tabular}{" + "l" * len(cols) + "}", r"\hline", " & ".join(cols) + r" \\", r"\hline"]
    lines += [" & ".join(fmt(v) for v in row) + r" \\" for row in df.itertuples(index=False)]
    lines += [r"\hline", r"\end{tabular}"]
    path.write_text(("% " + caption + "\n" if caption else "") + "\n".join(lines) + "\n", encoding="utf-8")


def save_table(df: pd.DataFrame, name: str, caption: str) -> None:
    df.to_csv(TABLES / f"{name}.csv", index=False)
    to_tex(df, TABLES / f"{name}.tex", caption)


def newest_val_eval() -> Path:
    cands = sorted((ROOT / "artifacts" / "eval" / "task2").glob("2*_val"))
    if not cands:
        raise SystemExit("no artifacts/eval/task2/*_val folder; run scripts/evaluate.py --task t2cls first")
    return cands[-1]


# ------------------------------------------------------------------------------------------ tables
def tables(eval_dir: Path, studies: bool = True) -> None:
    cols = ["cond", "severity", "MAE", "SSIM", "PSNR", "J", "SSIM_input", "J_input", "count"]
    for mode in ("oracle", "predicted"):
        t = pd.read_csv(eval_dir / f"table_cond_severity_{mode}.csv")[cols]
        save_table(t, f"results_{mode}", f"Validation results with {mode} routing (J_input/SSIM_input = no restoration)")

    r = json.loads((eval_dir / "classifier_report.json").read_text(encoding="utf-8"))
    rows = [{"class": c, **{k: v[k] for k in ("precision", "recall", "f1", "support")}} for c, v in r["per_class"].items()]
    rows += [{"class": "macro average", "precision": r["macro_precision"], "recall": r["macro_recall"], "f1": r["macro_f1"],
              "support": sum(v["support"] for v in r["per_class"].values())},
             {"class": "accuracy", "precision": "", "recall": "", "f1": r["accuracy"], "support": ""}]
    save_table(pd.DataFrame(rows), "classifier_report", "Classifier metrics on the validation manifest")
    cm = pd.DataFrame(r["confusion_normalised"], index=CLASS_NAMES, columns=CLASS_NAMES).round(4)
    save_table(cm.reset_index().rename(columns={"index": "true class"}), "confusion_normalised", "Row-normalised confusion matrix")

    mis = pd.read_csv(eval_dir / "misroute_confusion.csv")
    save_table(mis, "misroute_confusion", "Routing failures: true class (rows) vs predicted class (columns), counts")

    if not studies:                 # the Optuna study tables are the same for the validation and the test run
        return
    summary = []
    for name, path in (("t2_classifier", ROOT / "studies/t2_classifier/trials.csv"),
                       ("t2_specialist_shared", ROOT / "studies/t2_specialist_shared/trials.csv")):
        t = pd.read_csv(path)
        counts = t["state"].value_counts().to_dict()
        done = t[t["state"] == "COMPLETE"]
        best = done.loc[done["value"].idxmax() if name == "t2_classifier" else done["value"].idxmin()]
        plain = lambda v: v.item() if hasattr(v, "item") else v  # noqa: E731  (numpy numbers -> python numbers for json)
        params = {c.replace("params_", ""): (round(plain(best[c]), 6) if isinstance(plain(best[c]), float) else plain(best[c]))
                  for c in t.columns if c.startswith("params_")}
        summary.append({"study": name, "trials": len(t), "complete": counts.get("COMPLETE", 0), "pruned": counts.get("PRUNED", 0),
                        "failed": counts.get("FAIL", 0), "best_trial": int(best["number"]), "best_value": round(best["value"], 5),
                        "best_params": json.dumps(params)})
    save_table(pd.DataFrame(summary), "study_summary", "Optuna studies: trial counts and best trial")


# ------------------------------------------------------------------------------------------ figures
def curves() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    runs = {c: next((ROOT / "artifacts/runs" / RUN_GLOBS[c]).glob("*/metrics.jsonl")) for c in RUN_GLOBS}
    cls = pd.DataFrame([json.loads(line) for line in open(runs["classifier"], encoding="utf-8")])
    axes[0].plot(cls["epoch"], cls["val_macro_f1"], label="val macro-F1")
    axes[0].plot(cls["epoch"], cls["val_accuracy"], label="val accuracy", linestyle="--")
    axes[0].set_title("Classifier (validation)"), axes[0].set_xlabel("epoch"), axes[0].legend(), axes[0].grid(alpha=.3)
    axes[1].plot(cls["epoch"], cls["train_loss"], label="train loss")
    axes[1].plot(cls["epoch"], cls["val_loss"], label="val loss")
    axes[1].set_title("Classifier loss (cross-entropy)"), axes[1].set_xlabel("epoch"), axes[1].legend(), axes[1].grid(alpha=.3)
    for c in ("salt", "blur", "occlusion"):
        d = pd.DataFrame([json.loads(line) for line in open(runs[c], encoding="utf-8")])
        axes[2].plot(d["epoch"], d["val_J"], label=f"{c} (own corruption)")
    axes[2].set_title("Specialists: validation J (lower is better)"), axes[2].set_xlabel("epoch"), axes[2].legend(), axes[2].grid(alpha=.3)
    fig.tight_layout(), fig.savefig(FIGURES / "training_curves.png", dpi=150), plt.close(fig)


def bars(eval_dir: Path) -> None:
    tabs = {m: pd.read_csv(eval_dir / f"table_cond_severity_{m}.csv") for m in ("oracle", "predicted")}
    order = ["clean", "salt_pepper", "gaussian_blur", "occlusion", "overall"]
    pick = lambda t, col: [float(t[(t["cond"] == c) & (t["severity"] == "all")][col].iloc[0]) if c != "overall"
                           else float(t[t["cond"] == "overall"][col].iloc[0]) for c in order]  # noqa: E731
    vals = {"input (no restoration)": pick(tabs["oracle"], "J_input"), "oracle routing": pick(tabs["oracle"], "J"),
            "predicted routing": pick(tabs["predicted"], "J")}
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.27
    for i, (name, v) in enumerate(vals.items()):
        ax.bar([x + (i - 1) * w for x in range(len(order))], v, w, label=name)
    ax.set_xticks(range(len(order))), ax.set_xticklabels(order), ax.set_ylabel("validation J (lower is better)")
    ax.legend(), ax.grid(axis="y", alpha=.3), fig.tight_layout()
    fig.savefig(FIGURES / "j_by_condition.png", dpi=150), plt.close(fig)


@torch.no_grad()
def example_grids() -> dict:
    """Per corruption: N_EXAMPLES val rows (medium severity, first rows of the manifest), rows of target|input|output|error."""
    paths = resolve_data_paths("local")
    ds = PetsManifestDataset(Path(paths["manifests"]) / "pets_val_manifest.jsonl", "val", False, paths)
    system = HardRoutedSystem(load_task2_models({c: ROOT / "models/checkpoints" / f"{n}.pt" for c, n in CKPTS.items()}))
    for cond, cond_name in COND_OF.items():
        rows = [i for i, r in enumerate(ds.rows) if r["cond_name"] == cond_name and r["severity"] == "medium"][:N_EXAMPLES]
        tiles = []
        for i in rows:
            corrupted, clean, cid, _ = ds[i]
            out = system.restore_by_route(corrupted.unsqueeze(0), torch.tensor([int(cid)]))[0]
            tiles += [clean, corrupted, out, ((out - clean).abs() * ERROR_GAIN).clamp(0, 1)]
        save_image(make_grid(torch.stack(tiles), nrow=4, padding=2), FIGURES / f"restoration_examples_{cond}.png")
    return {c: m for c, m in system.models.items()}


def model_facts(models: dict) -> None:
    rows = [{"model": name, "parameters": count_parameters(m)} for name, m in models.items()]
    cfg = json.loads(json.dumps({k: v for k, v in models["blur"].hparams.items()}))
    rows.append({"model": "specialist architecture", "parameters": json.dumps(cfg)})
    rows.append({"model": "classifier architecture", "parameters": json.dumps(models["classifier"].hparams)})
    save_table(pd.DataFrame(rows), "model_facts", "Parameter counts and constructor arguments of the four Task 2 models")


def copy_existing(eval_dir: Path, studies: bool = True) -> None:
    for src, dst in ((eval_dir / "confusion_normalised.png", "confusion_normalised.png"),
                     (eval_dir / "routing_failures_worst.png", "routing_failures_worst.png")):
        if src.exists():
            shutil.copyfile(src, FIGURES / dst)
    for study, prefix in ((("t2_classifier", "cls"), ("t2_specialist_shared", "spec")) if studies else ()):
        for plot in ("optimization_history", "param_importances", "parallel_coordinate"):
            src = ROOT / "studies" / study / f"{plot}.png"
            if src.exists():
                shutil.copyfile(src, FIGURES / f"{prefix}_{plot}.png")
            else:
                print(f"missing study plot: {src}")


def comparison_with_task1(eval_dir: Path) -> None:
    """Test comparison on the IDENTICAL test manifest tensors: input vs Task 1 universal AE vs Task 2 oracle / predicted routing."""
    t1 = pd.read_csv(ROOT / "report" / "tables" / "t1_test_results.csv")
    t1 = t1[t1["severity"] == "all"].set_index("cond")
    oracle = pd.read_csv(eval_dir / "table_cond_severity_oracle.csv")
    pred = pd.read_csv(eval_dir / "table_cond_severity_predicted.csv")
    rows = []
    for cond in ["clean", "salt_pepper", "gaussian_blur", "occlusion", "overall"]:
        o = oracle[(oracle["cond"] == cond) & (oracle["severity"] == "all")].iloc[0]
        q = pred[(pred["cond"] == cond) & (pred["severity"] == "all")].iloc[0]
        rows.append({"condition": cond, "J input": o["J_input"], "J Task 1 universal": float(t1.loc[cond, "J"]),
                     "J Task 2 oracle": o["J"], "J Task 2 predicted": q["J"], "SSIM input": o["SSIM_input"],
                     "SSIM Task 1": float(t1.loc[cond, "SSIM"]), "SSIM Task 2 oracle": o["SSIM"], "SSIM Task 2 predicted": q["SSIM"], "images": int(o["count"])})
    save_table(pd.DataFrame(rows), "comparison_t1_t2_test", "Test set (36,690 cases): do-nothing input, Task 1 universal AE, Task 2 oracle and predicted routing (J lower is better)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", default=None)
    args = ap.parse_args()
    eval_dir = Path(args.eval_dir) if args.eval_dir else newest_val_eval()
    if not eval_dir.is_absolute():
        eval_dir = ROOT / eval_dir
    is_test = eval_dir.name.endswith("_test")
    assert is_test or eval_dir.name.endswith("_val"), "give a *_val or *_test evaluation folder"
    global TABLES, FIGURES
    if is_test:                      # test assets go to their own sub-folders; the validation ones stay as they are
        TABLES, FIGURES = TABLES / "test", FIGURES / "test"
    TABLES.mkdir(parents=True, exist_ok=True), FIGURES.mkdir(parents=True, exist_ok=True)
    print("evaluation folder:", eval_dir)
    tables(eval_dir, studies=not is_test)
    copy_existing(eval_dir, studies=not is_test)
    bars(eval_dir)
    if is_test:
        comparison_with_task1(eval_dir)
    else:                            # curves and example grids come from the training runs / validation rows
        curves()
        model_facts(example_grids())
    print("tables:", sorted(p.name for p in TABLES.iterdir()))
    print("figures:", sorted(p.name for p in FIGURES.iterdir()))


if __name__ == "__main__":
    main()
