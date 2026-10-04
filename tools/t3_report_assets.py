"""Task 3 report assets: tables (CSV + LaTeX) and figures, built only from files that already exist.

    python tools/t3_report_assets.py                       # the newest local *_val evaluation folder
    python tools/t3_report_assets.py --eval-dir artifacts/eval/task3/<stamp>_val
    python tools/t3_report_assets.py --eval-dir artifacts/eval/task3/<stamp>_test --allow-test   # the approved test run

Reads : the evaluation folder (tasks/task3/evaluate.py), the final run's metrics.jsonl, studies/t3_moe/ (trials.csv and
        the Optuna plots), configs/task3_moe.yaml (search space), configs/task3_moe_final.yaml, optionally the pipeline's
        t3_summary.json and the T3 checkpoint (parameter counts). Nothing is trained, no image is rendered from the test set.
Writes: report/tables/task3/*.csv|*.tex and report/figures/task3/*.png   (folders can be changed with --tables-dir / --figures-dir)

A *_test evaluation folder is refused unless --allow-test is given; its assets go to a `test` sub-folder of both output
folders and only the tables and figures that come from the evaluation folder are made (the study and the training run
are the same for the validation and the test numbers).

Missing optional inputs (no run folder, no study, no config) are skipped with a printed note, so the script can be run
for whatever exists. The list of files follows docs/TASK3_PLAN.md section G.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from genai.common.constants import CLASS_NAMES  # noqa: E402
from genai.tasks.task3 import BRANCH_NAMES  # noqa: E402

W_COLUMNS = ["w_" + name for name in BRANCH_NAMES]
SYSTEM_LABELS = {"input": "input (no restoration)", "t1": "Task 1 universal", "t2_oracle": "Task 2 oracle",
                 "t2_predicted": "Task 2 predicted", "t3": "Task 3 soft MoE"}
CONDITIONS = ["clean", "salt_pepper", "gaussian_blur", "occlusion", "overall"]


# ------------------------------------------------------------------------------------------ helpers
def to_tex(df: pd.DataFrame, path: Path, caption: str = "") -> None:
    """A plain LaTeX tabular (pandas' to_latex needs jinja2, which is not installed).

    Same helper as tools/t2_report_assets.py (a refactor candidate: that file is a script, not an importable module).
    """
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    cols = [str(c).replace("_", r"\_") for c in df.columns]
    lines = [r"\begin{tabular}{" + "l" * len(cols) + "}", r"\hline", " & ".join(cols) + r" \\", r"\hline"]
    lines += [" & ".join(fmt(v) for v in row) + r" \\" for row in df.itertuples(index=False)]
    lines += [r"\hline", r"\end{tabular}"]
    path.write_text(("% " + caption + "\n" if caption else "") + "\n".join(lines) + "\n", encoding="utf-8")


class Assets:
    """Collects what is written, so the script (and the tests) can list it."""

    def __init__(self, tables_dir: Path, figures_dir: Path):
        self.tables_dir, self.figures_dir = Path(tables_dir), Path(figures_dir)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.written, self.skipped = [], []

    def table(self, df: pd.DataFrame, name: str, caption: str) -> None:
        df.to_csv(self.tables_dir / f"{name}.csv", index=False)
        to_tex(df, self.tables_dir / f"{name}.tex", caption)
        self.written += [f"tables/{name}.csv", f"tables/{name}.tex"]

    def figure(self, fig, name: str) -> None:
        fig.savefig(self.figures_dir / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.written.append(f"figures/{name}")

    def copy(self, src: Path, name: str) -> None:
        if Path(src).exists():
            shutil.copyfile(src, self.figures_dir / name)
            self.written.append(f"figures/{name}")
        else:
            self.skip(f"figures/{name}", f"{src} does not exist")

    def skip(self, what: str, why: str) -> None:
        self.skipped.append(f"{what}: {why}")
        print(f"skipped {what}: {why}")


def newest_val_eval(root: Path = ROOT) -> Path:
    candidates = sorted((root / "artifacts" / "eval" / "task3").glob("2*_val"))
    if not candidates:
        raise SystemExit("no artifacts/eval/task3/*_val folder; run scripts/evaluate.py --task t3 first")
    return candidates[-1]


def newest_run(root: Path = ROOT):
    """The newest folder with a metrics.jsonl under artifacts/runs/task3 (the final training), or None."""
    candidates = sorted(p for p in (root / "artifacts" / "runs" / "task3").glob("*/metrics.jsonl"))
    return candidates[-1].parent if candidates else None


# ------------------------------------------------------------------------------------------ evaluation tables
def eval_tables(eval_dir: Path, out: Assets) -> dict:
    """The tables that come straight from the evaluation folder. Returns the parsed summary.json."""
    for name, caption in (
            ("comparison_cond_severity", "J and SSIM of input / Task 1 / Task 2 oracle / Task 2 predicted / Task 3 (lower J is better)"),
            ("weights_by_class_severity", "Mean routing weights per true class and severity"),
            ("expert_activity", "Expert activity: mean weight, argmax share, weight on own and on other classes, flags"),
            ("expert_drift", "Expert drift: J of each expert's own output on its own class, Task 3 vs the Task 2 specialist")):
        path = eval_dir / f"{name}.csv"
        if path.exists():
            out.table(pd.read_csv(path), name, caption)
        else:
            out.skip(f"tables/{name}", f"{path} does not exist")

    summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
    gate = summary["gate_vs_classifier"]
    rows = [{"class": c, "T3 gate accuracy": gate["t3_gate_accuracy_by_class"].get(c, float("nan")),
             "Task 2 classifier accuracy": gate["t2_classifier_accuracy_by_class"].get(c, float("nan"))}
            for c in CLASS_NAMES]
    rows.append({"class": "all rows", "T3 gate accuracy": gate["t3_gate_accuracy"],
                 "Task 2 classifier accuracy": gate["t2_classifier_accuracy"]})
    out.table(pd.DataFrame(rows), "gate_vs_classifier",
              f"Gate accuracy vs the Task 2 classifier on the same {gate['n_rows']} rows")
    return summary


# ------------------------------------------------------------------------------------------ evaluation figures
def weights_distribution(per_image: pd.DataFrame, out: Assets) -> None:
    """One panel per true class: box plots of the four weights over its rows."""
    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(14, 3.6), sharey=True)
    for c, ax in enumerate(axes):
        rows = per_image[per_image["true_class"] == c]
        if len(rows):
            ax.boxplot([rows[w] for w in W_COLUMNS], tick_labels=list(BRANCH_NAMES))
        ax.set_title(f"true class: {CLASS_NAMES[c]} (n={len(rows)})", fontsize=9)
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("routing weight")
    fig.tight_layout()
    out.figure(fig, "weights_distribution.png")


def j_comparison(comparison: pd.DataFrame, out: Assets) -> None:
    """Grouped bars per condition (all severities): J of every system that is in the comparison table."""
    systems = [s for s in SYSTEM_LABELS if f"J_{s}" in comparison.columns]
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / len(systems)
    for i, system in enumerate(systems):
        values = []
        for cond in CONDITIONS:
            row = comparison[(comparison["cond"] == cond) & (comparison["severity"] == "all")]
            values.append(float(row[f"J_{system}"].iloc[0]) if len(row) else float("nan"))
        ax.bar([x + (i - (len(systems) - 1) / 2) * width for x in range(len(CONDITIONS))], values, width,
               label=SYSTEM_LABELS[system])
    ax.set_xticks(range(len(CONDITIONS)), CONDITIONS)
    ax.set_ylabel("validation J (lower is better)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out.figure(fig, "j_comparison.png")


def architecture(out: Assets) -> None:
    """Gate + identity branch + three experts + weighted sum, drawn with boxes and arrows."""
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 10), ax.set_ylim(0, 6), ax.axis("off")

    def box(x, y, w, h, text, color="#dbe9f6"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=color, ec="#33506b"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#33506b"))

    box(0.1, 2.4, 1.4, 1.0, r"input $\tilde{x}$" + "\n3x128x128", "#f3f3f3")
    box(2.6, 4.7, 2.4, 0.9, "gate G (Task 2 classifier\ninit.): 4 logits", "#fde7c8")
    box(5.7, 4.7, 2.2, 0.9, r"$w=\mathrm{softmax}(G(\tilde{x})/\tau)$", "#fde7c8")
    branches = [("identity: $\\tilde{x}$", 0.3), ("$A_{salt}(\\tilde{x})$", 1.3),
                ("$A_{blur}(\\tilde{x})$", 2.3), ("$A_{occlusion}(\\tilde{x})$", 3.3)]
    for k, (text, y) in enumerate(branches):
        box(2.6, y, 2.4, 0.7, f"branch {k}: {text}", "#dbe9f6" if k else "#e5f2dc")
        arrow(1.6, 2.9, 2.55, y + 0.35)                                   # the input goes to every branch
        arrow(5.05, y + 0.35, 7.35, 2.7)                                  # and each branch output to the sum
    arrow(1.6, 3.2, 2.55, 5.1)                                            # the input also goes to the gate
    arrow(5.05, 5.15, 5.65, 5.15)
    arrow(6.8, 4.65, 8.0, 3.35)                                           # w to the sum
    ax.text(7.75, 4.15, "w0..w3", fontsize=8)
    box(7.4, 2.1, 2.2, 1.2, r"$\hat{x}=\sum_k w_k\,\mathrm{branch}_k$", "#e5f2dc")
    arrow(8.5, 2.05, 8.5, 1.3)
    ax.text(8.5, 0.95, r"output $\hat{x}$ and weights $w$", ha="center", fontsize=9)
    ax.set_title("Task 3: soft mixture of experts (all four branches run, the weights mix them)", fontsize=10)
    out.figure(fig, "t3_architecture.png")


# ------------------------------------------------------------------------------------------ training run
def read_metrics(run_dir: Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(line) for line in open(run_dir / "metrics.jsonl", encoding="utf-8") if line.strip()])


def training_curves(metrics: pd.DataFrame, out: Assets) -> None:
    """Val J per epoch (warm-up / joint boundary, epoch 0 marked), train loss parts, val mean weights, gate accuracy."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    # first joint-stage epoch (the epoch-0 row, measured before any training, does not count)
    boundary = metrics[(metrics["stage"] == "joint") & (metrics["epoch"] >= 1)]["epoch"].min() if "stage" in metrics else math.nan
    val = metrics[metrics["val_J"].notna()] if "val_J" in metrics else metrics.iloc[0:0]

    ax = axes[0][0]
    ax.plot(val["epoch"], val["val_J"], marker="o", ms=3, label="val J")
    start = val[val["epoch"] == 0]
    if len(start):
        ax.scatter(start["epoch"], start["val_J"], marker="*", s=140, color="tab:red", zorder=3,
                   label="epoch 0 (Task 2 copy, soft mode)")
    ax.set_title("Validation J (lower is better)"), ax.set_xlabel("epoch"), ax.legend(fontsize=8)

    ax = axes[0][1]
    train = metrics[metrics["train_loss"].notna()] if "train_loss" in metrics else metrics.iloc[0:0]
    for column, label in (("train_recon", "reconstruction"), ("train_ce", "cross-entropy"), ("train_balance", "balance")):
        if column in train:
            ax.plot(train["epoch"], train[column], marker="o", ms=3, label=label)
    ax.set_yscale("log"), ax.set_title("Training loss terms (before the lambda weights)"), ax.set_xlabel("epoch"), ax.legend(fontsize=8)

    ax = axes[1][0]
    weighted = val[val["val_mean_w"].apply(lambda v: isinstance(v, list))] if "val_mean_w" in val else val.iloc[0:0]
    if len(weighted):
        w = np.array(weighted["val_mean_w"].tolist())
        for k, name in enumerate(BRANCH_NAMES):
            ax.plot(weighted["epoch"], w[:, k], marker="o", ms=3, label=name)
    ax.set_title("Mean routing weight per branch (validation)"), ax.set_xlabel("epoch"), ax.legend(fontsize=8)

    ax = axes[1][1]
    if "val_gate_accuracy" in val:
        ax.plot(val["epoch"], val["val_gate_accuracy"], marker="o", ms=3, color="tab:green")
    ax.set_title("Gate accuracy (validation)"), ax.set_xlabel("epoch")

    for ax in axes.ravel():
        ax.grid(alpha=0.3)
        if not math.isnan(boundary):
            ax.axvline(boundary - 0.5, color="gray", linestyle="--", linewidth=1)    # warm-up | joint
    axes[0][0].text(boundary - 0.4 if not math.isnan(boundary) else 0, axes[0][0].get_ylim()[1], " joint stage", va="top", fontsize=8)
    fig.tight_layout()
    out.figure(fig, "training_curves.png")


# ------------------------------------------------------------------------------------------ study tables
def range_edge_flags(search_space: list, best_params: dict) -> dict:
    """{param: "low" | "high"} for a float parameter whose best value lies within 10 % of a range end
    (measured on the log scale for log ranges, linear otherwise)."""
    flags = {}
    for p in search_space:
        name, value = p["name"], best_params.get(p["name"])
        if p.get("type") != "float" or value is None:
            continue
        lo, hi = float(p["low"]), float(p["high"])
        position = ((math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo)) if p.get("log")
                    else (value - lo) / (hi - lo))
        if position <= 0.10:
            flags[name] = "low"
        elif position >= 0.90:
            flags[name] = "high"
    return flags


def study_summary(trials: pd.DataFrame, search_space: list, summary_json: dict | None, out: Assets) -> None:
    """Counts, pruned reasons, best trial, the PDF-start trial 0 and the range-edge flags, as item / value rows."""
    counts = trials["state"].value_counts().to_dict()
    complete = trials[trials["state"] == "COMPLETE"]
    rows = [("trials", len(trials)), ("complete", counts.get("COMPLETE", 0)), ("pruned", counts.get("PRUNED", 0)),
            ("failed", counts.get("FAIL", 0))]
    reasons = trials["user_attrs_pruned_reason"].dropna().value_counts().to_dict() if "user_attrs_pruned_reason" in trials else {}
    rows.append(("pruned reasons", json.dumps(reasons)))
    if summary_json:                    # the pipeline's t3_summary.json ("study" section) or the study folder's study_run.json
        timeout_hit = (summary_json.get("study") or summary_json).get("timeout_hit")
        rows.append(("timeout hit (stopped before the trial budget)", timeout_hit))
    def params(row) -> dict:
        """The params_* columns of one trial as plain python numbers / strings (json can not store numpy numbers)."""
        plain = {c.replace("params_", ""): row[c].item() if hasattr(row[c], "item") else row[c]
                 for c in trials.columns if c.startswith("params_") and not pd.isna(row[c])}
        return {k: round(v, 6) if isinstance(v, float) else v for k, v in plain.items()}

    if len(complete):
        best = complete.loc[complete["value"].idxmin()]
        best_params = params(best)
        rows += [("best trial", int(best["number"])), ("best value (fixed J on the trial validation subset)", round(float(best["value"]), 5)),
                 ("best params", json.dumps(best_params)),
                 ("range-edge flags (best value within 10 % of a range end)", json.dumps(range_edge_flags(search_space, best_params)))]
    first = trials[trials["number"] == 0]
    if len(first):                                       # trial 0 = the PDF start values (plan B17)
        first = first.iloc[0]
        value = "" if pd.isna(first["value"]) else round(float(first["value"]), 5)
        rows += [("trial 0 (PDF start values) state", first["state"]), ("trial 0 value", value),
                 ("trial 0 params", json.dumps(params(first)))]
    out.table(pd.DataFrame(rows, columns=["item", "value"]), "study_summary",
              "Optuna study t3_moe: trial counts, best trial, PDF-start trial 0, range edges")


def final_config_table(final_cfg: dict, trials: pd.DataFrame | None, out: Assets) -> None:
    """tau and the four lambdas of the final model next to trial 0 (the PDF start) and the other final settings."""
    model, train = final_cfg.get("model") or {}, final_cfg.get("train") or {}
    final = {"tau": model.get("tau"), "joint_lr": train.get("joint_lr"), "lambda_1": train.get("lambda_1"),
             "lambda_s": train.get("lambda_s"), "lambda_c": train.get("lambda_c"), "lambda_b": train.get("lambda_b")}
    start = {}
    if trials is not None and len(trials[trials["number"] == 0]):
        t0 = trials[trials["number"] == 0].iloc[0]
        share = t0.get("params_recon_l1_share")
        start = {"tau": t0.get("params_tau"), "joint_lr": t0.get("params_joint_lr"), "lambda_c": t0.get("params_lambda_c"),
                 "lambda_b": t0.get("params_lambda_b"),
                 "lambda_1": share, "lambda_s": None if share is None or pd.isna(share) else 1.0 - float(share)}
    rows = [(k, v, start.get(k, "")) for k, v in final.items()]
    rows += [("warm-up epochs", train.get("warmup_epochs"), ""), ("joint epochs", train.get("joint_epochs"), ""),
             ("warm-up lr", train.get("warmup_lr"), ""), ("batch size", train.get("batch_size"), ""),
             ("weight decay", train.get("weight_decay"), ""), ("collapse thresholds", json.dumps(final_cfg.get("collapse") or {}), ""),
             ("final_config_source", json.dumps(final_cfg.get("final_config_source") or {}, default=str), "")]
    out.table(pd.DataFrame(rows, columns=["parameter", "final model", "trial 0 (PDF start)"]), "final_config",
              "Final Task 3 configuration (best study trial) next to trial 0 (the PDF start values)")


def model_facts(summary: dict, checkpoint, out: Assets) -> None:
    """Parameter counts (from the checkpoint's own model config), tau, lambdas, thresholds and hashes."""
    rows = []
    if checkpoint is not None and Path(checkpoint).exists():
        from genai.common.checkpoint import load_checkpoint
        from genai.models.autoencoder import count_parameters
        from genai.models.moe import SoftMoE
        model = SoftMoE.from_config(load_checkpoint(checkpoint)["config"]["model"])
        rows += [("parameters, total", count_parameters(model, trainable_only=False)),
                 ("parameters, gate", count_parameters(model.gate, trainable_only=False))]
        rows += [(f"parameters, expert {n}", count_parameters(m, trainable_only=False)) for n, m in model.experts.items()]
    else:
        out.skip("model_facts parameter counts", "no readable T3 checkpoint (pass --checkpoint)")
    ck = summary.get("checkpoint", {})
    rows += [("tau", summary.get("tau")), ("branches", ", ".join(BRANCH_NAMES))]
    rows += [(k, v) for k, v in (summary.get("train_params") or {}).items()]
    rows += [(f"threshold: {k}", v) for k, v in (summary.get("thresholds") or {}).items()]
    rows += [("collapse thresholds", json.dumps({k: v for k, v in summary["collapse"].items() if k.endswith("weight")})),
             ("T3 checkpoint sha256", ck.get("t3_checkpoint_sha256")), ("run id", ck.get("run_id")),
             ("split", summary.get("split")), ("rows evaluated", summary.get("n_rows"))]
    rows += [(f"source sha256: {n}", r["sha256"]) for n, r in (summary.get("sources") or {}).items()]
    out.table(pd.DataFrame(rows, columns=["item", "value"]), "model_facts", "Task 3 model facts")


# ------------------------------------------------------------------------------------------ driver
def build_assets(eval_dir, tables_dir, figures_dir, run_dir=None, study_dir=None, config_path=None,
                 final_config=None, summary_json=None, checkpoint=None, allow_test: bool = False) -> Assets:
    """Make every table and figure that the given inputs allow. Returns the Assets object (written / skipped lists)."""
    eval_dir = Path(eval_dir)
    is_test = eval_dir.name.endswith("_test")
    assert eval_dir.name.endswith("_val") or (is_test and allow_test), (
        f"{eval_dir.name}: give a *_val evaluation folder (a *_test folder needs --allow-test, the approved test run)")
    if is_test:                                  # test assets stay apart from the validation ones
        tables_dir, figures_dir = Path(tables_dir) / "test", Path(figures_dir) / "test"
    out = Assets(tables_dir, figures_dir)
    print("evaluation folder:", eval_dir)

    summary = eval_tables(eval_dir, out)
    per_image = pd.read_csv(eval_dir / "per_image.csv")
    weights_distribution(per_image, out)
    j_comparison(pd.read_csv(eval_dir / "comparison_cond_severity.csv"), out)
    for name in ("routing_heatmap.png", "examples_dominant.png", "examples_distributed.png"):
        out.copy(eval_dir / name, name)
    if is_test:
        return out

    architecture(out)
    ckpt = checkpoint or (summary.get("checkpoint") or {}).get("path")
    model_facts(summary, ckpt, out)

    if run_dir is not None and (Path(run_dir) / "metrics.jsonl").exists():
        training_curves(read_metrics(Path(run_dir)), out)
    else:
        out.skip("figures/training_curves.png", f"no metrics.jsonl in {run_dir}")

    study_dir = Path(study_dir) if study_dir else None
    search_space = (yaml.safe_load(Path(config_path).read_text(encoding="utf-8")).get("tuned_params", [])
                    if config_path and Path(config_path).exists() else [])
    trials = pd.read_csv(study_dir / "trials.csv") if study_dir and (study_dir / "trials.csv").exists() else None
    if trials is not None:
        summary_path = Path(summary_json) if summary_json else study_dir / "study_run.json"     # written by the study
        sj = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
        study_summary(trials, search_space, sj, out)
        for plot, name in (("optimization_history", "t3_optuna_history"), ("param_importances", "t3_optuna_importance"),
                           ("parallel_coordinate", "t3_optuna_parallel")):
            out.copy(study_dir / f"{plot}.png", f"{name}.png")
    else:
        out.skip("study tables and Optuna figures", f"no trials.csv in {study_dir}")
    if final_config and Path(final_config).exists():
        final_config_table(yaml.safe_load(Path(final_config).read_text(encoding="utf-8")), trials, out)
    else:
        out.skip("tables/final_config", f"{final_config} does not exist")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", default=None, help="evaluation folder (default: the newest artifacts/eval/task3/*_val)")
    ap.add_argument("--run-dir", default=None, help="final training run folder with metrics.jsonl (default: the newest one)")
    ap.add_argument("--study-dir", default=str(ROOT / "studies" / "t3_moe"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "task3_moe.yaml"), help="study config (search space)")
    ap.add_argument("--final-config", default=str(ROOT / "configs" / "task3_moe_final.yaml"))
    ap.add_argument("--summary-json", default=None, help="the pipeline's t3_summary.json (default: <study-dir>/study_run.json); adds 'timeout hit' to the study table")
    ap.add_argument("--checkpoint", default=None, help="T3 checkpoint for the parameter counts (default: the one named in summary.json)")
    ap.add_argument("--tables-dir", default=str(ROOT / "report" / "tables" / "task3"))
    ap.add_argument("--figures-dir", default=str(ROOT / "report" / "figures" / "task3"))
    ap.add_argument("--allow-test", action="store_true", help="accept a *_test evaluation folder (the approved test run)")
    args = ap.parse_args(argv)

    eval_dir = Path(args.eval_dir) if args.eval_dir else newest_val_eval()
    if not eval_dir.is_absolute():
        eval_dir = ROOT / eval_dir
    run_dir = Path(args.run_dir) if args.run_dir else newest_run()
    out = build_assets(eval_dir, args.tables_dir, args.figures_dir, run_dir=run_dir, study_dir=args.study_dir,
                       config_path=args.config, final_config=args.final_config, summary_json=args.summary_json,
                       checkpoint=args.checkpoint, allow_test=args.allow_test)
    print("written:", *out.written, sep="\n  ")
    if out.skipped:
        print("skipped:", *out.skipped, sep="\n  ")


if __name__ == "__main__":
    main()
