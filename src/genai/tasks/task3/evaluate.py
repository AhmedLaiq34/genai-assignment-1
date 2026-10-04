"""Task 3 evaluation of the soft mixture of experts, with the routing analysis of PDF page 7.

    run_evaluation(cfg, checkpoint, final_test=False) -> output directory (Path)    plan 10.1 entry point
    evaluate_task3(cfg, checkpoint, final_test=False, sources=None) -> output directory
    load_soft_moe(ckpt_path) -> (SoftMoE in eval mode, checkpoint dict)

* final_test=False (default): the VAL manifest. final_test=True: the locked TEST manifest
  (PetsManifestDataset logs the access to artifacts/test_access.log). Never done during development.
* Optional cfg["eval"] keys (all have defaults):
      max_rows    evaluate only the first N manifest rows (default: all; a multiple of 4 keeps whole images)
      batch_size  images per forward pass (default 64)
      n_examples  rows drawn in each example figure (default 8)
  Optional cfg["collapse"] keys (the thresholds of the training study, plan B16): min_mean_weight (0.02),
  max_foreign_weight (0.9).

One pass over the manifest. For every batch the SAME corrupted tensors go through
    input      the "do nothing" baseline (the corrupted image scored against the clean target)
    T1         the Task 1 universal autoencoder (only when its checkpoint was found; otherwise no T1 column)
    T2 oracle  the Task 2 hard-routed system routed by the TRUE class      (task2.routing.HardRoutedSystem)
    T2 predicted   the same system routed by the classifier's argmax
    T3         the soft mixture, with the four branch outputs kept for the analysis
The Task 2 networks and T1 come from `sources` (default: find_sources(cfg), see tasks/task3/sources.py).
If the T3 checkpoint recorded the sha256 of its source files, the files found are checked against it.

Files in <output_root>/eval/task3/<YYYYmmdd-HHMMSS>_<val|test>/
  per_image.csv                  one row per manifest row: row, image_id, cond, severity, true_class, the four
                                 weights, dominant branch, max_w, entropy (natural log), MAE SSIM PSNR J of T3,
                                 SSIM_input J_input, J_t1 (if T1), J_t2_oracle, J_t2_predicted,
                                 t2_predicted_class, and J_branch_<name>: each branch output ALONE against the target
  comparison_cond_severity.csv   J and SSIM of input / T1 / T2 oracle / T2 predicted / T3 per condition x severity,
                                 per condition ("all") and overall, with the row count
  weights_by_class_severity.csv  mean of the four weights per TRUE class x severity (clean: one row)
  routing_heatmap.png            that table as a heatmap (PDF page 7)
  expert_activity.csv            per branch: mean weight, share of rows where it is the argmax, mean weight on its own
                                 class, largest mean weight on another class, flags inactive / dominates_unrelated
  expert_drift.csv               per expert: J of its output alone on its own class rows, T3 vs the Task 2 specialist
  examples_dominant.png          rows where one branch has max weight >= 0.95 (PDF page 7)
  examples_distributed.png       rows where the weights are spread (max weight <= 0.60)
  examples.csv                   which rows were drawn, the weights and the threshold that was used
  summary.json                   overall numbers, gate accuracy vs the Task 2 classifier on the same rows, collapse
                                 flags, activity flags, source and T3 checkpoint sha256, manifest / split sha256

Definitions (plan B19, recommendations stated again in the report): dominant row = max weight >= 0.95;
distributed row = max weight <= 0.60 (if fewer than n_examples such rows exist, the n_examples rows with the
most extreme max weight are drawn and the threshold reached is written to examples.csv); branch "inactive" =
mean weight < 0.02 or never the argmax; branch "dominates unrelated inputs" = mean weight > 0.5 on the rows of
another true class.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                       # draw to files only, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from genai.common.checkpoint import load_checkpoint, sha256_file  # noqa: E402
from genai.common.constants import CLASS_NAMES, SEVERITY_NAMES  # noqa: E402
from genai.common.metrics import l1, objective_J, psnr, ssim  # noqa: E402
from genai.export.onnx_export import load_t1_model  # noqa: E402
from genai.models.moe import SoftMoE  # noqa: E402
from genai.pets.dataset import PetsManifestDataset, resolve_data_paths  # noqa: E402
from genai.pets.split import split_sha256  # noqa: E402
from genai.tasks.task1.config import resolve  # noqa: E402
from genai.tasks.task1.evaluate import ERROR_GAIN, TEST_MANIFEST_NAME, VAL_MANIFEST_NAME  # noqa: E402
from genai.tasks.task2.routing import HardRoutedSystem, load_task2_models  # noqa: E402
from genai.tasks.task3 import BRANCH_NAMES  # noqa: E402
from genai.tasks.task3.sources import REQUIRED, find_sources, source_record, verify_sources  # noqa: E402

# ----------------------------------------------------------------------------- thresholds (plan B19)
DOMINANT_MIN_WEIGHT = 0.95        # "one expert dominates": the largest weight of the row is at least this
DISTRIBUTED_MAX_WEIGHT = 0.60     # "weights are distributed": the largest weight of the row is at most this
INACTIVE_MEAN_WEIGHT = 0.02       # a branch with a smaller mean weight is inactive
UNRELATED_MEAN_WEIGHT = 0.50      # a branch with a larger mean weight on another class's rows dominates unrelated inputs

W_COLUMNS = ["w_" + name for name in BRANCH_NAMES]                    # w_identity, w_salt, w_blur, w_occlusion
BRANCH_J_COLUMNS = ["J_branch_" + name for name in BRANCH_NAMES]
SEVERITY_ORDER = ["none", *SEVERITY_NAMES]                            # clean rows have severity "none"

# Columns of per_image.csv, in this order (J_t1 is left out when there is no Task 1 checkpoint).
PER_IMAGE_COLUMNS = (["row", "image_id", "cond", "severity", "true_class"] + W_COLUMNS
                     + ["dominant", "max_w", "entropy", "MAE", "SSIM", "PSNR", "J", "SSIM_input", "J_input",
                        "J_t1", "J_t2_oracle", "J_t2_predicted", "t2_predicted_class"] + BRANCH_J_COLUMNS)

# The systems of the comparison: (name, J column, SSIM column) in the wide table. T3's own columns are J / SSIM.
SYSTEMS = [("input", "J_input", "SSIM_input"), ("t1", "J_t1", "SSIM_t1"),
           ("t2_oracle", "J_t2_oracle", "SSIM_t2_oracle"), ("t2_predicted", "J_t2_predicted", "SSIM_t2_predicted"),
           ("t3", "J", "SSIM")]


# ======================================================================= loading
def load_soft_moe(ckpt_path) -> tuple:
    """Rebuild the SoftMoE of a Task 3 checkpoint. Returns (model in eval mode, checkpoint dict).

    The model is rebuilt from config.model (written by SoftMoE.model_config), so the Task 2 files are not needed.
    """
    ckpt = load_checkpoint(ckpt_path)
    component = (ckpt.get("config") or {}).get("component")
    if component != "soft_moe":
        raise ValueError(f"{ckpt_path} has component {component!r}, expected 'soft_moe'")
    model = SoftMoE.from_config(ckpt["config"]["model"])
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt            # eval(): gate dropout off, BatchNorm uses running statistics


def _is_smoke(ckpt: dict) -> bool:
    cfg = ckpt.get("config") or {}
    return bool((cfg.get("run") or {}).get("smoke")) or str(cfg.get("run_id", "")).endswith("_smoke")


def _find_and_check_sources(cfg: dict, ckpt: dict, sources) -> dict:
    """The Task 2 files (and T1 when present) used for the comparison; checked against the T3 checkpoint's record."""
    paths = dict(sources) if sources is not None else find_sources(cfg)
    recorded = (ckpt.get("config") or {}).get("source_checkpoints") or {}
    expected = {name: recorded[name]["sha256"] for name in REQUIRED if name in recorded}
    t1_sha = recorded["t1"]["sha256"] if paths.get("t1") and "t1" in recorded else None
    verify_sources(paths, expected, t1_sha=t1_sha)      # ValueError (with both hashes) if a file is not the recorded one
    return paths


# ======================================================================= running all systems
def _metrics(out: torch.Tensor, clean: torch.Tensor) -> dict:
    """MAE, SSIM, PSNR and J of a batch of outputs against the clean targets (tensors of shape (N,))."""
    mae, s, p = l1(out, clean), ssim(out, clean), psnr(out, clean)
    return {"MAE": mae, "SSIM": s, "PSNR": p, "J": objective_J(mae, s)}


@torch.no_grad()
def _run_all(model: SoftMoE, system: HardRoutedSystem, t1, ds, n_rows: int, batch_size: int, device) -> pd.DataFrame:
    """The same corrupted tensors through input, T1 (optional), T2 oracle, T2 predicted and T3. One row per manifest row."""
    loader = DataLoader(Subset(ds, range(n_rows)), batch_size=batch_size, shuffle=False, num_workers=0)
    records, i = [], 0
    for corrupted, clean, cond, _sev in loader:
        corrupted, clean, cond = corrupted.to(device), clean.to(device), cond.to(device)

        # ---- T3: the mixture, with the four branch outputs (branch 0 is the input itself)
        x_hat, w, _logits, branches = model(corrupted, return_branches=True)
        m3 = _metrics(x_hat, clean)
        entropy = -(w * w.clamp_min(1e-12).log()).sum(dim=1)
        branch_j = [_metrics(branches[:, k], clean)["J"] for k in range(len(BRANCH_NAMES))]

        # ---- the baseline and the Task 2 system on the same tensors
        m_in = _metrics(corrupted, clean)
        probs = system.classify(corrupted)
        t2_class = probs.argmax(dim=1)
        m_oracle = _metrics(system.restore_by_route(corrupted, cond), clean)
        m_pred = _metrics(system.restore_by_route(corrupted, t2_class), clean)
        m_t1 = _metrics(t1(corrupted), clean) if t1 is not None else None

        w_cpu, entropy = w.cpu(), entropy.cpu()
        for k in range(len(cond)):
            row = ds.rows[i]
            record = {"row": i, "image_id": row["image_id"], "cond": row["cond_name"],
                      "severity": row["severity"] or "none", "true_class": int(cond[k])}
            for b, column in enumerate(W_COLUMNS):
                record[column] = float(w_cpu[k, b])
            record["dominant"] = BRANCH_NAMES[int(w_cpu[k].argmax())]
            record["max_w"] = float(w_cpu[k].max())
            record["entropy"] = float(entropy[k])
            for name in ("MAE", "SSIM", "PSNR", "J"):
                record[name] = float(m3[name][k])
            record["SSIM_input"], record["J_input"] = float(m_in["SSIM"][k]), float(m_in["J"][k])
            if m_t1 is not None:
                record["J_t1"], record["SSIM_t1"] = float(m_t1["J"][k]), float(m_t1["SSIM"][k])
            record["J_t2_oracle"], record["SSIM_t2_oracle"] = float(m_oracle["J"][k]), float(m_oracle["SSIM"][k])
            record["J_t2_predicted"], record["SSIM_t2_predicted"] = float(m_pred["J"][k]), float(m_pred["SSIM"][k])
            record["t2_predicted_class"] = int(t2_class[k])
            for b, column in enumerate(BRANCH_J_COLUMNS):
                record[column] = float(branch_j[b][k])
            records.append(record)
            i += 1
    return pd.DataFrame(records)


# ======================================================================= tables
def _cond_severity_table(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Mean of `columns` per condition x severity, then a per-condition row (severity "all"), then "overall".

    Same idea as Task 1 / Task 2's tables, but in the order of the manifest's classes and severities.
    """
    rows = []
    for cond in CLASS_NAMES:
        part = df[df["cond"] == cond]
        if part.empty:
            continue
        for severity in SEVERITY_ORDER:
            group = part[part["severity"] == severity]
            if len(group):
                rows.append({"cond": cond, "severity": severity, **group[columns].mean().to_dict(), "count": len(group)})
        rows.append({"cond": cond, "severity": "all", **part[columns].mean().to_dict(), "count": len(part)})
    rows.append({"cond": "overall", "severity": "all", **df[columns].mean().to_dict(), "count": len(df)})
    return pd.DataFrame(rows)


def _comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """J and SSIM of every system that was run (T1 only if its column exists), named J_<system> / SSIM_<system>."""
    present = [(name, j, s) for name, j, s in SYSTEMS if j in df.columns]
    table = _cond_severity_table(df, [c for _n, j, s in present for c in (j, s)])
    table = table.rename(columns={c: f"{kind}_{name}" for name, j, s in present for kind, c in (("J", j), ("SSIM", s))})
    ordered = ["cond", "severity"] + [f"{kind}_{name}" for name, _j, _s in present for kind in ("J", "SSIM")] + ["count"]
    return table[ordered]


def _weights_by_class_severity(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of the four weights per TRUE class x severity (clean has one row, severity "none")."""
    rows = []
    for cond in CLASS_NAMES:
        for severity in SEVERITY_ORDER:
            group = df[(df["cond"] == cond) & (df["severity"] == severity)]
            if len(group):
                rows.append({"cond": cond, "severity": severity, **group[W_COLUMNS].mean().to_dict(), "count": len(group)})
    return pd.DataFrame(rows)


def _mean_weights_by_class(df: pd.DataFrame) -> dict:
    """{true class id: [mean weight of each of the four branches]} for the classes that occur."""
    return {int(c): g[W_COLUMNS].mean().tolist() for c, g in df.groupby("true_class")}


def _expert_activity(df: pd.DataFrame) -> pd.DataFrame:
    """One row per branch: is it used, is it used on the right inputs, does it take over foreign inputs (B19)."""
    by_class = _mean_weights_by_class(df)
    argmax_share = df["dominant"].value_counts(normalize=True)
    rows = []
    for k, name in enumerate(BRANCH_NAMES):
        others = {c: means[k] for c, means in by_class.items() if c != k}      # mean weight of branch k on other classes
        worst_class = max(others, key=others.get) if others else None
        max_other = others[worst_class] if others else float("nan")
        mean_weight = float(df[W_COLUMNS[k]].mean())
        share = float(argmax_share.get(name, 0.0))
        rows.append({
            "branch": name, "mean_weight": mean_weight, "argmax_share": share,
            "mean_weight_own_class": by_class[k][k] if k in by_class else float("nan"),
            "max_mean_weight_other_class": max_other,
            "other_class_of_max": CLASS_NAMES[worst_class] if worst_class is not None else "",
            "inactive": bool(mean_weight < INACTIVE_MEAN_WEIGHT or share == 0.0),
            "dominates_unrelated": bool(others and max_other > UNRELATED_MEAN_WEIGHT),
        })
    return pd.DataFrame(rows)


def _expert_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Per expert, on the rows of its own class: its output alone in T3 vs the Task 2 specialist (= oracle routing).

    delta = J_t3_branch - J_t2_specialist: positive means the joint training made the expert worse on its own corruption.
    """
    rows = []
    for k, name in enumerate(BRANCH_NAMES):
        if k == 0:
            continue                                     # the identity branch has no parameters that could drift
        own = df[df["true_class"] == k]
        if own.empty:
            continue
        j_t3, j_t2 = float(own[BRANCH_J_COLUMNS[k]].mean()), float(own["J_t2_oracle"].mean())
        rows.append({"expert": name, "own_class": CLASS_NAMES[k], "n_rows": len(own), "J_t3_branch": j_t3,
                     "J_t2_specialist": j_t2, "delta": j_t3 - j_t2, "J_input": float(own["J_input"].mean())})
    return pd.DataFrame(rows)


def _collapse_flags(df: pd.DataFrame, cfg: dict) -> dict:
    """The routing-collapse rule of the study (plan B16), applied to this evaluation's weights."""
    rule = cfg.get("collapse") or {}
    min_mean, max_foreign = float(rule.get("min_mean_weight", 0.02)), float(rule.get("max_foreign_weight", 0.9))
    mean_w = df[W_COLUMNS].mean().tolist()
    by_class = _mean_weights_by_class(df)
    reasons = [f"branch {BRANCH_NAMES[k]} has mean weight {mean_w[k]:.4f} < {min_mean}"
               for k in range(len(BRANCH_NAMES)) if mean_w[k] < min_mean]
    for c, means in by_class.items():
        reasons += [f"branch {BRANCH_NAMES[k]} has mean weight {means[k]:.4f} > {max_foreign} on {CLASS_NAMES[c]} rows"
                    for k in range(len(BRANCH_NAMES)) if k != c and means[k] > max_foreign]
    return {"collapsed": bool(reasons), "reasons": reasons, "min_mean_weight": min_mean, "max_foreign_weight": max_foreign}


# ======================================================================= figures
def _draw_heatmap(table: pd.DataFrame, path: Path) -> None:
    """Rows: true class x severity. Columns: the four branches. The mean weight is written in every cell."""
    labels = [c if s == "none" else f"{c} / {s}" for c, s in zip(table["cond"], table["severity"])]
    values = table[W_COLUMNS].to_numpy()
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(labels) + 1.8))
    image = ax.imshow(values, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(BRANCH_NAMES)), BRANCH_NAMES)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("branch")
    ax.set_ylabel("true corruption / severity")
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            ax.text(c, r, f"{values[r, c]:.2f}", ha="center", va="center",
                    color="white" if values[r, c] < 0.6 else "black", fontsize=9)
    fig.colorbar(image, ax=ax, label="mean routing weight")
    ax.set_title("Mean routing weights per true class and severity")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _pick_examples(df: pd.DataFrame, kind: str, n: int) -> tuple:
    """Choose n rows for the "dominant" or "distributed" figure. Returns (rows, threshold used, rule).

    Normal case: the rows that satisfy the threshold, 2 per true class where possible (the first rows of the
    manifest), topped up with further qualifying rows. If fewer than n rows qualify, the n rows with the most
    extreme max weight are taken (rule "fallback") and the threshold reached is reported.
    """
    if kind == "dominant":
        eligible, threshold = df[df["max_w"] >= DOMINANT_MIN_WEIGHT], DOMINANT_MIN_WEIGHT
        extreme = df.sort_values(["max_w", "row"], ascending=[False, True]).head(n)
        reached = float(extreme["max_w"].min()) if len(extreme) else float("nan")
    else:
        eligible, threshold = df[df["max_w"] <= DISTRIBUTED_MAX_WEIGHT], DISTRIBUTED_MAX_WEIGHT
        extreme = df.sort_values(["max_w", "row"], ascending=[True, True]).head(n)
        reached = float(extreme["max_w"].max()) if len(extreme) else float("nan")

    if len(eligible) < n:                                   # too few: show the most extreme rows instead
        return extreme, reached, "fallback"
    per_class = max(1, -(-n // len(CLASS_NAMES)))           # ceil(n / 4) rows from every class
    chosen = pd.concat([eligible[eligible["true_class"] == c].head(per_class) for c in range(len(CLASS_NAMES))])
    rest = eligible[~eligible["row"].isin(chosen["row"])]
    chosen = pd.concat([chosen, rest.head(max(0, n - len(chosen)))]).sort_values("row").head(n)
    return chosen, threshold, "threshold"


@torch.no_grad()
def _draw_examples(model: SoftMoE, ds, rows: pd.DataFrame, path: Path, device) -> None:
    """One image row per example: target | input | the four branch outputs (titled with weights) | x_hat | |error| x4."""
    n = max(1, len(rows))
    fig, axes = plt.subplots(n, 8, figsize=(16, 2.2 * n), squeeze=False)
    for r, (_, example) in enumerate(rows.iterrows()):
        corrupted, clean, _c, _s = ds[int(example["row"])]
        x_hat, w, _logits, branches = model(corrupted.unsqueeze(0).to(device), return_branches=True)
        x_hat, w, branches = x_hat.cpu()[0], w.cpu()[0], branches.cpu()[0]
        error = ((x_hat - clean).abs() * ERROR_GAIN).clamp(0, 1)
        tiles = [("target", clean), ("input", corrupted)]
        tiles += [(f"{name}  w={float(w[b]):.2f}", branches[b]) for b, name in enumerate(BRANCH_NAMES)]
        tiles += [("x_hat (mixture)", x_hat), ("|error| x4", error)]
        for c, (title, image) in enumerate(tiles):
            ax = axes[r][c]
            ax.imshow(image.permute(1, 2, 0).clamp(0, 1).numpy())
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0 or c in (2, 3, 4, 5):
                ax.set_title(title, fontsize=8)
        axes[r][0].set_ylabel(f"{example['cond']}\n{example['severity']}", fontsize=8)
    for r in range(len(rows), n):                           # an empty figure still gets a valid file
        for ax in axes[r]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ======================================================================= summary
def _accuracy_by_class(true: np.ndarray, predicted: np.ndarray) -> dict:
    return {CLASS_NAMES[c]: float((predicted[true == c] == c).mean()) for c in range(len(CLASS_NAMES)) if (true == c).any()}


def _system_means(df: pd.DataFrame) -> dict:
    """Overall J and SSIM of every system that was run."""
    return {name: {"J": float(df[j].mean()), "SSIM": float(df[s].mean())} for name, j, s in SYSTEMS if j in df.columns}


# ======================================================================= entry points
def evaluate_task3(cfg: dict, checkpoint, final_test: bool = False, sources: dict | None = None) -> Path:
    """Evaluate a Task 3 checkpoint and compare it with input, T1, T2 oracle and T2 predicted (see module docstring)."""
    eval_cfg = cfg.get("eval") or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ckpt = load_soft_moe(checkpoint)
    model = model.to(device)
    paths = _find_and_check_sources(cfg, ckpt, sources)
    system = HardRoutedSystem(load_task2_models({name: paths[name] for name in REQUIRED}, device), device)
    t1 = load_t1_model(paths["t1"])[0].to(device) if paths.get("t1") else None      # optional comparison column

    data_paths = resolve_data_paths(cfg)
    split_name = "test" if final_test else "val"
    manifest = Path(data_paths["manifests"]) / (TEST_MANIFEST_NAME if final_test else VAL_MANIFEST_NAME)
    ds = PetsManifestDataset(manifest, split=split_name, final_test=final_test, data_root=data_paths)
    n_rows = min(len(ds), int(eval_cfg["max_rows"])) if eval_cfg.get("max_rows") else len(ds)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = resolve(cfg["output_root"]) / "eval" / "task3" / f"{stamp}_{split_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- one pass over the manifest: all systems on identical tensors
    df = _run_all(model, system, t1, ds, n_rows, int(eval_cfg.get("batch_size", 64)), device)
    df[[c for c in PER_IMAGE_COLUMNS if c in df.columns]].to_csv(out_dir / "per_image.csv", index=False)
    _comparison_table(df).to_csv(out_dir / "comparison_cond_severity.csv", index=False)

    # ---- routing analysis (PDF page 7)
    weights_table = _weights_by_class_severity(df)
    weights_table.to_csv(out_dir / "weights_by_class_severity.csv", index=False)
    _draw_heatmap(weights_table, out_dir / "routing_heatmap.png")
    activity = _expert_activity(df)
    activity.to_csv(out_dir / "expert_activity.csv", index=False)
    _expert_drift(df).to_csv(out_dir / "expert_drift.csv", index=False)

    n_examples = int(eval_cfg.get("n_examples", 8))
    example_rows = []
    for kind in ("dominant", "distributed"):
        chosen, threshold, rule = _pick_examples(df, kind, n_examples)
        _draw_examples(model, ds, chosen, out_dir / f"examples_{kind}.png", device)
        example_rows.append(chosen[["row", "image_id", "cond", "severity", "true_class"] + W_COLUMNS + ["max_w"]]
                            .assign(figure=kind, threshold=threshold, rule=rule))
    if example_rows:
        pd.concat(example_rows).to_csv(out_dir / "examples.csv", index=False)

    # ---- summary: the gate against the Task 2 classifier, flags, hashes
    true = df["true_class"].to_numpy()
    gate_pred = df[W_COLUMNS].to_numpy().argmax(axis=1)          # argmax of the weights = argmax of the logits
    t2_pred = df["t2_predicted_class"].to_numpy()
    by_cond = {cond: {"J": float(g["J"].mean()), "J_input": float(g["J_input"].mean()),
                      "beats_input": bool(g["J"].mean() < g["J_input"].mean())} for cond, g in df.groupby("cond")}
    train_cfg = (ckpt.get("config") or {}).get("train") or {}
    summary = {
        "split": split_name,
        "n_rows": int(n_rows),
        "systems": _system_means(df),                                       # J and SSIM of input / t1 / t2 / t3
        "t3": {m: float(df[m].mean()) for m in ("MAE", "SSIM", "PSNR", "J")},
        "input_baseline": {m: float(df[m].mean()) for m in ("SSIM_input", "J_input")},
        "t3_vs_input_by_condition": by_cond,
        "has_t1": t1 is not None,
        "gate_vs_classifier": {
            "n_rows": int(n_rows),
            "t3_gate_accuracy": float((gate_pred == true).mean()),
            "t2_classifier_accuracy": float((t2_pred == true).mean()),
            "t3_gate_accuracy_by_class": _accuracy_by_class(true, gate_pred),
            "t2_classifier_accuracy_by_class": _accuracy_by_class(true, t2_pred),
            "gate_agrees_with_t2_classifier": float((gate_pred == t2_pred).mean()),
        },
        "collapse": _collapse_flags(df, cfg),
        "expert_activity": {r["branch"]: {"inactive": r["inactive"], "dominates_unrelated": r["dominates_unrelated"]}
                            for r in activity.to_dict("records")},
        "thresholds": {"dominant_min_weight": DOMINANT_MIN_WEIGHT, "distributed_max_weight": DISTRIBUTED_MAX_WEIGHT,
                       "inactive_mean_weight": INACTIVE_MEAN_WEIGHT, "unrelated_mean_weight": UNRELATED_MEAN_WEIGHT},
        "tau": model.tau,
        "train_params": {k: train_cfg.get(k) for k in ("lambda_1", "lambda_s", "lambda_c", "lambda_b",
                                                      "warmup_lr", "joint_lr", "warmup_epochs", "joint_epochs")},
        "checkpoint": {"path": str(checkpoint), "t3_checkpoint_sha256": sha256_file(checkpoint),
                       "run_id": (ckpt.get("config") or {}).get("run_id", ""), "global_step": ckpt.get("global_step"),
                       "stage": ckpt.get("stage"), "smoke": _is_smoke(ckpt)},
        "smoke": _is_smoke(ckpt),                                           # true: numbers come from smoke / fixture models
        "sources": source_record(paths),
        "sources_checked_against_checkpoint": bool(((ckpt.get("config") or {}).get("source_checkpoints"))),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "split_sha256": split_sha256(ds.split),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    gate = summary["gate_vs_classifier"]
    print(f"evaluated {n_rows} rows on {split_name}: T3 J {summary['t3']['J']:.4f} "
          f"(input {summary['input_baseline']['J_input']:.4f}, T2 oracle {summary['systems']['t2_oracle']['J']:.4f}, "
          f"T2 predicted {summary['systems']['t2_predicted']['J']:.4f}); gate accuracy {gate['t3_gate_accuracy']:.3f} "
          f"vs classifier {gate['t2_classifier_accuracy']:.3f}; collapsed={summary['collapse']['collapsed']}; "
          f"results in {out_dir}")
    return out_dir


def run_evaluation(cfg: dict, checkpoint=None, final_test: bool = False) -> Path:
    """Entry point of scripts/evaluate.py (plan 10.1). `checkpoint` is the path of a Task 3 checkpoint (.pt)."""
    if checkpoint is None:
        raise ValueError("Task 3 evaluation needs a checkpoint: pass --ckpt <path to a soft_moe checkpoint>")
    return evaluate_task3(cfg, checkpoint, final_test=final_test)
