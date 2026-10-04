"""Task 2 evaluation of the hard-routed system in BOTH modes (PDF page 5).

    evaluate_task2(cfg, checkpoints, final_test=False) -> output directory (Path)

    checkpoints = {"classifier": path, "salt": path, "blur": path, "occlusion": path}

* final_test=False (default): the VAL manifest. final_test=True: the locked TEST manifest
  (PetsManifestDataset logs the access to artifacts/test_access.log). Never done during development.
* Optional cfg["eval"] keys (all have defaults):
      max_rows    evaluate only the first N manifest rows (default: all). A manifest lists 4 (val) or
                  10 (test) rows per image in a row, so use a multiple of 4 / 10 to keep whole images.
      batch_size  images per forward pass (default 64)
      n_worst     how many worst misroutes are drawn in routing_failures_worst.png (default 8)

Oracle and predicted mode
-------------------------
For every manifest row the SAME corrupted input tensor goes through the system twice:
  oracle    routed by the true class id (clean -> identity bypass)
  predicted routed by the classifier's argmax
The predicted output is computed by re-using the oracle output wherever both modes pick the same
route and only re-running the images that were misrouted, so oracle and predicted numbers on rows
with equal routes are EXACTLY equal.
A "misroute" is a row whose predicted route differs from the oracle route (= a classifier error).

Files in <output_root>/eval/task2/<YYYYmmdd-HHMMSS>_<val|test>/
  per_image.csv                     LONG format: two rows per manifest row, one per `mode`
                                    (oracle / predicted). Columns: row, image_id, cond, severity,
                                    true_class, predicted_class (class ids 0..3), oracle_route,
                                    predicted_route (identity / salt / blur / occlusion), mode,
                                    route_used (the route of THIS row's mode), MAE, SSIM, PSNR, J
                                    (of this mode's output against the clean target), SSIM_input, J_input
                                    (the "do nothing" baseline: the corrupted INPUT scored against the clean
                                    target; a restoration that is worse than this is not restoring anything),
                                    p_clean, p_salt_pepper, p_gaussian_blur, p_occlusion.
                                    Long format makes "filter on mode" the only step to get a table.
  table_cond_severity_oracle.csv    same layout as Task 1's table_cond_severity.csv
  table_cond_severity_predicted.csv   (cond x severity, per-condition rows, overall row; MAE SSIM PSNR J SSIM_input J_input count)
  classifier_report.json            accuracy, macro P/R/F1, per class, confusion matrices
  confusion_normalised.csv / .png   row-normalised 4x4 confusion matrix of the classifier
  routing_failures.csv              every misroute: oracle_J, predicted_J, J_drop = predicted_J - oracle_J
                                    (positive = the routing error made the result worse), probabilities;
                                    sorted by J_drop, worst first
  misroute_confusion.csv            4x4 counts (true class x predicted class) of the MISROUTES ONLY, so
                                    the diagonal is 0; the full counts are in classifier_report.json
  routing_failures_worst.png        one row per case (the first n_worst rows of routing_failures.csv):
                                    target | input | oracle output | predicted output | |error| x4 of the
                                    predicted output. If there is no misroute, routing_failures_worst_NOTE.txt
                                    explains that instead.
  summary.json                      overall metrics of both modes, the input baseline (overall and per condition,
                                    with a flag whether the ORACLE output beats the input), classifier accuracy /
                                    macro-F1, misroute count, checkpoints used (sha256), manifest and split hashes
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid, save_image

from genai.common.checkpoint import sha256_file
from genai.common.constants import CLASS_NAMES
from genai.common.metrics import l1, objective_J, psnr, ssim
from genai.pets.dataset import PetsManifestDataset, resolve_data_paths
from genai.pets.split import split_sha256
from genai.tasks.task1.config import resolve
from genai.tasks.task1.evaluate import ERROR_GAIN, TEST_MANIFEST_NAME, VAL_MANIFEST_NAME
from genai.tasks.task2 import EXPERT_FOR_CLASS
from genai.tasks.task2.routing import HardRoutedSystem, describe_checkpoints, load_task2_models

METRICS = ["MAE", "SSIM", "PSNR", "J"]
BASELINE = ["SSIM_input", "J_input"]       # the corrupted input scored against the clean target (same for both modes)
PROB_COLUMNS = ["p_" + name for name in CLASS_NAMES]      # p_clean, p_salt_pepper, ...


# ======================================================================= running the system
def _metrics(out: torch.Tensor, clean: torch.Tensor) -> dict:
    """MAE, SSIM, PSNR and J of a batch of outputs against the clean targets (lists of floats)."""
    mae, s, p = l1(out, clean), ssim(out, clean), psnr(out, clean)
    return {"MAE": mae.tolist(), "SSIM": s.tolist(), "PSNR": p.tolist(), "J": objective_J(mae, s).tolist()}


@torch.no_grad()
def _run_both_modes(system: HardRoutedSystem, ds, n_rows: int, batch_size: int) -> pd.DataFrame:
    """Oracle and predicted routing over the first n_rows manifest rows. One WIDE row per manifest row."""
    device = system.device
    loader = DataLoader(Subset(ds, range(n_rows)), batch_size=batch_size, shuffle=False, num_workers=0)
    records, i = [], 0
    for corrupted, clean, cond, _sev in loader:
        corrupted, clean, cond = corrupted.to(device), clean.to(device), cond.to(device)

        probs = system.classify(corrupted)                       # (N,4) softmax
        predicted = probs.argmax(dim=1)                          # routing decision of the classifier

        oracle_out = system.restore_by_route(corrupted, cond)    # route = true class
        predicted_out = oracle_out.clone()                       # same where the routes agree ...
        wrong = predicted != cond
        if wrong.any():                                          # ... re-run only the misrouted images
            predicted_out[wrong] = system.restore_by_route(corrupted[wrong], predicted[wrong])

        m_oracle, m_pred = _metrics(oracle_out, clean), _metrics(predicted_out, clean)
        # baseline "do nothing": how far the corrupted INPUT already is from the clean target
        s_in = ssim(corrupted, clean)
        j_in = objective_J(l1(corrupted, clean), s_in)
        for k in range(len(cond)):
            row = ds.rows[i]
            record = {"row": i, "image_id": row["image_id"], "cond": row["cond_name"],
                      "severity": row["severity"] or "none",
                      "true_class": int(cond[k]), "predicted_class": int(predicted[k]),
                      "oracle_route": EXPERT_FOR_CLASS[int(cond[k])],
                      "predicted_route": EXPERT_FOR_CLASS[int(predicted[k])]}
            for c, column in enumerate(PROB_COLUMNS):
                record[column] = float(probs[k, c])
            record["SSIM_input"], record["J_input"] = float(s_in[k]), float(j_in[k])
            for metric in METRICS:
                record["oracle_" + metric] = m_oracle[metric][k]
                record["predicted_" + metric] = m_pred[metric][k]
            records.append(record)
            i += 1
    return pd.DataFrame(records)


def _to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Wide (one row per manifest row) -> long (one row per manifest row and mode), see module docstring."""
    shared = ["row", "image_id", "cond", "severity", "true_class", "predicted_class",
              "oracle_route", "predicted_route"]
    parts = []
    for mode in ("oracle", "predicted"):
        part = wide[shared].copy()
        part["mode"] = mode
        part["route_used"] = wide[mode + "_route"]
        for metric in METRICS:
            part[metric] = wide[f"{mode}_{metric}"]
        part[BASELINE] = wide[BASELINE]
        part[PROB_COLUMNS] = wide[PROB_COLUMNS]
        parts.append(part)
    long = pd.concat(parts, ignore_index=True)
    # sort by row, then mode ("oracle" < "predicted" alphabetically): the two rows of a pair sit together
    return long.sort_values(["row", "mode"], kind="stable").reset_index(drop=True)


# ======================================================================= tables
def _cond_severity_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per condition x severity + per-condition rows + an overall row.

    Same layout as Task 1's table_cond_severity.csv (the code is repeated here because Task 1 builds
    the table inline inside its run_evaluation).
    """
    columns = METRICS + BASELINE
    table = df.groupby(["cond", "severity"])[columns].mean().assign(count=df.groupby(["cond", "severity"]).size())
    per_cond = df.groupby("cond")[columns].mean().assign(count=df.groupby("cond").size())
    per_cond.index = pd.MultiIndex.from_product([per_cond.index, ["all"]])
    overall = df[columns].mean().to_frame().T.assign(count=len(df))
    overall.index = pd.MultiIndex.from_tuples([("overall", "all")])
    return pd.concat([table, per_cond, overall]).sort_index().rename_axis(["cond", "severity"])


# ======================================================================= routing failure analysis
def _routing_failures(wide: pd.DataFrame) -> pd.DataFrame:
    """Rows whose predicted route differs from the oracle route, worst J drop first."""
    failures = wide[wide["predicted_route"] != wide["oracle_route"]].copy()
    failures["J_drop"] = failures["predicted_J"] - failures["oracle_J"]       # > 0: the misroute hurt
    columns = ["row", "image_id", "cond", "severity", "true_class", "predicted_class", "oracle_route",
               "predicted_route", "oracle_J", "predicted_J", "J_input", "J_drop"] + PROB_COLUMNS
    return failures[columns].sort_values("J_drop", ascending=False).reset_index(drop=True)


def _misroute_confusion(failures: pd.DataFrame) -> pd.DataFrame:
    """4x4 counts of misroutes: rows = true class, columns = predicted class (diagonal stays 0)."""
    counts = pd.DataFrame(0, index=list(CLASS_NAMES), columns=list(CLASS_NAMES))
    for t, p in zip(failures["true_class"], failures["predicted_class"]):
        counts.iloc[int(t), int(p)] += 1
    counts.index.name = "true_class"
    return counts


@torch.no_grad()
def _draw_worst(system: HardRoutedSystem, ds, failures: pd.DataFrame, path: Path) -> None:
    """One image row per failure: target | input | oracle output | predicted output | |error| x4."""
    tiles = []
    for _, f in failures.iterrows():
        corrupted, clean, _c, _s = ds[int(f["row"])]
        x = corrupted.unsqueeze(0)
        oracle = system.restore_by_route(x, torch.tensor([int(f["true_class"])])).cpu()[0]
        predicted = system.restore_by_route(x, torch.tensor([int(f["predicted_class"])])).cpu()[0]
        err = ((predicted - clean).abs() * ERROR_GAIN).clamp(0, 1)         # error of the operational system
        tiles += [clean, corrupted, oracle, predicted, err]
    save_image(make_grid(torch.stack(tiles), nrow=5, padding=2), path)


# ======================================================================= entry point
def evaluate_task2(cfg: dict, checkpoints: dict, final_test: bool = False) -> Path:
    """Evaluate the hard-routed system in oracle and predicted mode (see module docstring)."""
    from genai.tasks.task2.classifier import classification_metrics, save_confusion_matrix   # lazy import

    eval_cfg = cfg.get("eval") or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system = HardRoutedSystem(load_task2_models(checkpoints, device), device)

    paths = resolve_data_paths(cfg)
    split_name = "test" if final_test else "val"
    manifest = Path(paths["manifests"]) / (TEST_MANIFEST_NAME if final_test else VAL_MANIFEST_NAME)
    ds = PetsManifestDataset(manifest, split=split_name, final_test=final_test, data_root=paths)
    n_rows = min(len(ds), int(eval_cfg["max_rows"])) if eval_cfg.get("max_rows") else len(ds)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = resolve(cfg["output_root"]) / "eval" / "task2" / f"{stamp}_{split_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- run both modes and write the per-image table + condition x severity tables
    wide = _run_both_modes(system, ds, n_rows, int(eval_cfg.get("batch_size", 64)))
    long = _to_long(wide)
    long.drop(columns=["row"]).to_csv(out_dir / "per_image.csv", index=False)
    for mode in ("oracle", "predicted"):
        _cond_severity_table(long[long["mode"] == mode]).to_csv(out_dir / f"table_cond_severity_{mode}.csv")

    # ---- classifier report (true class vs the class the classifier predicted)
    report = classification_metrics(wide["true_class"].to_numpy(), wide["predicted_class"].to_numpy())
    (out_dir / "classifier_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_confusion_matrix(report["confusion_normalised"], out_dir / "confusion_normalised.csv",
                          out_dir / "confusion_normalised.png")

    # ---- routing failure analysis
    failures = _routing_failures(wide)
    failures.to_csv(out_dir / "routing_failures.csv", index=False)
    _misroute_confusion(failures).to_csv(out_dir / "misroute_confusion.csv")
    if len(failures) > 0:
        _draw_worst(system, ds, failures.head(int(eval_cfg.get("n_worst", 8))), out_dir / "routing_failures_worst.png")
    else:
        (out_dir / "routing_failures_worst_NOTE.txt").write_text(
            "No misroutes: the classifier predicted the true class for every evaluated row, so there is "
            "no routing failure to draw.\n", encoding="utf-8")

    # ---- summary
    info = describe_checkpoints(checkpoints)
    summary = {
        "split": split_name,
        "n_rows": int(n_rows),
        "oracle": {m: float(wide["oracle_" + m].mean()) for m in METRICS},
        "predicted": {m: float(wide["predicted_" + m].mean()) for m in METRICS},
        "input_baseline": {m: float(wide[m].mean()) for m in BASELINE},
        # does the ORACLE-routed output (the specialist on its own corruption) beat doing nothing?
        "oracle_vs_input_by_condition": {
            cond: {"J": float(g["oracle_J"].mean()), "J_input": float(g["J_input"].mean()),
                   "beats_input": bool(g["oracle_J"].mean() < g["J_input"].mean())}
            for cond, g in wide.groupby("cond")},
        "classifier_accuracy": report["accuracy"],
        "classifier_macro_f1": report["macro_f1"],
        "n_misroutes": int(len(failures)),
        "misroute_rate": float(len(failures) / max(1, n_rows)),
        "mean_J_drop_on_misroutes": float(failures["J_drop"].mean()) if len(failures) else 0.0,
        "smoke": any(c["smoke"] for c in info.values()),      # true: numbers come from smoke/fixture models
        "checkpoints": info,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "split_sha256": split_sha256(ds.split),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"evaluated {n_rows} rows on {split_name}: oracle J {summary['oracle']['J']:.4f}, "
          f"predicted J {summary['predicted']['J']:.4f}, classifier accuracy {report['accuracy']:.3f}, "
          f"{len(failures)} misroutes; results in {out_dir}")
    return out_dir
