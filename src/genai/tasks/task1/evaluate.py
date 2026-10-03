"""Task 1 evaluation. Implements CONTRACTS 3.4 (test lock), 3.6 and plan section P5.

Entry point:  run_evaluation(cfg, checkpoint, final_test=False) -> output directory (Path)

* final_test=False (default): evaluates the VAL manifest.
* final_test=True: evaluates the locked TEST manifest (PetsManifestDataset logs the access to
  artifacts/test_access.log). Only the final evaluation step may use this.

Outputs (<output_root>/eval/task1/<run_id>_<val|test>/):
  per_image.csv          one row per (image, condition): MAE, SSIM, PSNR, J
  table_cond_severity.csv  mean metrics per condition x severity (clean, salt, blur, occlusion x low/medium/high)
  summary.json           overall numbers and the checkpoint used
  selected.csv           the 12 representative and 4 worst examples that were drawn
  representative_12.png / worst_4.png   rows of: target | input | output | |error| x4
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from genai.common.checkpoint import load_checkpoint
from genai.common.metrics import l1, objective_J, psnr, ssim
from genai.models.autoencoder import UniversalAE
from genai.pets.dataset import PetsManifestDataset, resolve_data_paths
from genai.tasks.task1.config import resolve

TEST_MANIFEST_NAME = "pets_test_manifest.jsonl"
VAL_MANIFEST_NAME = "pets_val_manifest.jsonl"
ERROR_GAIN = 4.0     # |error| is multiplied by this before drawing so small errors are visible


@torch.no_grad()
def _per_image_metrics(model, ds, device, batch_size: int = 128) -> pd.DataFrame:
    """Run the model over the whole manifest dataset in order; one row per manifest row."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    rows, i = [], 0
    for corrupted, clean, _cond, _sev in loader:
        out = model(corrupted.to(device)).cpu()
        mae, s, p = l1(out, clean), ssim(out, clean), psnr(out, clean)
        j = objective_J(mae, s)
        for k in range(len(mae)):
            r = ds.rows[i]
            rows.append({"row": i, "image_id": r["image_id"], "cond": r["cond_name"],
                         "severity": r["severity"] or "none", "MAE": mae[k].item(),
                         "SSIM": s[k].item(), "PSNR": p[k].item(), "J": j[k].item()})
            i += 1
    return pd.DataFrame(rows)


def select_examples(df: pd.DataFrame) -> pd.DataFrame:
    """Automatic selection: 12 representative + 4 worst cases.

    Representative = for each of the 4 conditions, the 3 rows whose J is closest to that
    condition's median J (typical behaviour, not cherry-picked). Worst = the 4 highest-J rows
    among corrupted inputs, at most one per image so the failures are different pictures.
    """
    picks = []
    for cond, group in df.groupby("cond"):
        med = group["J"].median()
        near = group.assign(dist=(group["J"] - med).abs()).sort_values("dist").head(3)
        picks.append(near.assign(group="representative"))
    worst = df[df["cond"] != "clean"].sort_values("J", ascending=False)
    worst = worst.drop_duplicates("image_id").head(4)
    picks.append(worst.assign(group="worst"))
    return pd.concat(picks).drop(columns=["dist"], errors="ignore")[
        ["group", "row", "image_id", "cond", "severity", "MAE", "SSIM", "PSNR", "J"]]


@torch.no_grad()
def _draw(model, ds, row_ids, device, path: Path) -> None:
    """One image row per example: target | input | output | |error|*gain."""
    tiles = []
    for i in row_ids:
        corrupted, clean, _c, _s = ds[int(i)]
        out = model(corrupted.unsqueeze(0).to(device)).cpu()[0]
        err = ((out - clean).abs() * ERROR_GAIN).clamp(0, 1)
        tiles += [clean, corrupted, out, err]
    save_image(make_grid(torch.stack(tiles), nrow=4, padding=2), path)


def run_evaluation(cfg: dict, checkpoint: str, final_test: bool = False) -> Path:
    """Evaluate a Task 1 checkpoint (see module docstring)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(checkpoint)
    model = UniversalAE.from_config(ckpt["config"]["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()                                   # dropout off, BatchNorm uses running stats

    paths = resolve_data_paths(cfg)
    split_name = "test" if final_test else "val"
    manifest = Path(paths["manifests"]) / (TEST_MANIFEST_NAME if final_test else VAL_MANIFEST_NAME)
    ds = PetsManifestDataset(manifest, split=split_name, final_test=final_test, data_root=paths)

    run_id = ckpt["config"].get("run_id", Path(checkpoint).parent.name)
    out_dir = resolve(cfg["output_root"]) / "eval" / "task1" / f"{run_id}_{split_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _per_image_metrics(model, ds, device)
    df.drop(columns=["row"]).to_csv(out_dir / "per_image.csv", index=False)

    # condition x severity table (+ per-condition and overall rows)
    metrics = ["MAE", "SSIM", "PSNR", "J"]
    table = df.groupby(["cond", "severity"])[metrics].mean().assign(count=df.groupby(["cond", "severity"]).size())
    per_cond = df.groupby("cond")[metrics].mean().assign(count=df.groupby("cond").size())
    per_cond.index = pd.MultiIndex.from_product([per_cond.index, ["all"]])
    overall = df[metrics].mean().to_frame().T.assign(count=len(df))
    overall.index = pd.MultiIndex.from_tuples([("overall", "all")])
    pd.concat([table, per_cond, overall]).sort_index().rename_axis(["cond", "severity"]).to_csv(out_dir / "table_cond_severity.csv")

    chosen = select_examples(df)
    chosen.to_csv(out_dir / "selected.csv", index=False)
    _draw(model, ds, chosen[chosen["group"] == "representative"]["row"], device, out_dir / "representative_12.png")
    _draw(model, ds, chosen[chosen["group"] == "worst"]["row"], device, out_dir / "worst_4.png")

    summary = {"checkpoint": str(checkpoint), "run_id": run_id, "split": split_name,
               "n_rows": len(df), "global_step": ckpt["global_step"],
               **{k: float(df[k].mean()) for k in metrics}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"evaluated {len(df)} rows on {split_name}: J {summary['J']:.4f} SSIM {summary['SSIM']:.4f} "
          f"PSNR {summary['PSNR']:.2f}; results in {out_dir}")
    return out_dir
