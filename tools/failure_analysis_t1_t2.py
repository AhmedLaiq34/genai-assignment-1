"""Failure analysis for Task 1 (worst test cases) and Task 2 (routing failures).

Reads the saved test-set evaluation outputs (no model is run, CPU only) and measures which image
properties go with the failures: how much of the picture is near-black ("dark fraction": pixels with
mean RGB < 0.06) and how much fine detail it has ("texture": variance of the Laplacian of the grey image).
Writes small tables to report/tables/failure_analysis_*.csv that the report text refers to.

Run:  python tools/failure_analysis_t1_t2.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
T1_DIR = ROOT / "artifacts/eval/task1/20261004-1514_kaggle_t1_final_v2_test"
T2_DIR = ROOT / "artifacts/eval/task2/20261004-234854_test"
OUT = ROOT / "report/tables"
CACHE = ROOT / "data/cache/pets128"
DARK_LEVEL = 0.06          # a pixel counts as "dark" when its mean RGB is below this
DARK_IMAGE = 0.30          # an image counts as "dark-background" when this share of pixels is dark


def image_features() -> pd.DataFrame:
    """Dark fraction and texture of every official test image (from the 128x128 cache)."""
    ids = json.load(open(CACHE / "test_ids.json"))
    ids = ids if isinstance(ids, list) else ids.get("ids", ids)
    x = np.load(CACHE / "test_images.npy").astype(np.float32) / 255.0
    grey = x.mean(-1) if x.shape[-1] == 3 else x.mean(1)
    return pd.DataFrame({
        "image_id": ids,
        "dark_frac": (grey < DARK_LEVEL).mean(axis=(1, 2)),
        "texture": [ndimage.laplace(g).var() for g in grey],
    })


def task1(feat: pd.DataFrame) -> None:
    df = pd.read_csv(T1_DIR / "per_image.csv").merge(feat, on="image_id")
    rows = []
    for cond, sev in [("clean", "none"), ("gaussian_blur", "high"), ("occlusion", "medium"), ("occlusion", "high")]:
        s = df[(df.cond == cond) & (df.severity == sev)]
        rows.append({"cond": cond, "severity": sev, "n": len(s),
                     "corr_J_texture": s.J.corr(s.texture), "corr_J_dark_frac": s.J.corr(s.dark_frac),
                     "J_texture_q1_lowest": s[s.texture <= s.texture.quantile(.25)].J.mean(),
                     "J_texture_q4_highest": s[s.texture >= s.texture.quantile(.75)].J.mean(),
                     "J_dark_images": s[s.dark_frac >= DARK_IMAGE].J.mean(),
                     "J_other_images": s[s.dark_frac < DARK_IMAGE].J.mean(),
                     "n_dark_images": int((s.dark_frac >= DARK_IMAGE).sum())})
    pd.DataFrame(rows).round(4).to_csv(OUT / "failure_analysis_t1_properties.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())
    worst = (df[(df.cond == "occlusion") & (df.severity == "high")].sort_values("J", ascending=False).head(4))
    worst[["image_id", "J", "J_input", "SSIM", "PSNR", "dark_frac", "texture"]].round(4).to_csv(
        OUT / "failure_analysis_t1_worst4.csv", index=False)
    print(worst[["image_id", "J", "J_input", "dark_frac", "texture"]].round(3).to_string())
    print("texture percentiles 50/75/95/99:", np.percentile(feat.texture, [50, 75, 95, 99]).round(4))


def task2(feat: pd.DataFrame) -> None:
    pi = pd.read_csv(T2_DIR / "per_image.csv")
    pi = pi[pi["mode"] == "predicted"] if "mode" in pi else pi
    pi = pi.merge(feat, on="image_id")
    wrong = pi[pi.oracle_route != pi.predicted_route]
    print("predicted-mode rows:", len(pi), "misrouted:", len(wrong))
    rf = pd.read_csv(T2_DIR / "routing_failures.csv").merge(feat, on="image_id")
    names = {0: "clean", 1: "salt_pepper", 2: "gaussian_blur", 3: "occlusion"}
    rf["true"] = rf.true_class.map(names)
    rf["pred"] = rf.predicted_class.map(names)
    summ = (rf.groupby(["true", "pred"]).agg(n=("image_id", "size"), mean_J_drop=("J_drop", "mean"),
                                              mean_oracle_J=("oracle_J", "mean"), mean_predicted_J=("predicted_J", "mean"),
                                              mean_J_input=("J_input", "mean"), mean_dark_frac=("dark_frac", "mean"),
                                              mean_texture=("texture", "mean"), mean_p_pred=("p_occlusion", "mean"))
            .reset_index())
    summ.round(4).to_csv(OUT / "failure_analysis_t2_misroutes.csv", index=False)
    print(summ.round(3).to_string())
    print("test images mean dark_frac %.3f, mean texture %.4f" % (feat.dark_frac.mean(), feat.texture.mean()))
    # occlusion -> clean: does severity or darkness explain the miss?
    oc = rf[(rf.true == "occlusion") & (rf.pred == "clean")]
    print("occlusion->clean by severity:", oc.severity.value_counts().to_dict(), "| dark-image share:",
          round((oc.dark_frac >= DARK_IMAGE).mean(), 3), "vs all test images:",
          round((feat.dark_frac >= DARK_IMAGE).mean(), 3))
    cl = rf[(rf.true == "clean")]
    print("clean misrouted: dark-image share:", round((cl.dark_frac >= DARK_IMAGE).mean(), 3),
          "| mean p_occlusion where predicted occlusion:", round(cl[cl.pred == "occlusion"].p_occlusion.mean(), 3))
    # damage of the two main kinds of error
    for name, sub in [("clean->corrupted (specialist damages a clean image)", rf[rf.true == "clean"]),
                      ("corrupted->clean (restoration skipped)", rf[(rf.true != "clean") & (rf.pred == "clean")])]:
        print(name, "n=%d mean J_drop=%.3f median=%.3f" % (len(sub), sub.J_drop.mean(), sub.J_drop.median()))
    # overall effect on the system
    tot = pi.groupby("mode").J.mean() if "mode" in pi else None
    print("mean J (predicted-mode rows):", round(pi.J.mean(), 4))


if __name__ == "__main__":
    feat = image_features()
    print("=== Task 1 ===")
    task1(feat)
    print("=== Task 2 ===")
    task2(feat)
