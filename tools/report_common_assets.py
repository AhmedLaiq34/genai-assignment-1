"""Report assets shared by Tasks 1 and 2 (and later 3 and 4), built only from the repository's own files.

    python tools/report_common_assets.py

Writes (CSV + LaTeX tabular) to report/tables/:
    corruption_config        the complete corruption configuration (training distributions and fixed test severities)
    dataset_split            image counts, manifest row counts and sha256 of the split and manifests
    search_spaces            the Optuna search space of every study (Task 1 v2, Task 2 classifier, Task 2 specialists)
    final_configs            the hyperparameters of the final models (best trial values and fixed settings)
and to report/figures/:
    t2_pipeline.png          hard-routed restoration pipeline diagram (classifier -> argmax -> identity / three specialists)
Reads constants, YAML configs, data/splits and data/manifests. No training, no test-set images.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from genai.common import constants as C  # noqa: E402

TABLES, FIGURES = ROOT / "report" / "tables", ROOT / "report" / "figures"


def to_tex(df: pd.DataFrame, path: Path, caption: str) -> None:
    """Plain LaTeX tabular (pandas' to_latex needs jinja2, which is not installed)."""
    esc = lambda v: (f"{v:.4g}" if isinstance(v, float) else str(v)).replace("_", r"\_").replace("%", r"\%")  # noqa: E731
    lines = [r"\begin{tabular}{" + "l" * len(df.columns) + "}", r"\hline",
             " & ".join(esc(c) for c in df.columns) + r" \\", r"\hline"]
    lines += [" & ".join(esc(v) for v in row) + r" \\" for row in df.itertuples(index=False)]
    lines += [r"\hline", r"\end{tabular}"]
    path.write_text("% " + caption + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def save(df: pd.DataFrame, name: str, caption: str) -> None:
    df.to_csv(TABLES / f"{name}.csv", index=False)
    to_tex(df, TABLES / f"{name}.tex", caption)


def corruption_config() -> None:
    rows = [
        ["clean", "training", "no corruption", "-", "-"],
        ["salt_pepper", "training", "corruption probability p ~ U(0.02, 0.15); pixels replaced by black or white with equal probability (per pixel, all 3 channels together)",
         f"p in {C.SALT_P_RANGE}", "new draw at every image load"],
        ["gaussian_blur", "training", "kernel size from {3, 5, 7} (uniform), sigma ~ U(0.5, 2.5), reflect padding",
         f"kernel in {C.BLUR_KERNELS}, sigma in {C.BLUR_SIGMA_RANGE}", "new draw at every image load"],
        ["occlusion", "training", "1 to 3 black rectangles; union coverage ~ U(0.10, 0.35); random positions; rejection sampling until the union is in range",
         f"n in {C.OCC_N_RECTS}, coverage in {C.OCC_COVERAGE_RANGE}", "new draw at every image load"],
    ]
    for level, p, (k, s), (n, cov) in zip(C.SEVERITY_NAMES, C.TEST_SALT_P, C.TEST_BLUR, C.TEST_OCC):
        rows += [["salt_pepper", f"fixed {level}", f"p = {p}", f"p = {p}", "validation/test manifest"],
                 ["gaussian_blur", f"fixed {level}", f"kernel {k}, sigma {s}", f"({k}, {s})", "validation/test manifest"],
                 ["occlusion", f"fixed {level}", f"{n} non-overlapping rectangle(s), union about {int(cov * 100)}% (+/- {int(C.TEST_OCC_TOLERANCE * 100)} pp)",
                  f"n = {n}, coverage = {cov}", "validation/test manifest"]]
    save(pd.DataFrame(rows, columns=["condition", "split / level", "definition", "parameters", "when sampled"]),
         "corruption_config", "Complete corruption configuration (training distributions and fixed test severities)")


def dataset_split() -> None:
    split = json.loads((ROOT / "data/splits/pets_split.json").read_text(encoding="utf-8"))
    count = lambda name: sum(1 for _ in open(ROOT / "data/manifests" / name, encoding="utf-8"))  # noqa: E731
    sha = lambda name: (ROOT / "data/manifests" / (name + ".sha256")).read_text(encoding="utf-8").split()[0]  # noqa: E731
    n_tv = len(split["train"]) + len(split["val"])
    rows = [
        ["official trainval", n_tv, "-", "-"],
        ["train (80%, seed 42)", len(split["train"]), "-", split["sha256"]["train"]],
        ["validation (20%, seed 42)", len(split["val"]), count("pets_val_manifest.jsonl"), split["sha256"]["val"]],
        ["official test (locked until the final evaluation)", len(split["test"]["ids"]), count("pets_test_manifest.jsonl"), split["sha256"]["test"]],
        ["validation manifest file", "-", "4 rows per image (clean + 3 corruptions)", sha("pets_val_manifest.jsonl")],
        ["test manifest file", "-", "10 rows per image (clean + 3 types x 3 severities)", sha("pets_test_manifest.jsonl")],
    ]
    save(pd.DataFrame(rows, columns=["part", "images", "manifest rows", "sha256"]), "dataset_split",
         "Oxford-IIIT Pet split (seed 42, 128x128 RGB) and corruption manifests with their sha256")


def search_spaces() -> None:
    rows = []
    for study, path in (("t1_universal_v2 (Task 1)", "configs/task1_universal.yaml"),
                        ("t2_classifier (Task 2)", "configs/task2_classifier.yaml"),
                        ("t2_specialist_shared (Task 2)", "configs/task2_specialist.yaml")):
        cfg = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
        for p in cfg["tuned_params"]:
            space = f"{p['low']} to {p['high']}" + (" (log scale)" if p.get("log") else "") if p["type"] == "float" else "{" + ", ".join(map(str, p["choices"])) + "}"
            rows.append([study, p["name"], p["type"], space, cfg["n_trials"], cfg["epochs_per_trial"], cfg["pruner"]["name"],
                         f"startup {cfg['pruner']['n_startup_trials']}, warm-up {cfg['pruner']['n_warmup_steps']}", cfg["sampler"]["name"]])
    save(pd.DataFrame(rows, columns=["study", "parameter", "type", "range / choices", "n_trials (budget)", "epochs per trial", "pruner", "pruner settings", "sampler"]),
         "search_spaces", "Optuna search spaces (budgets are the configured ones; completed counts are in the study summaries)")


def final_configs() -> None:
    rows = []
    for name, path in (("Task 1 universal AE", "configs/task1_final_v2.yaml"), ("Task 2 classifier", "configs/task2_classifier_final.yaml"),
                       ("Task 2 specialists (salt, blur, occlusion)", "configs/task2_specialist_final.yaml")):
        cfg = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
        t, m = cfg["train"], cfg["model"]
        rows.append([name, json.dumps(m), t["lr"], t["batch_size"], t.get("alpha", "-"), t["weight_decay"], t["epochs"],
                     t.get("scheduler"), "AdamW", "yes" if t.get("amp") else "no", cfg["seed"]])
    save(pd.DataFrame(rows, columns=["model", "architecture arguments", "lr", "batch", "alpha (L1 weight)", "weight decay", "epochs", "schedule", "optimiser", "mixed precision", "seed"]),
         "final_configs", "Hyperparameters of the final models (best Optuna trial plus fixed settings)")


def pipeline_figure() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.9))
    ax.set_xlim(0, 11), ax.set_ylim(0, 4.9), ax.axis("off")

    def box(x, y, w, h, text, fc="#e8eaf6", ec="#3949ab"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec=ec, lw=1.5))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    def arrow(x0, y0, x1, y1, label=""):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.4, color="#444"))
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, label, ha="center", fontsize=8, color="#444")

    box(0.1, 1.7, 1.5, 0.8, "corrupted input\n3x128x128, [0,1]")
    box(2.1, 1.5, 2.6, 1.2, "CNN classifier\n4 conv blocks (16-32-64-128)\nGAP, dropout, linear", fc="#fff3e0", ec="#ef6c00")
    box(5.1, 1.7, 1.5, 0.8, "softmax, argmax\nr = argmax p_k")
    arrow(1.6, 2.1, 2.1, 2.1), arrow(4.7, 2.1, 5.1, 2.1, "logits")
    experts = [("r = clean: identity bypass\n(no restoration expert)", 3.45, "#f5f5f5", "#757575"),
               ("r = salt-and-pepper:\nspecialist A_salt", 2.35, "#e8f5e9", "#2e7d32"),
               ("r = blur:\nspecialist A_blur", 1.35, "#e8f5e9", "#2e7d32"),
               ("r = occlusion:\nspecialist A_occlusion", 0.35, "#e8f5e9", "#2e7d32")]
    for text, y, fc, ec in experts:
        box(7.2, y, 2.2, 0.7, text, fc=fc, ec=ec)
        arrow(6.6, 2.1, 7.2, y + 0.35)
        arrow(9.4, y + 0.35, 9.9, 2.1)
    box(9.9, 1.7, 1.0, 0.8, "restored\nimage")
    ax.text(5.5, 4.75, "Hard-routed restoration (oracle mode replaces r by the known corruption label)", ha="center", fontsize=10, weight="bold")
    fig.savefig(FIGURES / "t2_pipeline.png", dpi=160, bbox_inches="tight"), plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True), FIGURES.mkdir(parents=True, exist_ok=True)
    for fn in (corruption_config, dataset_split, search_spaces, final_configs, pipeline_figure):
        fn()
        print("done:", fn.__name__)


if __name__ == "__main__":
    main()
