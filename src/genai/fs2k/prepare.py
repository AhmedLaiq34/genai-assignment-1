"""FS2K data preparation: pairs -> split -> 128/143 PNG caches -> audit grid -> optional zip / app samples.

Run from the repo root (steps in this order, printing pair counts and style counts per split):
    python -m genai.fs2k.prepare            (or scripts/prepare_fs2k.py)
Options: --config, --device-profile, --package, --include-test, --app-samples.

Cache layout (cache_root = data/cache):
    cache_root/fs2k128/{photo,sketch}/<pair_id>.png   ALL pairs, resized straight to 128x128
    cache_root/fs2k143/{photo,sketch}/<pair_id>.png   official training pairs (train + val ids) at 143x143
Preprocessing = CONTRACTS C5: PIL exif_transpose -> RGB (photo) or L (sketch) -> bicubic resize
(no crop), the same as the backend's preprocess_bytes, so the app sees what validation sees.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps

from genai.fs2k import pairs as pr
from genai.fs2k import split as sp
from genai.fs2k.dataset import profile_data_root

CACHE128, CACHE143 = "fs2k128", "fs2k143"
SIZE_FINAL, SIZE_AUG = 128, 143


# --------------------------------------------------------------------------------------
# Loading and caching
# --------------------------------------------------------------------------------------


def load_photo(path, size: int = SIZE_FINAL) -> np.ndarray:
    """One photo -> uint8 [size,size,3]. EXIF rotation fixed, RGB, direct bicubic resize (no crop)."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.asarray(img.resize((size, size), Image.BICUBIC), dtype=np.uint8)


def load_sketch(path, size: int = SIZE_FINAL) -> np.ndarray:
    """One sketch -> uint8 [size,size], grayscale ("L"), same steps as load_photo."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("L")
    return np.asarray(img.resize((size, size), Image.BICUBIC), dtype=np.uint8)


def _save_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, format="PNG")


def build_cache(raw_root, split: dict, cache_root, raw_subdir: str | None = None) -> None:
    """Write fs2k128 (every pair) and fs2k143 (train + val pairs only) under cache_root.
    The raw paths come from split["pairs"] and are relative to the folder holding the anno files."""
    root = pr.find_raw_root(raw_root, "anno_train.json", raw_subdir)
    cache_root = Path(cache_root)
    official_train = set(split["train"]) | set(split["val"])  # the 143 cache never holds test images
    for pair_id, info in split["pairs"].items():
        photo_path, sketch_path = root / info["photo"], root / info["sketch"]
        for size, name in ((SIZE_FINAL, CACHE128), (SIZE_AUG, CACHE143)):
            if name == CACHE143 and pair_id not in official_train:
                continue
            _save_png(load_photo(photo_path, size), cache_root / name / "photo" / f"{pair_id}.png")
            _save_png(load_sketch(sketch_path, size), cache_root / name / "sketch" / f"{pair_id}.png")


# --------------------------------------------------------------------------------------
# Audit grid
# --------------------------------------------------------------------------------------


def save_pair_audit_grid(cache_root, split: dict, out_png, per_style: int = 8) -> None:
    """24 rows (per_style rows for each of the 3 styles): photo | sketch | "style k".

    Uses train + val pairs only (never test). In each style the rows are evenly spread over the
    sorted ids, so different photo folders show up. The student checks that each sketch belongs
    to its photo and that the style labels look consistent."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cache128 = Path(cache_root) / CACHE128
    pool = sorted(set(split["train"]) | set(split["val"]))
    rows = []
    for style in (0, 1, 2):
        ids = [i for i in pool if split["pairs"][i]["style"] == style]
        picks = np.linspace(0, len(ids) - 1, min(per_style, len(ids))).round().astype(int)
        rows += [(ids[k], style) for k in picks]
    fig, axes = plt.subplots(len(rows), 3, figsize=(4.8, 1.55 * len(rows)),
                             gridspec_kw={"width_ratios": [1, 1, 1.2]})
    for r, (pair_id, style) in enumerate(rows):
        photo = Image.open(cache128 / "photo" / f"{pair_id}.png")
        sketch = Image.open(cache128 / "sketch" / f"{pair_id}.png")
        axes[r, 0].imshow(photo)
        axes[r, 1].imshow(sketch, cmap="gray", vmin=0, vmax=255)
        axes[r, 2].text(0.0, 0.5, f"style {style} (UI Style {style + 1})\n{pair_id}", va="center", fontsize=8)
        for ax in axes[r]:
            ax.axis("off")
    axes[0, 0].set_title("photo", fontsize=9)
    axes[0, 1].set_title("sketch", fontsize=9)
    axes[0, 2].set_title("style", fontsize=9)
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=80)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Package for the cloud, app samples
# --------------------------------------------------------------------------------------


def package(data_root, out_zip, include_test: bool = False):
    """Zip splits/fs2k_split.json + the cached images of both caches (paths relative to data_root).

    Test images go in ONLY with include_test=True (the split file lists test ids, not images)."""
    data_root = Path(data_root)
    split = sp.load_split(data_root / "splits" / "fs2k_split.json")
    ids = set(split["train"]) | set(split["val"])
    if include_test:
        ids |= set(split["test"])
    files = [data_root / "splits" / "fs2k_split.json"]
    for name in (CACHE128, CACHE143):
        for kind in ("photo", "sketch"):
            for p in sorted((data_root / "cache" / name / kind).glob("*.png")):
                if p.stem in ids:
                    files.append(p)
    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            arc = p.relative_to(data_root).as_posix()
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {arc}")
            z.write(p, arc)
        z.writestr("CONTENTS.sha256", "\n".join(lines) + "\n")
    return out_zip


def export_app_samples(cache_root, split: dict, out_dir, n: int = 6) -> list:
    """Write n display-only 128 px TEST photos as <pair_id>.png into out_dir (created if needed).

    Deterministic: sorted ids, taking them style by style in turn (2 per style for n = 6, as long
    as the style has that many). The student agreed to these few test photos living in the repo.
    Returns the list of ids written."""
    cache128 = Path(cache_root) / CACHE128
    by_style = {s: sorted(i for i in split["test"] if split["pairs"][i]["style"] == s) for s in (0, 1, 2)}
    chosen, k = [], 0
    while len(chosen) < n and any(k < len(v) for v in by_style.values()):
        for s in (0, 1, 2):
            if k < len(by_style[s]) and len(chosen) < n:
                chosen.append(by_style[s][k])
        k += 1
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for pair_id in chosen:
        Image.open(cache128 / "photo" / f"{pair_id}.png").save(out_dir / f"{pair_id}.png", format="PNG")
    return chosen


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------


def _under(data_root: Path, cfg_path: str, root: Path) -> Path:
    """Config paths are repo-relative ("data/cache/..."). For a device profile with another data
    root, the leading "data" is replaced by that root; anything else stays repo-relative."""
    p = Path(cfg_path)
    if p.is_absolute():
        return p
    if p.parts[0] == "data":
        return data_root.joinpath(*p.parts[1:])
    return root / p


def main(argv=None) -> None:
    from genai.common.paths import ROOT

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data_fs2k.yaml")
    ap.add_argument("--device-profile", choices=["local", "kaggle", "colab"], default="local")
    ap.add_argument("--package", action="store_true", help="also write the cloud zip (no test images unless --include-test)")
    ap.add_argument("--include-test", action="store_true", help="put the test images in the zip (final evaluation only)")
    ap.add_argument("--app-samples", action="store_true", help="export ~6 display-only test photos for the app")
    args = ap.parse_args(argv)

    cfg_file = Path(args.config)
    cfg = yaml.safe_load((cfg_file if cfg_file.is_absolute() else ROOT / cfg_file).read_text(encoding="utf-8"))
    data_root = profile_data_root(args.device_profile)
    rules = cfg.get("pairs", {})
    rule_kwargs = {k: v for k, v in {
        "style_field": rules.get("style_field"),
        "extensions": tuple(rules["extensions"]) if "extensions" in rules else None,
        "photo_template": rules.get("photo_template"),
        "sketch_template": rules.get("sketch_template"),
    }.items() if v is not None}

    raw_root = _under(data_root, cfg["raw_dir"], ROOT)
    subdir = cfg.get("raw_subdir")
    train_pairs = pr.resolve_pairs(raw_root, cfg["annotations"]["train"], **rule_kwargs)
    test_pairs = pr.resolve_pairs(raw_root, cfg["annotations"]["test"], **rule_kwargs)
    print(f"resolved pairs: train {len(train_pairs)}, test {len(test_pairs)}")

    split = sp.make_split(train_pairs, test_pairs, seed=cfg["seed"], val_fraction=cfg["val_fraction"])
    split_file = _under(data_root, cfg["split_file"], ROOT)
    sp.save_split(split, split_file)
    print(f"wrote {split_file}")
    for name in ("train", "val", "test"):
        print(f"  {name}: {len(split[name])} pairs, style counts {split['style_counts'][name]}")

    cache_root = _under(data_root, cfg["cache_dir"], ROOT).parent  # .../cache (holds fs2k128 and fs2k143)
    if _under(data_root, cfg["cache_dir"], ROOT).name != CACHE128 or _under(data_root, cfg["cache143_dir"], ROOT).parent != cache_root:
        raise ValueError("cache_dir / cache143_dir in the config must be <cache_root>/fs2k128 and <cache_root>/fs2k143")
    build_cache(raw_root, split, cache_root, subdir)
    print(f"cached images under {cache_root}")

    grid = _under(data_root, cfg["audit_grid"], ROOT)
    save_pair_audit_grid(cache_root, split, grid)
    print(f"wrote audit grid {grid}")

    if args.package:
        out = package(data_root, _under(data_root, cfg["package_out"], ROOT), include_test=args.include_test)
        print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, include_test={args.include_test})")
    if args.app_samples:
        chosen = export_app_samples(cache_root, split, ROOT / "app" / "backend" / "samples_sketch", n=6)
        print(f"wrote {len(chosen)} app sample photos: {chosen}")


if __name__ == "__main__":
    main(sys.argv[1:])
