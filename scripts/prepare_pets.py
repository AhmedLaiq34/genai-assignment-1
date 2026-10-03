"""Download Oxford-IIIT Pet, cache 128x128, split, manifests, corruption grid (CONTRACTS 3.1, 3.3, 3.4).

Steps: download (torchvision) -> load each image (EXIF fix, RGB, bicubic 128x128) -> save uint8
cache -> write split JSON -> write val/test manifests (+ .sha256) -> save corruption example grid.
Run from the repo root:  .venv\\Scripts\\python.exe scripts/prepare_pets.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genai.common import constants as C  # noqa: E402
from genai.pets import corruptions as corr  # noqa: E402
from genai.pets import manifests as mf  # noqa: E402
from genai.pets import split as sp  # noqa: E402


def load_128(path) -> np.ndarray:
    """One image -> uint8 [128,128,3]. EXIF rotation fixed, RGB, direct bicubic resize (no crop)."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img = img.resize((C.IMG_SIZE, C.IMG_SIZE), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def cache_partition(raw_dir: Path, part: str, cache_dir: Path) -> list:
    """Download one official partition ("trainval" or "test") and cache it. Returns the ids."""
    from torchvision.datasets import OxfordIIITPet

    ds = OxfordIIITPet(root=str(raw_dir), split=part, target_types="category", download=True)
    paths = [Path(p) for p in ds._images]  # image file paths (torchvision keeps them in _images)
    ids = [p.stem for p in paths]  # e.g. "Abyssinian_1"
    images = np.stack([load_128(p) for p in paths])
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"{part}_images.npy", images)
    (cache_dir / f"{part}_ids.json").write_text(json.dumps(ids))
    print(f"cached {part}: {images.shape} {images.dtype}")
    return ids


def save_example_grid(cache_dir: Path, out_path: Path) -> None:
    """Rows: clean + 3 corruptions; columns: low / medium / high (fixed test severities)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    images = np.load(cache_dir / "trainval_images.npy", mmap_mode="r")
    clean = torch.from_numpy(np.array(images[0])).permute(2, 0, 1).float() / 255.0
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    for r, cond_id in enumerate((1, 2, 3)):
        axes[r, 0].imshow(clean.permute(1, 2, 0))
        axes[r, 0].set_title("clean" if r == 0 else "", fontsize=9)
        for c, sev in enumerate(C.SEVERITY_NAMES):
            params = corr.test_params(cond_id, sev, rng)
            out = corr.apply(clean, params, rng)
            axes[r, c + 1].imshow(out.permute(1, 2, 0).clamp(0, 1))
            axes[r, c + 1].set_title(f"{C.CLASS_NAMES[cond_id]} {sev}", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/data_pets.yaml")
    ap.add_argument("--device-profile", choices=["local", "kaggle", "colab"], default="local")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())

    raw_dir, cache_dir = ROOT / cfg["raw_dir"], ROOT / cfg["cache_dir"]
    trainval_ids = cache_partition(raw_dir, "trainval", cache_dir)
    test_ids = cache_partition(raw_dir, "test", cache_dir)

    split = sp.make_split(trainval_ids, test_ids, seed=cfg["seed"])
    sp.write_split(split, ROOT / cfg["split_file"])
    val_sha = mf.build_val_manifest(split, ROOT / cfg["val_manifest"])
    test_sha = mf.build_test_manifest(split, ROOT / cfg["test_manifest"])
    save_example_grid(cache_dir, ROOT / "report" / "figures" / "corruption_examples.png")

    print(f"trainval={len(trainval_ids)} test={len(test_ids)} "
          f"train={len(split['train'])} val={len(split['val'])}")
    print(f"val manifest rows={len(split['val']) * 4} sha256={val_sha}")
    print(f"test manifest rows={len(test_ids) * 10} sha256={test_sha}")
    print(f"split sha256={sp.split_sha256(split)}")


if __name__ == "__main__":
    main()
