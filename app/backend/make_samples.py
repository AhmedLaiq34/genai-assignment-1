"""One-off script: write 8 clean sample PNGs from the official TEST cache into app/backend/samples/.

DISPLAY ONLY: these images are shown on the demo page. They are never used for training,
tuning or model selection. We commit the small PNGs so the Docker image does not need the
data volume. Run from the repo root:  python app/backend/make_samples.py
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache" / "pets128"
OUT = Path(__file__).resolve().parent / "samples"
N = 8


def main() -> None:
    ids = json.loads((CACHE / "test_ids.json").read_text())
    images = np.load(CACHE / "test_images.npy", mmap_mode="r")  # uint8 [N,128,128,3]
    OUT.mkdir(exist_ok=True)
    # Evenly spaced indices: the ids are sorted by breed, so this gives 8 different breeds.
    for i in np.linspace(0, len(ids) - 1, N).astype(int):
        Image.fromarray(np.asarray(images[i])).save(OUT / f"{ids[i]}.png")
        print("wrote", ids[i])


if __name__ == "__main__":
    main()
