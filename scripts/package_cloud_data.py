"""Zip the pet data for upload as a private Kaggle Dataset / Colab Drive copy (PLAN 10.4).

Output: dist/cloud_data/pets_data.zip containing
    data/cache/pets128/   (trainval only; add the test cache with --include-test)
    data/splits/
    data/manifests/
    CONTENTS.sha256       (sha256 of every file in the zip, `sha256sum -c` style)
Nothing is uploaded; this only builds the file.
"""
import argparse
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--include-test", action="store_true", help="also pack the test image cache (final evaluation only)")
    ap.add_argument("--out", default="dist/cloud_data/pets_data.zip")
    args = ap.parse_args()

    files = []
    for folder in ("data/cache/pets128", "data/splits", "data/manifests"):
        for p in sorted((ROOT / folder).rglob("*")):
            if not p.is_file() or p.name == ".gitkeep":
                continue
            if p.name.startswith("test_") and p.parent.name == "pets128" and not args.include_test:
                continue  # keep the locked test images out of training uploads
            files.append(p)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            arc = p.relative_to(ROOT).as_posix()
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            lines.append(f"{digest}  {arc}")
            z.write(p, arc)
        z.writestr("CONTENTS.sha256", "\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(files)} files, include_test={args.include_test})")


if __name__ == "__main__":
    main()
