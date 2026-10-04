"""Resolve FS2K pairs, stratified split, caches, pair audit grid, Kaggle package (CONTRACTS 3.7).

Thin wrapper: all the work is in genai.fs2k.prepare.main (so it is tested once).
Examples:
    python scripts/prepare_fs2k.py                     # pairs -> split -> caches -> audit grid
    python scripts/prepare_fs2k.py --package           # also dist/cloud_data/fs2k_data.zip (no test images)
    python scripts/prepare_fs2k.py --app-samples       # also ~6 display-only test photos for the app
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without `pip install -e .`


def main() -> None:
    from genai.fs2k.prepare import main as prepare_main
    prepare_main(sys.argv[1:])


if __name__ == "__main__":
    main()
