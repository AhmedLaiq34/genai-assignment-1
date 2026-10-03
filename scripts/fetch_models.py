"""Download model files and check sha256 against models/MANIFEST.json."""
import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="models/MANIFEST.json")
    ap.add_argument("--dest", default="models/onnx")
    ap.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
