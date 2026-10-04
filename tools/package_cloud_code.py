"""Build dist/cloud_code/genai_code.zip: the code the Kaggle notebook needs (no GitHub required).

Upload the zip as a PRIVATE Kaggle Dataset (Kaggle unzips it, so the dataset folder looks like the repository).
Contents: src/, configs/, scripts/, tools/, pyproject.toml, requirements_kaggle.txt. No data, no models, no app.

    python tools/package_cloud_code.py
"""
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "cloud_code" / "genai_code.zip"
FOLDERS = ["src", "configs", "scripts", "tools"]
FILES = ["pyproject.toml", "requirements_kaggle.txt"]


def wanted(path: Path) -> bool:
    return "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo") and ".egg-info" not in str(path)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    names = []
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in FOLDERS:
            for f in sorted((ROOT / folder).rglob("*")):
                if f.is_file() and wanted(f.relative_to(ROOT)):
                    z.write(f, f.relative_to(ROOT).as_posix())
                    names.append(f.relative_to(ROOT).as_posix())
        for name in FILES:
            z.write(ROOT / name, name)
            names.append(name)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    (OUT.parent / "genai_code.zip.sha256").write_text(f"{digest}  genai_code.zip\n", encoding="utf-8")
    print(f"{OUT}  {OUT.stat().st_size / 1024:.0f} KB, {len(names)} files, sha256 {digest[:16]}...")


if __name__ == "__main__":
    main()
