"""Fetch seven ONNX models from a local folder or HTTPS release; verify SHA-256."""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path, entry):
    if path.stat().st_size != entry['size'] or sha256(path) != entry['sha256']:
        raise ValueError(f"Size or SHA-256 mismatch: {path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, default=ROOT / 'models/MANIFEST.json')
    ap.add_argument('--dest', type=Path, default=ROOT / 'models/onnx')
    ap.add_argument('--source', type=Path, help='Local folder containing the ONNX files')
    ap.add_argument('--release-base', help='HTTPS release download base URL')
    ap.add_argument('--verify-only', action='store_true')
    args = ap.parse_args()
    try:
        models = [r['onnx'] for r in json.loads(args.manifest.read_text()) if r.get('onnx')]
        if len(models) != 7:
            raise ValueError('Manifest must list seven ONNX models')
        pending = []
        for entry in models:
            name = entry['file']
            if Path(name).name != name or not name.endswith('.onnx'):
                raise ValueError('Invalid model filename')
            target = args.dest / name
            if target.exists():
                verify(target, entry)
                print(f'Verified: {name}')
                continue
            if args.verify_only:
                raise FileNotFoundError(f'Missing: {name}')
            origin = args.source / name if args.source else ((args.release_base.rstrip('/') + '/' + name) if args.release_base else entry.get('url'))
            if not origin:
                raise ValueError('URLs not published: use --source FOLDER or --release-base URL')
            if args.source:
                verify(origin, entry)
            elif not origin.startswith('https://'):
                raise ValueError('Only HTTPS downloads are supported')
            pending.append((entry, target, origin))
        if pending:
            args.dest.mkdir(parents=True, exist_ok=True)
        for entry, target, origin in pending:
            with tempfile.NamedTemporaryFile(dir=args.dest, suffix='.download', delete=False) as stream:
                temporary = Path(stream.name)
            try:
                if args.source:
                    shutil.copyfile(origin, temporary)
                else:
                    request = urllib.request.Request(origin, headers={'User-Agent': 'genai-model-fetcher'})
                    with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as output:
                        shutil.copyfileobj(response, output)
                verify(temporary, entry)
                if target.exists():
                    verify(target, entry)
                else:
                    temporary.rename(target)
                print(f'Installed and verified: {entry["file"]}')
            finally:
                temporary.unlink(missing_ok=True)
        print('All seven inference models verified.')
    except (OSError, ValueError, KeyError) as error:
        print(f'Model setup failed: {error}', file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
