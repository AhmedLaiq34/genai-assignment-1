"""Export models to ONNX and verify parity (CONTRACTS 3.9)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai.common.constants import ONNX_FILES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(ONNX_FILES), help="e.g. t1_universal")
    ap.add_argument("--ckpt", default=None, help="checkpoint to export (default: models/checkpoints/<model>_ae.pt)")
    ap.add_argument("--out-dir", default="models/onnx")
    ap.add_argument("--verify", action="store_true", help="also run the ONNX Runtime parity check")
    ap.add_argument("--tag", default="", help="label for the parity CSV row, e.g. smoke")
    args = ap.parse_args()

    from genai.export.onnx_export import export_model
    from genai.export.onnx_verify import verify_parity

    if args.ckpt is None:
        raise SystemExit("--ckpt is required (path to a ckpt_best.pt / ckpt_last.pt or a promoted checkpoint)")
    out = Path(args.out_dir) / ONNX_FILES[args.model]
    export_model(args.model, args.ckpt, out)
    meta = json.loads(Path(str(out) + ".meta.json").read_text(encoding="utf-8"))
    if meta["smoke"]:
        print("*** SMOKE MODEL: exported from a smoke run, for integration tests only. Never a final model. ***")
    if args.verify:
        tag = args.tag or ("smoke" if meta["smoke"] else "")
        result = verify_parity(args.model, args.ckpt, out, tag=tag)
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit("parity FAILED (max abs diff above tolerance)")


if __name__ == "__main__":
    main()
