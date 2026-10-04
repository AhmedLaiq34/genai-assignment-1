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
    ap.add_argument("--ckpt", default=None, help="checkpoint to export (a ckpt_best.pt / ckpt_last.pt or a promoted .pt)")
    ap.add_argument("--out-dir", default="models/onnx")
    ap.add_argument("--verify", action="store_true", help="also run the ONNX Runtime parity check")
    ap.add_argument("--tag", default="", help="label for the parity CSV row, e.g. smoke")
    ap.add_argument("--device-profile", choices=["local", "kaggle", "colab"], default="local",
                    help="where the val data is (configs/devices/<profile>.yaml); on Kaggle pass 'kaggle'")
    args = ap.parse_args()

    from genai.export.onnx_export import export_model
    from genai.export.onnx_verify import verify_parity
    # Task 2 models (t2_classifier, t2_salt, t2_blur, t2_occlusion) have their own exporter / parity check
    from genai.export.task2_export import (export_task2_model, verify_routing_parity,
                                           verify_task2_parity)

    if args.ckpt is None:
        raise SystemExit("--ckpt is required (path to a ckpt_best.pt / ckpt_last.pt or a promoted checkpoint)")
    is_t2 = args.model.startswith("t2_")
    if args.model == "t4_generator":
        # Task 4 generator (photo + style id -> sketch) has its own exporter / parity check
        from genai.export.task4_export import export_t4_generator, verify_t4_parity
        export, verify = export_t4_generator, verify_t4_parity
    elif args.model == "t3_soft_moe":
        # Task 3 soft mixture of experts (outputs `output` and `weights`) has its own exporter / parity check
        from genai.export.task3_export import export_t3, verify_t3_parity
        export, verify = export_t3, verify_t3_parity
    else:
        export, verify = (export_task2_model, verify_task2_parity) if is_t2 else (export_model, verify_parity)
    out = Path(args.out_dir) / ONNX_FILES[args.model]
    export(args.model, args.ckpt, out)
    meta = json.loads(Path(str(out) + ".meta.json").read_text(encoding="utf-8"))
    if meta["smoke"]:
        print("*** SMOKE MODEL: exported from a smoke run, for integration tests only. Never a final model. ***")
    if args.verify:
        tag = args.tag or ("smoke" if meta["smoke"] else "")
        # the device profile name tells the verifier where the val data is (never a hidden "local" default)
        result = verify(args.model, args.ckpt, out, tag=tag, data_root=args.device_profile)
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit("parity FAILED (max abs diff above tolerance)")
        if args.model == "t2_classifier":        # the ONNX classifier must also route like PyTorch does
            routing = verify_routing_parity(args.ckpt, out, data_root=args.device_profile)
            print(json.dumps(routing, indent=2))
            if not routing["passed"]:
                raise SystemExit("routing parity FAILED (ONNX classifier picks a different class)")


if __name__ == "__main__":
    main()
