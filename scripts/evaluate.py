"""Evaluate a task on val, or on test only with --final-test (CONTRACTS 3.4)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genai.tasks.cli import build_config, entry_point  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=["t1", "t2cls", "t2spec", "t3", "t4"], required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device-profile", choices=["local", "kaggle", "colab"], default="local")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint (.pt) to evaluate. Task 2 (t2cls / t2spec) may omit it: the four checkpoints "
                         "come from cfg['eval']['checkpoints'], e.g. --set eval.checkpoints.classifier=<path> "
                         "--set eval.checkpoints.salt=<path> --set eval.checkpoints.blur=<path> "
                         "--set eval.checkpoints.occlusion=<path>")
    ap.add_argument("--final-test", action="store_true", help="opens the locked test set; logged")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    if args.ckpt is None and args.task not in ("t2cls", "t2spec"):
        ap.error("--ckpt is required for this task")
    cfg = build_config(args.task, args.config, args.device_profile, args.set)
    out = entry_point(args.task, "evaluate", "run_evaluation")(cfg, args.ckpt, final_test=args.final_test)
    print("results directory:", out)


if __name__ == "__main__":
    main()
