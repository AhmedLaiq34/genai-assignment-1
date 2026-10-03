"""Train a task (CONTRACTS 3.8, 3.12). Only parses arguments, then calls run_training(cfg, resume)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without `pip install -e .`

from genai.tasks.cli import build_config, entry_point  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=["t1", "t2cls", "t2spec", "t3", "t4"], required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device-profile", choices=["local", "kaggle", "colab"], default="local")
    ap.add_argument("--resume", default=None, help="path to ckpt_last.pt, or 'auto' for the newest one")
    ap.add_argument("--run-name", default=None, help="short description used in the run id")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. --set train.max_steps=50 (repeatable)")
    args = ap.parse_args()

    overrides = list(args.set) + ([f"run.desc={args.run_name}"] if args.run_name else [])
    cfg = build_config(args.task, args.config, args.device_profile, overrides)
    run_dir = entry_point(args.task, "train", "run_training")(cfg, resume=args.resume)
    print("run directory:", run_dir)


if __name__ == "__main__":
    main()
