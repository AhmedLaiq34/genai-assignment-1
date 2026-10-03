"""Run an Optuna study for a task (CONTRACTS 3.13). Only parses arguments, then calls run_study(cfg)."""
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
    ap.add_argument("--n-trials", type=int, default=None, help="override n_trials (default: from the config)")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs_per_trial")
    ap.add_argument("--resume", action="store_true",
                    help="accepted for symmetry: a study always continues (load_if_exists=True)")
    ap.add_argument("--dry-run", action="store_true",
                    help="tiny study in artifacts/dryrun with a distinct study name; real studies are untouched")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    overrides = list(args.set)
    if args.n_trials is not None:
        overrides.append(f"n_trials={args.n_trials}")
    if args.epochs is not None:
        overrides.append(f"epochs_per_trial={args.epochs}")
    cfg = build_config(args.task, args.config, args.device_profile, overrides)
    if args.dry_run:
        from genai.common.paths import ARTIFACTS
        from genai.tasks.task1.tune import dry_run_overrides
        cfg = dry_run_overrides(cfg, ARTIFACTS / "dryrun")
        cfg["trial_train_subset"] = cfg["trial_train_subset"] or 256
        cfg["train"]["val_subset"] = cfg["train"]["val_subset"] or 64
    out = entry_point(args.task, "tune", "run_study")(cfg)
    print("study directory:", out)


if __name__ == "__main__":
    main()
