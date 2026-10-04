"""Gate G2 of Task 3 (docs/TASK3_PLAN.md section E): measure the soft mixture BEFORE any study.

Read-only: the four Task 2 checkpoints are only loaded (their sha256 is checked first), the validation
manifest is the only data used (never the test set), nothing is written under models/.

Part 1 (always): the soft model built straight from the Task 2 files (the "epoch 0" model), on the first N
validation rows, for several temperatures tau:
    (a) J of the soft mixture vs the Task 2 hard routing (oracle and predicted) and the "do nothing" input,
        all on the SAME tensors;
    (b) mean top weight and the median (top-1 minus top-2) logit margin of the gate;
    (c) mean weight per true class.
Part 2 (--train, GPU recommended): 1 warm-up + 3 joint epochs at the PDF start values, in a throw-away run
folder under artifacts/quickcheck_t3/, printing val J, the mean weights per true class and the seconds per
step after every epoch. This is the learning curve that decides the pruner warm-up (plan B17).

    CUDA_VISIBLE_DEVICES=-1 python tools/t3_quick_check.py --rows 736          # CPU part only
    python tools/t3_quick_check.py --rows 2944 --train                         # GPU: all of it

The numbers are printed and written to artifacts/eval/task3/quick_check.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genai.common.checkpoint import sha256_file  # noqa: E402
from genai.common.constants import CLASS_NAMES  # noqa: E402
from genai.common.metrics import objective_J_batch  # noqa: E402
from genai.models.moe import SoftMoE  # noqa: E402
from genai.pets.dataset import PetsManifestDataset, resolve_data_paths  # noqa: E402
from genai.tasks.task2.routing import HardRoutedSystem, load_task2_models  # noqa: E402
from genai.tasks.task3 import BRANCH_NAMES, TASK2_SHA256  # noqa: E402
from genai.tasks.task3.sources import find_sources, verify_sources  # noqa: E402

OUT_FILE = ROOT / "artifacts" / "eval" / "task3" / "quick_check.json"


def part1(args, device, paths, sources) -> dict:
    """Soft copy of Task 2 at initialisation vs the Task 2 hard routing, for every tau."""
    ds = PetsManifestDataset(Path(paths["manifests"]) / "pets_val_manifest.jsonl", "val", False, paths)
    n = min(args.rows, len(ds))
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(n)), batch_size=args.batch, shuffle=False)

    soft = SoftMoE.load_from_task2(sources, tau=1.0).to(device).eval()   # verifies the four sha256 first
    hard = HardRoutedSystem(load_task2_models(sources, device), device)

    taus = list(args.taus)
    J = {"input": [], "t2_oracle": [], "t2_predicted": [], **{f"soft_tau{t:g}": [] for t in taus}}
    cond_all, margins = [], []
    top_w = {t: [] for t in taus}
    w_all = {t: [] for t in taus}
    with torch.no_grad():
        for x, clean, cond, _sev in loader:
            x, clean = x.to(device), clean.to(device)
            J["input"].append(objective_J_batch(x, clean).cpu())
            J["t2_oracle"].append(objective_J_batch(hard.route(x, cond, "oracle")[0], clean).cpu())
            J["t2_predicted"].append(objective_J_batch(hard.route(x, None, "predicted")[0], clean).cpu())
            for t in taus:
                soft.tau = float(t)
                x_hat, w, logits = soft(x)
                J[f"soft_tau{t:g}"].append(objective_J_batch(x_hat, clean).cpu())
                top_w[t].append(w.max(dim=1).values.cpu())
                w_all[t].append(w.cpu())
            top2 = logits.topk(2, dim=1).values                    # the logits do not depend on tau
            margins.append((top2[:, 0] - top2[:, 1]).cpu())
            cond_all.append(cond)

    cond_all = torch.cat(cond_all)
    J = {k: torch.cat(v) for k, v in J.items()}
    margin = torch.cat(margins)
    result = {"rows": n, "device": str(device), "taus": taus}
    result["J_overall"] = {k: float(v.mean()) for k, v in J.items()}
    result["J_by_cond"] = {k: {CLASS_NAMES[c]: float(v[cond_all == c].mean()) for c in range(4)} for k, v in J.items()}
    result["logit_margin"] = {"median": float(margin.median()), "mean": float(margin.mean()),
                              "p10": float(margin.quantile(0.1)), "p90": float(margin.quantile(0.9))}
    result["mean_top_weight"] = {f"{t:g}": float(torch.cat(top_w[t]).mean()) for t in taus}
    result["share_top_weight_above_0.99"] = {f"{t:g}": float((torch.cat(top_w[t]) > 0.99).float().mean()) for t in taus}
    result["mean_weights_by_true_class"] = {
        f"{t:g}": {CLASS_NAMES[c]: [round(float(v), 4) for v in torch.cat(w_all[t])[cond_all == c].mean(dim=0)]
                   for c in range(4)} for t in taus}
    # sigmoid-like estimate of the top weight for a margin m (plan B18): 1 / (1 + 3 exp(-m / tau))
    m = result["logit_margin"]["median"]
    result["estimated_top_weight_at_median_margin"] = {f"{t:g}": 1.0 / (1.0 + 3.0 * float(torch.exp(torch.tensor(-m / t))))
                                                       for t in (1, 2, 4, 5, 8)}
    return result


def print_part1(r: dict) -> None:
    print(f"\n=== Part 1: soft copy of Task 2 at init, {r['rows']} validation rows, device {r['device']} ===")
    print("(a) J overall (lower is better), same tensors:")
    for k, v in r["J_overall"].items():
        print(f"    {k:<16s} {v:.4f}")
    print("    J by true class:")
    print("    " + " " * 16 + "".join(f"{c:>14s}" for c in CLASS_NAMES))
    for k, per in r["J_by_cond"].items():
        print(f"    {k:<16s}" + "".join(f"{per[c]:>14.4f}" for c in CLASS_NAMES))
    lm = r["logit_margin"]
    print(f"(b) gate logit margin top1-top2: median {lm['median']:.2f}, mean {lm['mean']:.2f}, "
          f"p10 {lm['p10']:.2f}, p90 {lm['p90']:.2f}")
    for t in r["mean_top_weight"]:
        print(f"    tau {t:>4s}: mean top weight {r['mean_top_weight'][t]:.4f}, "
              f"share of rows with top weight > 0.99: {r['share_top_weight_above_0.99'][t]:.3f}")
    print("    estimated top weight at the median margin (B18 formula): "
          + ", ".join(f"tau {t}: {v:.3f}" for t, v in r["estimated_top_weight_at_median_margin"].items()))
    print(f"(c) mean weights {list(BRANCH_NAMES)} per true class:")
    for t, per in r["mean_weights_by_true_class"].items():
        print(f"    tau {t}")
        for c, w in per.items():
            print(f"        {c:<14s} {w}")


def part2(args, sources_dir: Path) -> dict:
    """1 warm-up + 3 joint epochs at the PDF start values (a real, short training in a throw-away folder)."""
    from genai.tasks.cli import build_config
    from genai.tasks.task3.train import run_training

    out_root = ROOT / "artifacts" / "quickcheck_t3"
    overrides = [f"output_root={out_root}", f"persist_root={out_root}", "run.desc=t3quick",
                 "train.warmup_epochs=1", "train.joint_epochs=3", "train.val_every_epochs=1",
                 "train.sample_every_epochs=1000", f"train.val_subset={args.rows}",
                 f"sources.dir={sources_dir}", f"model.tau={args.train_tau}", "num_workers=0"]
    cfg = build_config("t3", None, args.device_profile, overrides)
    t0 = time.time()
    run_dir = Path(run_training(cfg))
    seconds = time.time() - t0
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    print(f"\n=== Part 2: 1 warm-up + 3 joint epochs, tau {args.train_tau}, run folder {run_dir} ({seconds:.0f} s) ===")
    curve = []
    def f(v, spec=".4f"):          # the epoch-0 row has no training columns (None)
        return "-" if v is None else format(v, spec)

    for r in rows:
        print(f"    epoch {r['epoch']} ({r['stage']}): train_loss {f(r.get('train_loss'))}  "
              f"val_J {f(r.get('val_J'))}  epoch_seconds {f(r.get('epoch_seconds'), '.1f')}  "
              f"val_mean_w {[round(v, 3) for v in r.get('val_mean_w') or []]}  collapsed {r.get('collapsed')}")
        for c, w in zip(CLASS_NAMES, r.get("val_mean_w_by_class") or []):
            print(f"        {c:<14s} {[round(v, 3) for v in w]}")
        curve.append({k: r.get(k) for k in ("epoch", "stage", "val_J", "val_mean_w", "val_mean_w_by_class",
                                           "val_gate_accuracy", "epoch_seconds", "collapsed")})
    return {"run_dir": str(run_dir), "total_seconds": seconds, "tau": args.train_tau, "curve": curve}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=736, help="first N validation rows (736 = the trial subset)")
    ap.add_argument("--taus", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device-profile", choices=["local", "kaggle", "colab"], default="local")
    ap.add_argument("--train", action="store_true", help="part 2: 1 warm-up + 3 joint epochs (GPU recommended)")
    ap.add_argument("--train-tau", type=float, default=1.0, help="tau of part 2 (the PDF start is 1.0)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = resolve_data_paths(args.device_profile)
    sources = find_sources({"sources": {"dir": str(ROOT / "models" / "checkpoints")}})   # the real Task 2 files
    sources = {k: v for k, v in sources.items() if k != "t1" and v is not None}
    before = verify_sources(sources, TASK2_SHA256)
    print("sources verified:", {k: v[:12] for k, v in before.items()})

    result = {"part1": part1(args, device, paths, sources)}
    print_part1(result["part1"])
    if args.train:
        if device.type != "cuda":
            print("\n--train needs the GPU; skipped (no CUDA device visible)")
        else:
            result["part2"] = part2(args, Path(sources["classifier"]).parent)
    after = {k: sha256_file(p) for k, p in sources.items()}
    assert after == before, "a Task 2 checkpoint changed during the quick check"
    result["source_sha256"] = before
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nwritten:", OUT_FILE)


if __name__ == "__main__":
    main()
