"""Benchmark peak GPU memory and seconds/step; appends rows to docs/BENCHMARKS.md.

Example:  python scripts/benchmark.py --model t1 --batch 32 64 128 256 --amp both

For every (batch size, AMP on/off) it measures, on REAL cached data:
  * s_per_step     warmed-up seconds per training step (forward+loss+backward+optimizer),
                   with torch.cuda.synchronize() around the timed block;
  * peak_mem_MB    torch.cuda.max_memory_allocated() during those steps;
  * loader img/s   data-loading throughput of the real train DataLoader (runtime corruption);
  * val pass       seconds for one pass over the whole val manifest.
Only t1 is implemented now; the other model names raise NotImplementedError.
"""
import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from genai.common.paths import ROOT  # noqa: E402
from genai.common.timing import Timer  # noqa: E402

BENCH_FILE = ROOT / "docs" / "BENCHMARKS.md"


def loader_throughput(loader, n_batches: int) -> float:
    """Images per second the DataLoader delivers (the first batch, which starts the workers, is not timed)."""
    it = iter(loader)
    next(it)
    t0, images = time.perf_counter(), 0
    for _ in range(min(n_batches, len(loader) - 1)):   # stay inside one epoch
        images += len(next(it)[0])
    return images / (time.perf_counter() - t0)


def bench_t1(batch: int, amp: bool, cfg: dict, steps: int, warmup: int, loader_batches: int) -> dict:
    """Measure one (batch, amp) configuration of the Task 1 training step."""
    from genai.common.metrics import evaluate_restoration
    from genai.models.autoencoder import UniversalAE, count_parameters
    from genai.pets.dataset import resolve_data_paths
    from genai.tasks.task1.train import build_train_loader, build_val_loader, restoration_loss

    device = torch.device("cuda")
    paths = resolve_data_paths(cfg)
    cfg["train"]["batch_size"] = batch
    _ds, _sampler, loader = build_train_loader(cfg, paths, batch)
    result = {"batch": batch, "amp": "on" if amp else "off"}
    try:
        img_s = loader_throughput(loader, loader_batches)       # real corruption pipeline speed
        corrupted, clean = next(iter(loader))[:2]
        corrupted, clean = corrupted.to(device), clean.to(device)  # one fixed batch: isolates GPU time

        torch.manual_seed(0)
        model = UniversalAE.from_config(cfg["model"]).to(device).train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)

        def step():
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                out = model(corrupted)
            loss = restoration_loss(out.float(), clean, 0.8)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(warmup):                                  # cuDNN autotune etc. happen here
            step()
        with Timer(device) as t:                                 # Timer synchronises CUDA before and after
            for _ in range(steps):
                step()
        result.update(s_per_step=t.seconds / steps,
                      peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20, loader_img_s=img_s,
                      params=count_parameters(model))

        _vds, n_val, val_loader, _sha = build_val_loader(cfg, paths)
        with Timer(device) as tv:
            evaluate_restoration(model, val_loader, device)
        result.update(val_pass_s=tv.seconds, val_rows=n_val)
    except torch.cuda.OutOfMemoryError:
        result["oom"] = True
    finally:
        del loader
        gc.collect()
        torch.cuda.empty_cache()
    return result


def append_rows(rows: list, device_name: str, model_name: str, cfg: dict, steps: int) -> None:
    mc = cfg["model"]
    cfg_text = f"base_ch={mc['base_channels']} depth={mc['depth']} z={mc['bottleneck_dim']} drop={mc['dropout']}"
    lines = []
    for r in rows:
        if r.get("oom"):
            lines.append(f"| {device_name} | {model_name} | {r['batch']} | {r['amp']} | OOM | OOM | {cfg_text} |")
            continue
        note = (f"{cfg_text}; {r['params']:,} params; {steps} timed steps after warmup; "
                f"loader {r['loader_img_s']:.0f} img/s (workers={cfg.get('num_workers')}); "
                f"val pass {r['val_pass_s']:.1f} s ({r['val_rows']} rows); train step incl. SSIM loss")
        lines.append(f"| {device_name} | {model_name} | {r['batch']} | {r['amp']} | {r['peak_mb']:.0f} | "
                     f"{r['s_per_step']:.4f} | {note} |")
    with open(BENCH_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["t1", "t2cls", "t2spec", "t3", "t4"], required=True)
    ap.add_argument("--batch", type=int, nargs="+", required=True)
    ap.add_argument("--amp", choices=["on", "off", "both"], default="both")
    ap.add_argument("--device-profile", choices=["local", "kaggle", "colab"], default="local")
    ap.add_argument("--steps", type=int, default=20, help="timed steps")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--loader-batches", type=int, default=30)
    ap.add_argument("--no-write", action="store_true", help="print only, do not append to docs/BENCHMARKS.md")
    args = ap.parse_args()
    if args.model != "t1":
        raise NotImplementedError(f"benchmark for {args.model} is not implemented yet")
    if not torch.cuda.is_available():
        raise SystemExit("benchmark needs a CUDA GPU")

    from genai.tasks.task1.config import load_config
    cfg = load_config(None, args.device_profile)
    torch.backends.cudnn.benchmark = True
    device_name = f"{args.device_profile}: {torch.cuda.get_device_name(0)}"
    modes = {"on": [True], "off": [False], "both": [False, True]}[args.amp]
    rows = []
    for batch in args.batch:
        for amp in modes:
            r = bench_t1(batch, amp, cfg, args.steps, args.warmup, args.loader_batches)
            print(r)
            rows.append(r)
    if not args.no_write:
        append_rows(rows, device_name, args.model, cfg, args.steps)
        print("appended", len(rows), "rows to", BENCH_FILE)


if __name__ == "__main__":
    main()
