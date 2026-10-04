"""Benchmark peak GPU memory and seconds/step; appends rows to docs/BENCHMARKS.md.

Example:  python scripts/benchmark.py --model t1 --batch 32 64 128 256 --amp both

For every (batch size, AMP on/off) it measures, on REAL cached data:
  * s_per_step     warmed-up seconds per training step (forward+loss+backward+optimizer),
                   with torch.cuda.synchronize() around the timed block;
  * peak_mem_MB    torch.cuda.max_memory_allocated() during those steps;
  * loader img/s   data-loading throughput of the real train DataLoader (runtime corruption);
  * val pass       seconds for one pass over the whole val manifest.
t1, t2cls (classifier, balanced batches), t2spec (one specialist; the three specialists cost the
same on the GPU, only data loading differs a little) and t4 (cGAN: one D step + one G step on real FS2K
data; also loops over --base-channels) and t3 (soft mixture of experts: one warm-up step and one
joint step with the four Task 2 sources, see genai.tasks.task3.train.measure_step_times) are implemented.
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


def _timed_steps(step, device, steps: int, warmup: int) -> float:
    """Seconds per call of step(): `warmup` untimed calls (cuDNN autotune), then `steps` timed ones."""
    for _ in range(warmup):
        step()
    with Timer(device) as t:                                     # Timer synchronises CUDA before and after
        for _ in range(steps):
            step()
    return t.seconds / steps


def _train_loader(ds, policy: str, batch: int, cfg: dict):
    """DataLoader for a PetsTrainDataset with the sampler of `policy` (same call as the real training code)."""
    from genai.common.seed import worker_init_fn
    from genai.pets.samplers import make_batch_sampler
    sampler = make_batch_sampler(policy, len(ds), batch, cfg["seed"])
    nw = int(cfg.get("num_workers", 0))
    return torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=nw, worker_init_fn=worker_init_fn,
                                       persistent_workers=nw > 0)


def bench_t2cls(batch: int, amp: bool, cfg: dict, steps: int, warmup: int, loader_batches: int) -> dict:
    """Measure one (batch, amp) configuration of the Task 2 classifier training step (balanced batches)."""
    from genai.models.autoencoder import count_parameters
    from genai.models.classifier import CorruptionClassifier
    from genai.pets.dataset import PetsManifestDataset, PetsTrainDataset, resolve_data_paths

    assert batch % 4 == 0, "classifier batch sizes must be multiples of 4 (balanced batches)"
    device = torch.device("cuda")
    paths = resolve_data_paths(cfg)
    loader = _train_loader(PetsTrainDataset("train", "balanced_batch", data_root=paths, seed=cfg["seed"]),
                           "balanced_batch", batch, cfg)
    result = {"batch": batch, "amp": "on" if amp else "off"}
    try:
        img_s = loader_throughput(loader, loader_batches)
        corrupted, _clean, cond, _sev = next(iter(loader))
        corrupted, cond = corrupted.to(device), cond.to(device)   # one fixed real batch: isolates GPU time

        torch.manual_seed(0)
        model = CorruptionClassifier.from_config(cfg["model"]).to(device).train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)

        def step():
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(corrupted)
            loss = torch.nn.functional.cross_entropy(logits.float(), cond)   # loss in float32
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        s_per_step = _timed_steps(step, device, steps, warmup)
        result.update(s_per_step=s_per_step, peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20,
                      loader_img_s=img_s, params=count_parameters(model))

        # validation pass: a forward pass over the whole val manifest (all 4 conditions)
        val = PetsManifestDataset(Path(paths["manifests"]) / "pets_val_manifest.jsonl", "val", False, paths)
        val_loader = torch.utils.data.DataLoader(val, batch_size=128, shuffle=False, num_workers=0)
        model.eval()
        with Timer(device) as tv, torch.no_grad():
            for x, *_ in val_loader:
                model(x.to(device))
        result.update(val_pass_s=tv.seconds, val_rows=len(val))
    except torch.cuda.OutOfMemoryError:
        result["oom"] = True
    finally:
        del loader
        gc.collect()
        torch.cuda.empty_cache()
    return result


def bench_t2spec(batch: int, amp: bool, cfg: dict, steps: int, warmup: int, loader_batches: int) -> dict:
    """Measure one (batch, amp) configuration of a Task 2 specialist training step.

    The three specialists have the same model and loss, so their GPU cost is the same; the
    loader uses corruption class 1 (salt-and-pepper) and the val pass uses only that class's rows.
    """
    from torch.utils.data import DataLoader, Subset

    from genai.common.metrics import evaluate_restoration
    from genai.models.autoencoder import UniversalAE, count_parameters
    from genai.pets.dataset import PetsManifestDataset, PetsTrainDataset, resolve_data_paths
    from genai.tasks.task1.train import restoration_loss

    device = torch.device("cuda")
    paths = resolve_data_paths(cfg)
    cond_id = 1
    loader = _train_loader(PetsTrainDataset("train", f"fixed:{cond_id}", data_root=paths, seed=cfg["seed"]),
                           f"fixed:{cond_id}", batch, cfg)
    result = {"batch": batch, "amp": "on" if amp else "off"}
    try:
        img_s = loader_throughput(loader, loader_batches)
        corrupted, clean = next(iter(loader))[:2]
        corrupted, clean = corrupted.to(device), clean.to(device)

        torch.manual_seed(0)
        model = UniversalAE.from_config(cfg["model"]).to(device).train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)

        def step():
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                out = model(corrupted)
            loss = restoration_loss(out.float(), clean, float(cfg["train"].get("alpha", 0.8)))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        s_per_step = _timed_steps(step, device, steps, warmup)
        result.update(s_per_step=s_per_step, peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20,
                      loader_img_s=img_s, params=count_parameters(model))

        # validation pass: only the val-manifest rows of this specialist's condition
        val = PetsManifestDataset(Path(paths["manifests"]) / "pets_val_manifest.jsonl", "val", False, paths)
        rows = [i for i, r in enumerate(val.rows) if r["cond_id"] == cond_id]
        val_loader = DataLoader(Subset(val, rows), batch_size=128, shuffle=False, num_workers=0)
        with Timer(device) as tv:
            evaluate_restoration(model, val_loader, device)
        result.update(val_pass_s=tv.seconds, val_rows=len(rows))
    except torch.cuda.OutOfMemoryError:
        result["oom"] = True
    finally:
        del loader
        gc.collect()
        torch.cuda.empty_cache()
    return result


def bench_t4(batch: int, amp: bool, cfg: dict, steps: int, warmup: int, loader_batches: int) -> dict:
    """Measure one (batch, amp) configuration of the Task 4 cGAN step (one D step + one G step).

    Model size comes from cfg['model'] (main() loops over --base-channels). s_per_epoch is derived as
    ceil(train pairs / batch) * s_per_step + val pass (the number to use for the epoch budget, plan C17).
    """
    import math

    from torch.utils.data import DataLoader

    from genai.common.seed import worker_init_fn
    from genai.fs2k.dataset import FS2KDataset
    from genai.models.autoencoder import count_parameters
    from genai.models.cgan import Discriminator, Generator
    from genai.tasks.task4.train import d_step, g_step, validate

    device = torch.device("cuda")
    mc, tc = cfg["model"], cfg["train"]
    train_ds = FS2KDataset("train", cfg["data_root"], augment=True)
    nw = int(cfg.get("num_workers", 0))
    loader = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True, num_workers=nw,
                        worker_init_fn=worker_init_fn, persistent_workers=nw > 0)
    result = {"batch": batch, "amp": "on" if amp else "off", "base_channels": mc["base_channels"]}
    try:
        img_s = loader_throughput(loader, loader_batches)
        photo, sketch, style = (t.to(device) for t in next(iter(loader))[:3])   # one fixed batch: isolates GPU time

        torch.manual_seed(0)
        G = Generator.from_config(mc).to(device).train()
        D = Discriminator.from_config(mc).to(device).train()
        opt_g = torch.optim.Adam(G.parameters(), lr=float(tc["lr_g"]), betas=tuple(tc["betas"]))
        opt_d = torch.optim.Adam(D.parameters(), lr=float(tc["lr_d"]), betas=tuple(tc["betas"]))
        scaler_g = torch.amp.GradScaler("cuda", enabled=amp)
        scaler_d = torch.amp.GradScaler("cuda", enabled=amp)

        def step():
            d_step(G, D, opt_d, (photo, sketch, style), scaler_d)
            g_step(G, D, opt_g, (photo, sketch, style), float(tc["lambda_l1"]), scaler_g)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        s_per_step = _timed_steps(step, device, steps, warmup)
        result.update(s_per_step=s_per_step, peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20,
                      loader_img_s=img_s, params=count_parameters(G) + count_parameters(D))

        # validation pass over the whole val split (159 pairs), generator only
        val = FS2KDataset("val", cfg["data_root"], augment=False)
        val_loader = DataLoader(val, batch_size=32, shuffle=False, num_workers=0)
        validate(G, val_loader, device)          # untimed first pass: cuDNN autotune would inflate the first row
        with Timer(device) as tv:
            validate(G, val_loader, device)
        result.update(val_pass_s=tv.seconds, val_rows=len(val),
                      s_per_epoch=math.ceil(len(train_ds) / batch) * s_per_step + tv.seconds)
    except torch.cuda.OutOfMemoryError:
        result["oom"] = True
    finally:
        del loader
        gc.collect()
        torch.cuda.empty_cache()
    return result


def bench_t3(batch: int, amp: bool, cfg: dict, steps: int, warmup: int, loader_batches: int) -> dict:
    """Measure the Task 3 soft mixture of experts (gate + three experts, balanced batches). The timing code lives in
    genai.tasks.task3.train.measure_step_times so that the Kaggle runner's `prepare` stage measures the same thing."""
    from genai.tasks.task3.sources import find_sources
    from genai.tasks.task3.train import measure_step_times

    assert batch % 4 == 0, "Task 3 batch sizes must be multiples of 4 (balanced batches)"
    result = {"batch": batch, "amp": "on" if amp else "off"}
    try:
        t = measure_step_times(cfg, batch, amp, steps, warmup, sources=find_sources(cfg))
        # s_per_step is the joint step (the expensive one); the warm-up step goes into the note
        result.update(s_per_step=t["joint_s_per_step"], warmup_s_per_step=t["warmup_s_per_step"],
                      peak_mb=t["peak_mb"], params=t["params"], val_full_s=t["val_full_s"],
                      val_subset_s=t["val_subset_s"])
    except torch.cuda.OutOfMemoryError:
        result["oom"] = True
    finally:
        gc.collect()
        torch.cuda.empty_cache()
    return result


BENCH_FUNCTIONS = {"t1": bench_t1, "t2cls": bench_t2cls, "t2spec": bench_t2spec, "t3": bench_t3, "t4": bench_t4}


def append_rows(rows: list, device_name: str, model_name: str, cfg: dict, steps: int) -> None:
    mc = cfg["model"]
    if model_name == "t3":                  # Task 3: gate + three experts, the architectures come from the sources
        cfg_text = f"soft MoE tau={mc.get('tau')}"
    elif "style_dim" in mc:                   # Task 4 cGAN: base channels differ per row, so it goes into each row's note
        cfg_text = None
    elif "channels" in mc:                    # Task 2 classifier
        cfg_text = f"channels={mc['channels']} drop={mc['dropout']}"
    else:                                   # UniversalAE (Task 1 and the Task 2 specialists)
        cfg_text = f"base_ch={mc['base_channels']} depth={mc['depth']} z={mc['bottleneck_dim']} drop={mc['dropout']}"
    lines = []
    for r in rows:
        if cfg_text is None:
            row_cfg = f"base_ch={r['base_channels']} style_dim={mc['style_dim']} drop={mc['dropout']}"
        else:
            row_cfg = cfg_text
        if r.get("oom"):
            lines.append(f"| {device_name} | {model_name} | {r['batch']} | {r['amp']} | OOM | OOM | {row_cfg} |")
            continue
        if "val_full_s" in r:               # Task 3 row: joint step in the time column, warm-up step and val passes here
            note = (f"{row_cfg}; {r['params']:,} params; {steps} timed steps after warmup; joint step (gate + 3 experts "
                    f"fwd+bwd); warm-up step {r['warmup_s_per_step']:.4f} s; val pass {r['val_full_s']:.1f} s "
                    f"(full manifest), {r['val_subset_s']:.1f} s (subset)")
            lines.append(f"| {device_name} | {model_name} | {r['batch']} | {r['amp']} | {r['peak_mb']:.0f} | "
                         f"{r['s_per_step']:.4f} | {note} |")
            continue
        note = (f"{row_cfg}; {r['params']:,} params; {steps} timed steps after warmup; "
                f"loader {r['loader_img_s']:.0f} img/s (workers={cfg.get('num_workers')}); "
                f"val pass {r['val_pass_s']:.1f} s ({r['val_rows']} rows); "
                + (f"D step + G step; ~{r['s_per_epoch']:.1f} s/epoch incl. val" if "s_per_epoch" in r
                   else "train step incl. SSIM loss"))
        lines.append(f"| {device_name} | {model_name} | {r['batch']} | {r['amp']} | {r['peak_mb']:.0f} | "
                     f"{r['s_per_step']:.4f} | {note} |")
    BENCH_FILE.parent.mkdir(parents=True, exist_ok=True)   # docs/ is not in the Kaggle code zip
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
    ap.add_argument("--base-channels", type=int, nargs="+", default=None,
                    help="t4 only: base channel counts to time (default: 32 and 64, the study's choices)")
    ap.add_argument("--no-write", action="store_true", help="print only, do not append to docs/BENCHMARKS.md")
    args = ap.parse_args()
    if args.model not in BENCH_FUNCTIONS:
        raise NotImplementedError(f"benchmark for {args.model} is not implemented yet")
    if not torch.cuda.is_available():
        raise SystemExit("benchmark needs a CUDA GPU")

    from genai.tasks.cli import build_config
    # the model's own default config file (t4: the final-training config, which holds the model/train sections)
    cfg = build_config(args.model, "configs/task4_final.yaml" if args.model == "t4" else None, args.device_profile)
    bench = BENCH_FUNCTIONS[args.model]
    torch.backends.cudnn.benchmark = True
    device_name = f"{args.device_profile}: {torch.cuda.get_device_name(0)}"
    modes = {"on": [True], "off": [False], "both": [False, True]}[args.amp]
    rows = []
    channel_options = (args.base_channels or [32, 64]) if args.model == "t4" else [None]
    for base_channels in channel_options:
        if base_channels is not None:
            cfg["model"]["base_channels"] = base_channels
        for batch in args.batch:
            for amp in modes:
                r = bench(batch, amp, cfg, args.steps, args.warmup, args.loader_batches)
                print(r)
                rows.append(r)
    if not args.no_write:
        append_rows(rows, device_name, args.model, cfg, args.steps)
        print("appended", len(rows), "rows to", BENCH_FILE)


if __name__ == "__main__":
    main()
