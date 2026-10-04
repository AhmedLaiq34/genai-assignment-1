"""Task 4 training loop on the mini-FS2K fixture (no real data, CPU only).

Fixture: 14 train pairs per style -> after the 15% split 35 train / 7 val pairs, so batch 4 gives 8 steps per epoch.
Plan: docs/TASK4_PLAN.md section E.
"""
import json
import math
from pathlib import Path

import pytest
import torch
from fs2k_fixture import build_mini_fs2k_data
from torch import nn

from genai.common.checkpoint import load_checkpoint, promote, sha256_file
from genai.fs2k.dataset import FS2KDataset
from genai.models.cgan import Discriminator, Generator
from genai.tasks.task4 import config as t4cfg
from genai.tasks.task4 import train as t4train
from genai.tasks.task4.train import (d_step, fixed_sample_ids, g_step, make_scheduler, run_training,
                                     save_sample_grid, split_checkpoint, validate)

MODEL = {"base_channels": 8, "style_dim": 4, "dropout": 0.1, "num_styles": 3}
STEPS = 8          # 35 train pairs // batch 4


@pytest.fixture(scope="module")
def fs2k(tmp_path_factory):
    return build_mini_fs2k_data(tmp_path_factory.mktemp("fs2k"))


def make_cfg(fs2k, tmp_path, **train):
    cfg = t4cfg.load_config("configs/task4_final.yaml", "local")
    cfg.update(device="test", data_root=str(fs2k["data_root"]), output_root=str(tmp_path / "out"),
               persist_root=str(tmp_path / "out"), num_workers=0)
    cfg["model"].update(MODEL)
    defaults = dict(batch_size=4, epochs=2, schedule="constant", sample_every_epochs=5, snapshot_every_epochs=0)
    cfg["train"].update({**defaults, **train})
    cfg["run"]["smoke"] = True
    return cfg


def read_rows(run_dir):
    return [json.loads(line) for line in (Path(run_dir) / "metrics.jsonl").read_text().splitlines()]


def val_dataset(fs2k):
    return FS2KDataset("val", fs2k["data_root"], augment=False)


# ----------------------------------------------------------------------------------- losses
class _ConstG(nn.Module):
    """Generator stub: every pixel is `a` (a trainable number)."""

    def __init__(self, a):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a))

    def forward(self, photo, style):
        return self.a * torch.ones(photo.shape[0], 1, 128, 128)


class _MeanD(nn.Module):
    """Discriminator stub: one logit per image = w * mean(sketch)."""

    def __init__(self, w):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(w))

    def forward(self, photo, sketch, style):
        return self.w * sketch.mean(dim=(1, 2, 3), keepdim=True)


def test_losses_match_formula():
    """Real sketch = 0.5 everywhere, G output = 0.25, D logit = w * mean(sketch), w = 2 -> logits 1.0 and 0.5."""
    G, D = _ConstG(0.25), _MeanD(2.0)
    photo = torch.zeros(2, 3, 128, 128)
    sketch = torch.full((2, 1, 128, 128), 0.5)
    batch = (photo, sketch, torch.tensor([0, 1]))
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    softplus = lambda x: math.log(1 + math.exp(x))  # noqa: E731   BCEWithLogits(x, 0) = softplus(x); (x, 1) = softplus(-x)

    d = d_step(G, D, torch.optim.SGD(D.parameters(), lr=0.0), batch, scaler)
    assert float(d["d_real"]) == pytest.approx(softplus(-1.0), abs=1e-6)          # D(real) logit 1.0, target 1
    assert float(d["d_fake"]) == pytest.approx(softplus(0.5), abs=1e-6)           # D(fake) logit 0.5, target 0
    assert float(d["d_loss"]) == pytest.approx(0.5 * (softplus(-1.0) + softplus(0.5)), abs=1e-6)
    assert float(d["d_real_prob"]) == pytest.approx(1 / (1 + math.exp(-1.0)), abs=1e-6)

    g = g_step(G, D, torch.optim.SGD(G.parameters(), lr=0.0), batch, 10.0, scaler)
    assert float(g["g_adv"]) == pytest.approx(softplus(-0.5), abs=1e-6)           # fake logit 0.5, target 1
    assert float(g["g_l1"]) == pytest.approx(0.25, abs=1e-6)                      # |0.5 - 0.25|
    assert float(g["g_loss"]) == pytest.approx(softplus(-0.5) + 10.0 * 0.25, abs=1e-6)


def test_one_step_losses_are_finite():
    torch.manual_seed(0)
    G, D = Generator.from_config(MODEL), Discriminator.from_config(MODEL)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    g = torch.Generator().manual_seed(1)
    batch = (torch.rand(3, 3, 128, 128, generator=g) * 2 - 1, torch.rand(3, 1, 128, 128, generator=g) * 2 - 1,
             torch.tensor([0, 1, 2]))
    d = d_step(G, D, torch.optim.Adam(D.parameters(), 1e-4), batch, scaler)
    gg = g_step(G, D, torch.optim.Adam(G.parameters(), 1e-4), batch, 100.0, scaler)
    for name in ("d_real", "d_fake", "d_loss"):
        assert math.isfinite(float(d[name])), name
    for name in ("g_adv", "g_l1", "g_loss"):
        assert math.isfinite(float(gg[name])), name


def test_d_step_with_shared_fake_matches_own_fake():
    """Passing the generator output in (as the training loop does) gives the same D losses as letting d_step make it."""
    torch.manual_seed(0)
    G, D = Generator.from_config({**MODEL, "dropout": 0.0}), Discriminator.from_config(MODEL)
    G.eval()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    photo = torch.rand(2, 3, 128, 128) * 2 - 1
    sketch = torch.rand(2, 1, 128, 128) * 2 - 1
    batch = (photo, sketch, torch.tensor([0, 2]))
    a = d_step(G, D, torch.optim.SGD(D.parameters(), lr=0.0), batch, scaler)
    b = d_step(G, D, torch.optim.SGD(D.parameters(), lr=0.0), batch, scaler, fake=G(photo, batch[2]))
    assert float(a["d_loss"]) == pytest.approx(float(b["d_loss"]), abs=1e-6)


def test_g_step_leaves_d_untouched():
    torch.manual_seed(0)
    G, D = Generator.from_config(MODEL), Discriminator.from_config(MODEL)
    before = {k: v.clone() for k, v in D.state_dict().items()}
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    batch = (torch.rand(2, 3, 128, 128) * 2 - 1, torch.rand(2, 1, 128, 128) * 2 - 1, torch.tensor([1, 2]))
    g_step(G, D, torch.optim.Adam(G.parameters(), 1e-3), batch, 100.0, scaler)
    assert all(torch.equal(before[k], v) for k, v in D.state_dict().items())
    assert all(p.requires_grad for p in D.parameters())          # unfrozen again afterwards


# -------------------------------------------------------------------------------- schedules
def test_linear_decay_half_schedule():
    for name, total in (("constant", 10), ("linear_decay_half", 10)):
        opt = torch.optim.Adam([nn.Parameter(torch.zeros(1))], lr=1.0)
        sched = make_scheduler(opt, name, total)
        lrs = []
        for _ in range(total):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()
        if name == "constant":
            assert lrs == [1.0] * total
        else:
            assert lrs[:5] == [1.0] * 5                              # constant for the first half
            assert lrs[5:] == pytest.approx([1.0, 0.8, 0.6, 0.4, 0.2])   # then a straight line down to 0
            assert opt.param_groups[0]["lr"] == 0.0
    with pytest.raises(ValueError):
        make_scheduler(opt, "cosine", 10)


# --------------------------------------------------------------------- validate / fixed samples
def test_validate_values_on_01_scale(fs2k):
    """A generator that outputs 0 (= 0.5 on [0,1]) has L1 = mean|0.5 - (sketch+1)/2|; keys are the contract keys."""
    ds = val_dataset(fs2k)
    G = _ConstG(0.0)
    loader = torch.utils.data.DataLoader(ds, batch_size=3)
    res = validate(G, loader, torch.device("cpu"))
    assert set(res) == {"l1", "ssim", "psnr", "l1_style0", "l1_style1", "l1_style2"}
    per_item = [(0.5 - (ds[i][1] + 1) / 2).abs().mean().item() for i in range(len(ds))]
    assert res["l1"] == pytest.approx(sum(per_item) / len(per_item), abs=1e-5)
    styles = [int(ds[i][2]) for i in range(len(ds))]
    for k in range(3):
        mine = [v for v, s in zip(per_item, styles) if s == k]
        assert res[f"l1_style{k}"] == pytest.approx(sum(mine) / len(mine), abs=1e-5)
    assert 0 <= res["ssim"] <= 1


def test_fixed_sample_ids_and_grid(fs2k, tmp_path):
    ds = val_dataset(fs2k)
    ids = fixed_sample_ids(ds)
    assert ids == fixed_sample_ids(ds)                                # stable
    by_id = {ds[i][3]: int(ds[i][2]) for i in range(len(ds))}
    styles = [by_id[i] for i in ids]
    assert styles == sorted(styles) and all(styles.count(k) <= n for k, n in enumerate((3, 3, 2)))
    for k in range(3):                                                # sorted ids inside each style
        assert [i for i in ids if by_id[i] == k] == sorted(i for i in ids if by_id[i] == k)
    torch.manual_seed(0)
    G = Generator.from_config(MODEL)
    grid = save_sample_grid(G, ds, ids, torch.device("cpu"), tmp_path / "grid.png")
    assert (tmp_path / "grid.png").exists()
    assert grid.shape[0] == 3 and 0.0 <= grid.min() and grid.max() <= 1.0
    assert grid.shape[2] == 5 * 128 + 6 * 2 and grid.shape[1] == len(ids) * 128 + (len(ids) + 1) * 2   # 5 columns


# ------------------------------------------------------------------------- full training runs
def test_two_epoch_run_writes_all_outputs(fs2k, tmp_path):
    cfg = make_cfg(fs2k, tmp_path, schedule="linear_decay_half")
    saved = []
    run_dir = run_training(cfg, on_checkpoint=saved.append)
    run_dir = Path(run_dir)
    assert run_dir.parent == tmp_path / "out" / "runs" / "task4"
    rows = read_rows(run_dir)
    assert [r["epoch"] for r in rows] == [1, 2] and rows[-1]["global_step"] == 2 * STEPS
    needed = ["train/d_real", "train/d_fake", "train/g_adv", "train/g_l1", "train/d_real_prob", "train/d_fake_prob",
              "val/l1", "val/ssim", "val/psnr", "val/l1_style0", "val/l1_style1", "val/l1_style2"]
    for key in needed:
        assert key in rows[0] and math.isfinite(rows[0][key]), key
    assert (run_dir / "samples" / "epoch001.png").exists()            # epoch-1 grid; the next one would be epoch 5
    assert (run_dir / "ckpt_last.pt").exists() and (run_dir / "ckpt_best.pt").exists()
    assert (run_dir / "config.yaml").exists() and saved and all(p.exists() for p in saved)
    assert rows[1]["lr_g"] < rows[0]["lr_g"]                          # linear decay during the second half

    ck = load_checkpoint(run_dir / "ckpt_last.pt")
    for key in ("model", "optimizer", "scheduler", "scaler", "epoch", "global_step", "best_metric", "config", "seed",
                "split_sha256", "manifest_sha256", "git_commit", "torch_version", "created_utc",
                "discriminator", "optimizer_d", "scheduler_d", "scaler_d", "rng", "step_in_epoch", "fixed_sample_ids"):
        assert key in ck, key
    assert ck["epoch"] == 2 and ck["step_in_epoch"] == 0 and ck["global_step"] == 2 * STEPS
    assert ck["split_sha256"] == sha256_file(fs2k["split_file"]) and ck["manifest_sha256"] is None
    assert ck["config"]["model"] == MODEL and ck["config"]["run_id"] == run_dir.name
    assert ck["best_metric"] == pytest.approx(min(r["val/l1"] for r in rows))
    Generator.from_config(ck["config"]["model"]).load_state_dict(ck["model"])
    Discriminator.from_config(ck["config"]["model"]).load_state_dict(ck["discriminator"])


def test_resume_continues_and_restores_state(fs2k, tmp_path):
    cfg1 = make_cfg(fs2k, tmp_path, epochs=1)
    run_dir = run_training(cfg1)
    first = load_checkpoint(run_dir / "ckpt_last.pt")
    assert first["epoch"] == 1 and first["global_step"] == STEPS
    assert first["optimizer"]["state"][0]["step"] == STEPS

    cfg2 = make_cfg(fs2k, tmp_path, epochs=2)
    run_dir2 = run_training(cfg2, resume=str(run_dir / "ckpt_last.pt"))
    assert run_dir2 == run_dir                                        # the same run folder is continued
    last = load_checkpoint(run_dir / "ckpt_last.pt")
    assert last["epoch"] == 2 and last["global_step"] == 2 * STEPS
    assert last["fixed_sample_ids"] == first["fixed_sample_ids"]
    # Adam keeps counting steps only if both optimiser states were restored (a fresh Adam would show STEPS)
    assert last["optimizer"]["state"][0]["step"] == 2 * STEPS
    assert last["optimizer_d"]["state"][0]["step"] == 2 * STEPS
    assert last["scheduler"]["last_epoch"] == 2 * STEPS and last["scheduler_d"]["last_epoch"] == 2 * STEPS
    assert [r["epoch"] for r in read_rows(run_dir)] == [1, 2]


def test_resume_auto_and_mid_epoch(fs2k, tmp_path):
    cfg = make_cfg(fs2k, tmp_path, epochs=1, pause_after_steps=3)
    run_dir = run_training(cfg)                                       # simulated interruption inside epoch 1
    paused = load_checkpoint(run_dir / "ckpt_last.pt")
    assert paused["global_step"] == 3 and paused["step_in_epoch"] == 3 and paused["epoch"] == 0
    assert not (run_dir / "metrics.jsonl").exists()                   # no validation when paused

    cfg2 = make_cfg(fs2k, tmp_path, epochs=1)
    assert run_training(cfg2, resume="auto") == run_dir               # found through find_latest_checkpoint
    done = load_checkpoint(run_dir / "ckpt_last.pt")
    assert done["epoch"] == 1 and done["global_step"] == STEPS and done["optimizer"]["state"][0]["step"] == STEPS
    assert len(read_rows(run_dir)) == 1 and read_rows(run_dir)[0]["partial_epoch"] is False


def test_resume_refuses_changed_settings(fs2k, tmp_path):
    run_dir = run_training(make_cfg(fs2k, tmp_path, epochs=1))
    for key, value in (("lr_g", 0.5), ("lambda_l1", 1.0), ("batch_size", 2), ("schedule", "linear_decay_half")):
        with pytest.raises(ValueError, match=key):
            run_training(make_cfg(fs2k, tmp_path, epochs=2, **{key: value}), resume=str(run_dir / "ckpt_last.pt"))
    bad = make_cfg(fs2k, tmp_path, epochs=2)
    bad["model"]["base_channels"] = 16
    with pytest.raises(ValueError, match="model config differs"):
        run_training(bad, resume=str(run_dir / "ckpt_last.pt"))


def test_snapshots_and_report_fn(fs2k, tmp_path):
    cfg = make_cfg(fs2k, tmp_path, snapshot_every_epochs=1)
    seen = []
    run_dir, best = t4train._train(cfg, report_fn=lambda epoch, val_l1: seen.append((epoch, val_l1)))
    assert [e for e, _ in seen] == [1, 2]
    rows = read_rows(run_dir)
    assert [v for _, v in seen] == pytest.approx([r["val/l1"] for r in rows])
    assert best == pytest.approx(min(r["val/l1"] for r in rows))
    assert (run_dir / "ckpt_epoch001.pt").exists() and (run_dir / "ckpt_epoch002.pt").exists()

    class Stop(Exception):
        pass

    def prune(epoch, value):
        raise Stop

    cfg2 = make_cfg(fs2k, tmp_path)
    with pytest.raises(Stop):
        t4train._train(cfg2, report_fn=prune)


def test_trial_runs_go_to_their_own_folder_and_skip_samples(fs2k, tmp_path):
    cfg = make_cfg(fs2k, tmp_path, epochs=1, sample_every_epochs=0)
    cfg["run"]["trial"] = True
    cfg["run"]["smoke"] = False
    cfg["run_id"] = "t4_trial0"
    run_dir, _ = t4train._train(cfg)
    assert run_dir == tmp_path / "out" / "runs" / "task4_trials" / "t4_trial0"
    assert list((run_dir / "samples").iterdir()) == []                # no grids for trials
    assert t4cfg.find_latest_checkpoint(make_cfg(fs2k, tmp_path)) is None   # resume="auto" never sees a trial


def test_never_opens_the_test_split(fs2k, tmp_path):
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return FS2KDataset(**kwargs)

    t4train._train(make_cfg(fs2k, tmp_path, epochs=1), dataset_factory=factory)
    assert {c["split"] for c in calls} == {"train", "val"}
    assert all("final_test" not in c for c in calls)
    assert not (tmp_path / "test_access.log").exists()


def test_epochs_tbd_guard_and_max_steps(fs2k, tmp_path):
    with pytest.raises(ValueError, match="TBD_AFTER_BENCHMARK"):
        run_training(make_cfg(fs2k, tmp_path, epochs="TBD_AFTER_BENCHMARK"))
    # with a step cap the schedule length is taken from max_steps (smoke runs): 5 steps = one partial epoch
    run_dir = run_training(make_cfg(fs2k, tmp_path, epochs="TBD_AFTER_BENCHMARK", max_steps=5,
                                    train_subset=20, val_subset=5))
    rows = read_rows(run_dir)
    assert len(rows) == 1 and rows[0]["global_step"] == 5 and rows[0]["partial_epoch"] is False   # 20 pairs // 4 = 5 steps
    assert (run_dir / "samples" / "epoch001.png").exists()


def test_batch_size_validation(fs2k, tmp_path):
    with pytest.raises(ValueError, match="fewer than batch_size"):
        run_training(make_cfg(fs2k, tmp_path, epochs=1, batch_size=8, train_subset=6))


# ------------------------------------------------------------------- split / promote checkpoint
def test_split_checkpoint_and_promote(fs2k, tmp_path):
    run_dir = run_training(make_cfg(fs2k, tmp_path, epochs=1))
    g_path, d_path = split_checkpoint(run_dir / "ckpt_best.pt", tmp_path / "split")
    assert g_path.name == "t4_generator.pt" and d_path.name == "t4_discriminator.pt"
    g_ck, d_ck = load_checkpoint(g_path), load_checkpoint(d_path)
    full = load_checkpoint(run_dir / "ckpt_best.pt")
    for ck in (g_ck, d_ck):
        assert ck["optimizer"] is None and ck["scheduler"] is None and ck["scaler"] is None
        assert ck["config"]["model"] == MODEL and ck["split_sha256"] == full["split_sha256"]
        assert "rng" not in ck and "optimizer_d" not in ck and "discriminator" not in ck
    G = Generator.from_config(g_ck["config"]["model"])
    D = Discriminator.from_config(d_ck["config"]["model"])
    G.load_state_dict(g_ck["model"])
    D.load_state_dict(d_ck["model"])
    G.eval()
    photo = torch.rand(2, 3, 128, 128) * 2 - 1
    ref = Generator.from_config(MODEL)
    ref.load_state_dict(full["model"])
    ref.eval()
    style = torch.tensor([0, 2])
    assert torch.equal(G(photo, style), ref(photo, style))            # the generator file holds the trained G

    models = tmp_path / "models"
    models.mkdir()
    (models / "MANIFEST.json").write_text("[]")                       # bare-list manifest
    dest_g = promote(g_path, "t4_generator", root=models)
    dest_d = promote(d_path, "t4_discriminator", root=models)
    manifest = json.loads((models / "MANIFEST.json").read_text())
    assert {e["name"]: e["sha256"] for e in manifest} == {"t4_generator": sha256_file(dest_g),
                                                         "t4_discriminator": sha256_file(dest_d)}
