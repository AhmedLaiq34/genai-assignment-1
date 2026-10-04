"""Task 3 training (soft mixture of experts) on the tiny SYNTHETIC dataset of tests/conftest.py and the
RANDOM fixture source checkpoints of tests/t3_fixtures.py. CPU only (run with CUDA_VISIBLE_DEVICES=-1), no real data.

Every test pins its own tiny budget in make_cfg(); nothing here reads configs/task3_moe.yaml except the one test
that checks that file's structure (in test_task3_tune.py).
"""
import json
from pathlib import Path

import pytest
import torch

from genai.common.checkpoint import load_checkpoint, sha256_file
from genai.common.metrics import l1, ssim
from genai.models.autoencoder import count_parameters
from genai.models.moe import SoftMoE, parameter_hash
from genai.pets import dataset as pets_dataset
from genai.pets.dataset import resolve_data_paths
from genai.tasks.task1.train import build_val_loader
from genai.tasks.task2.classifier import build_train_loader
from genai.tasks.task3 import BRANCH_NAMES, EXPERT_NAMES
from genai.tasks.task3 import train as tr
from genai.tasks.task3.train import (balance_loss, collapse_flags, measure_step_times, moe_loss, run_training,
                                     validate)
from t3_fixtures import make_t3_sources

NEVER = {"min_mean_weight": 0.0, "max_foreign_weight": 1.0}      # thresholds that never flag a collapse
REAL = {"min_mean_weight": 0.02, "max_foreign_weight": 0.9}


def make_cfg(root, tmp_path, paths, sha, **train):
    """A full Task 3 config shrunk to a tiny CPU run: 64 train images, batch 8 -> 8 steps per epoch (4 with
    train_subset=32), 1 warm-up + 2 joint epochs, 16 validation rows."""
    cfg = {
        "task": "t3", "component": "soft_moe", "seed": 42, "device": "test", "device_profile": "local",
        "data_root": str(root), "output_root": str(tmp_path / "out"), "persist_root": str(tmp_path / "out"),
        "num_workers": 0,
        "run": {"desc": "t3moe", "smoke": False, "project": "genai-a1"},
        "sources": {"dir": str(Path(paths["classifier"]).parent), "expected_sha256": sha},
        "model": {"tau": 2.0},
        "train": {"batch_size": 8, "warmup_epochs": 1, "joint_epochs": 2, "warmup_lr": 2e-4, "joint_lr": 5e-5,
                  "weight_decay": 1e-4, "lambda_1": 0.8, "lambda_s": 0.2, "lambda_c": 0.1, "lambda_b": 0.01,
                  "amp": False, "scheduler": "cosine", "grad_clip": None, "train_subset": 32, "val_subset": 16,
                  "val_every_epochs": 1, "val_batch_size": 16, "val_workers": 0, "epoch0_validation": True,
                  "checkpoint_every_minutes": 10, "log_every_steps": 5, "sample_every_epochs": 1,
                  "max_steps": None, "pause_after_steps": None},
        "collapse": dict(NEVER),
    }
    cfg["train"].update(train)
    return cfg


def read_rows(run_dir):
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]


@pytest.fixture(scope="module")
def t3_sources(tmp_path_factory):
    """(paths, sha) of random fixture source checkpoints; the tests only ever read them."""
    return make_t3_sources(tmp_path_factory.mktemp("t3src"))


@pytest.fixture
def cfg(tiny_pets_root, tmp_path, t3_sources):
    return make_cfg(tiny_pets_root, tmp_path, *t3_sources)


def rebuild(ckpt) -> SoftMoE:
    """The model of a checkpoint, rebuilt from its config only (no Task 2 file)."""
    model = SoftMoE.from_config(ckpt["config"]["model"])
    model.load_state_dict(ckpt["model"])
    return model


# ------------------------------------------------------------------------------------------- loss
def test_balance_loss_zero_for_balanced_one_hot_and_positive_when_collapsed():
    one_hot_balanced = torch.eye(4).repeat(3, 1)                      # 12 rows, 3 of each class, all one-hot
    assert balance_loss(one_hot_balanced).item() == pytest.approx(0.0, abs=1e-12)
    uniform = torch.full((8, 4), 0.25)
    assert balance_loss(uniform).item() == pytest.approx(0.0, abs=1e-12)
    collapsed = torch.tensor([[1.0, 0, 0, 0]]).repeat(8, 1)           # every row on branch 0
    assert balance_loss(collapsed).item() == pytest.approx(0.75 ** 2 + 3 * 0.25 ** 2)       # hand computed: 0.75
    two_rows = torch.tensor([[0.7, 0.1, 0.1, 0.1], [0.5, 0.3, 0.1, 0.1]])                     # means 0.6, 0.2, 0.1, 0.1
    assert balance_loss(two_rows).item() == pytest.approx(0.35 ** 2 + 0.05 ** 2 + 0.15 ** 2 + 0.15 ** 2)


def test_loss_terms_match_hand_computed_values():
    torch.manual_seed(0)
    x_hat, clean = torch.rand(4, 3, 128, 128), torch.rand(4, 3, 128, 128)
    logits = torch.randn(4, 4)
    w = torch.softmax(logits / 2.0, dim=1)
    labels = torch.tensor([0, 1, 2, 3])
    lam = {"lambda_1": 0.7, "lambda_s": 0.3, "lambda_c": 0.5, "lambda_b": 2.0}
    total, parts = moe_loss(x_hat, clean, logits, w, labels, lam)

    recon = 0.7 * l1(x_hat, clean).mean() + 0.3 * (1.0 - ssim(x_hat, clean).mean())          # lambda_1 L1 + lambda_s (1 - SSIM)
    ce = -torch.log_softmax(logits, dim=1)[torch.arange(4), labels].mean()                    # CE on the RAW logits (no tau)
    bal = ((w.mean(0) - 0.25) ** 2).sum()
    assert parts["recon"] == pytest.approx(recon.item(), rel=1e-5)
    assert parts["ce"] == pytest.approx(ce.item(), rel=1e-5)
    assert parts["balance"] == pytest.approx(bal.item(), rel=1e-5)
    assert total.item() == pytest.approx((recon + 0.5 * ce + 2.0 * bal).item(), rel=1e-5)
    assert total.dtype == torch.float32

    with pytest.raises(ValueError, match="lambda_1 \\+ lambda_s"):                           # decision B8: r and 1 - r
        moe_loss(x_hat, clean, logits, w, labels, {**lam, "lambda_s": 0.5})


def test_ce_uses_raw_logits_not_the_tempered_weights():
    x = torch.rand(4, 3, 128, 128)
    logits = torch.tensor([[4.0, 0, 0, 0], [0, 4.0, 0, 0], [0, 0, 4.0, 0], [0, 0, 0, 4.0]])
    labels = torch.arange(4)
    lam = {"lambda_1": 0.8, "lambda_s": 0.2, "lambda_c": 1.0, "lambda_b": 0.0}
    ces = []
    for tau in (0.5, 1.0, 5.0):
        _total, parts = moe_loss(x, x, logits, torch.softmax(logits / tau, dim=1), labels, lam)
        ces.append(parts["ce"])
    assert ces[0] == pytest.approx(ces[1]) == pytest.approx(ces[2])                           # independent of tau


# ------------------------------------------------------------------------------- collapse flags
def test_collapse_flags_rules():
    healthy = [[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]]
    ok = collapse_flags([0.25] * 4, healthy, REAL)
    assert ok == {"collapsed": False, "reasons": []}

    dead = collapse_flags([0.01, 0.3, 0.3, 0.39], healthy, {"collapse": REAL})                # (a) also with the whole cfg
    assert dead["collapsed"] and len(dead["reasons"]) == 1 and "branch 0 (identity)" in dead["reasons"][0]

    foreign = [row[:] for row in healthy]
    foreign[2] = [0.95, 0.02, 0.02, 0.01]                                                     # blur rows sent to identity
    flagged = collapse_flags([0.25] * 4, foreign, REAL)                                       # (b)
    assert flagged["collapsed"] and "true class 2 (gaussian_blur)" in flagged["reasons"][0]

    own_class = [row[:] for row in healthy]
    own_class[1] = [0.01, 0.95, 0.02, 0.02]                                                   # 0.95 on its OWN class is fine
    assert not collapse_flags([0.25] * 4, own_class, REAL)["collapsed"]
    absent = [[0.0] * 4] + healthy[1:]                                                        # a class without rows: no flag
    assert not collapse_flags([0.25] * 4, absent, REAL)["collapsed"]


# --------------------------------------------------------------------------- validate + stages
@pytest.fixture
def source_model(t3_sources):
    paths, sha = t3_sources
    return SoftMoE.load_from_task2(paths, tau=2.0, expected_sha256=sha)


def test_validate_returns_weights_gate_accuracy_and_restores_modes(cfg, source_model):
    cfg["train"]["val_subset"] = 32
    _ds, n, loader, _sha = build_val_loader(cfg, resolve_data_paths(cfg))
    for start_mode in (True, False):
        source_model.train(start_mode)
        res = validate(source_model, loader, torch.device("cpu"), REAL)
        assert source_model.training is start_mode and not source_model.experts.training      # B4: experts always eval
        assert res["count"] == 32 == n
        assert sum(res["mean_w"]) == pytest.approx(1.0, abs=1e-5)
        assert len(res["mean_w_by_class"]) == 4
        for row in res["mean_w_by_class"]:
            assert sum(row) == pytest.approx(1.0, abs=1e-5)
        assert 0.0 <= res["gate_accuracy"] <= 1.0 and 0.0 <= res["J"] <= 1.0
        assert set(res["flags"]) == {"collapsed", "reasons"} and len(res["per_condition"]) == 4
        json.dumps(res)                                                                        # JSON-serialisable


def test_warmup_leaves_experts_untouched_and_changes_the_gate(cfg, t3_sources, source_model):
    cfg["train"].update(warmup_epochs=1, joint_epochs=0)
    run_dir = run_training(cfg)
    ckpt = load_checkpoint(run_dir / "ckpt_last.pt")
    trained = rebuild(ckpt)
    # parameters AND buffers (BatchNorm statistics) of every expert are byte-identical to the source
    assert parameter_hash(trained.experts) == parameter_hash(source_model.experts)
    assert parameter_hash(trained.gate) != parameter_hash(source_model.gate)
    assert ckpt["stage"] == "warmup" and ckpt["scheduler"] is None
    n_gate = sum(1 for _ in source_model.gate.parameters())
    assert len(ckpt["optimizer"]["param_groups"]) == 1 and len(ckpt["optimizer"]["param_groups"][0]["params"]) == n_gate
    assert ckpt["optimizer"]["param_groups"][0]["lr"] == pytest.approx(2e-4)                  # constant warm-up lr
    assert not (run_dir / "ckpt_best.pt").exists()                                            # no joint epoch, no best


def test_joint_step_gives_every_part_a_gradient(cfg, source_model):
    paths = resolve_data_paths(cfg)
    _ds, _sampler, loader = build_train_loader(cfg, paths, 8)
    corrupted, clean, cond, _sev = next(iter(loader))
    optimizer, _sched = tr.make_stage(source_model, "joint", cfg["train"], joint_steps=10)
    assert not source_model.experts_frozen and len(optimizer.param_groups) == 1               # ONE group over everything
    source_model.train()
    tr._train_step(source_model, optimizer, torch.amp.GradScaler("cuda", enabled=False), corrupted, clean, cond,
                   {k: cfg["train"][k] for k in tr.LAMBDA_KEYS}, False, None)
    for name, part in [("gate", source_model.gate)] + [(n, source_model.experts[n]) for n in EXPERT_NAMES]:
        norm = sum(p.grad.abs().sum().item() for p in part.parameters() if p.grad is not None)
        assert norm > 0.0, f"{name} got no gradient in the joint stage"


def test_warmup_stage_freezes_experts_and_trains_only_the_gate(source_model, cfg):
    optimizer, scheduler = tr.make_stage(source_model, "warmup", cfg["train"], joint_steps=10)
    assert source_model.experts_frozen and scheduler is None
    assert all(p.requires_grad for p in source_model.gate.parameters())
    ids = {id(p) for p in optimizer.param_groups[0]["params"]}
    assert ids == {id(p) for p in source_model.gate.parameters()}
    assert tr.stage_of(0, 2) == "warmup" and tr.stage_of(1, 2) == "warmup" and tr.stage_of(2, 2) == "joint"


# ------------------------------------------------------------------------- a full run, layout
def test_full_run_checkpoint_layout_files_sources_and_smoke(cfg, t3_sources):
    paths, sha = t3_sources
    before = {name: sha256_file(p) for name, p in paths.items()}
    seen = []
    run_dir = run_training(cfg, on_checkpoint=seen.append)
    assert {name: sha256_file(p) for name, p in paths.items()} == before                      # Task 2 files byte-identical

    # smoke propagates from the (fixture) sources although run.smoke was false; folder and name follow the rules
    assert run_dir.parent.name == "task3" and run_dir.parent.parent.name == "runs"
    assert run_dir.name.endswith("_test_t3moe_smoke")
    for name in ("config.yaml", "ckpt_last.pt", "ckpt_best.pt", "metrics.jsonl"):
        assert (run_dir / name).exists()
    assert len(list((run_dir / "samples").glob("*.png"))) == 3                                # sample_every_epochs 1
    assert run_dir / "ckpt_last.pt" in seen and run_dir / "ckpt_best.pt" in seen              # hook called after each save

    rows = read_rows(run_dir)
    assert [r["epoch"] for r in rows] == [0, 1, 2, 3]                                         # epoch 0 + 3 trained epochs
    assert [r["stage"] for r in rows] == ["init", "warmup", "joint", "joint"]
    assert rows[0]["train_loss"] is None and rows[-1]["global_step"] == 12
    for key in ("epoch", "stage", "global_step", "train_loss", "train_recon", "train_ce", "train_balance",
                "train_mean_w", "lr", "val_J", "val_l1", "val_ssim", "val_psnr", "val_mean_w", "val_mean_w_by_class",
                "val_gate_accuracy", "collapsed", "collapse_reasons", "epoch_seconds"):
        assert key in rows[1], key
    assert len(rows[1]["train_mean_w"]) == 4 and sum(rows[1]["train_mean_w"]) == pytest.approx(1.0, abs=1e-4)
    assert len(rows[1]["val_mean_w_by_class"]) == 4 and rows[1]["lr"] == pytest.approx(2e-4)
    assert rows[2]["lr"] < 5e-5 + 1e-12 and rows[3]["lr"] < rows[2]["lr"]                     # cosine decay over the joint steps

    best = load_checkpoint(run_dir / "ckpt_best.pt")
    for key in ("model", "optimizer", "scheduler", "scaler", "epoch", "global_step", "best_metric", "config", "seed",
                "split_sha256", "manifest_sha256", "rng", "stage", "step_in_epoch", "epoch_sums"):
        assert key in best, key
    assert best["best_metric"] == pytest.approx(min(r["val_J"] for r in rows[2:]))            # best = lowest JOINT J
    config = best["config"]
    assert config["component"] == "soft_moe" and config["run_id"] == run_dir.name and config["run"]["smoke"] is True
    assert set(config["model"]) == {"gate", "experts", "tau", "branches"}
    assert config["model"]["tau"] == 2.0 and config["model"]["branches"] == list(BRANCH_NAMES)
    assert set(config["model"]["experts"]) == set(EXPERT_NAMES)
    assert config["source_checkpoints"] == {n: {"file": Path(paths[n]).name, "sha256": before[n]} for n in paths}

    # the checkpoint config alone rebuilds the model (no Task 2 file needed) and the weights load
    model = rebuild(best).eval()
    x = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        x_hat, w, _ = model(x)
    assert x_hat.shape == (2, 3, 128, 128) and torch.allclose(w.sum(dim=1), torch.ones(2), atol=1e-5)


def test_smoke_flag_comes_only_from_run_or_sources(cfg, monkeypatch):
    fake = {n: {"smoke": False} for n in ("classifier", "salt", "blur", "occlusion")}
    monkeypatch.setattr(tr, "describe_checkpoints", lambda paths: fake)
    cfg["train"].update(warmup_epochs=1, joint_epochs=0, train_subset=16)
    run_dir = run_training(cfg)
    assert not run_dir.name.endswith("_smoke") and load_checkpoint(run_dir / "ckpt_last.pt")["config"]["run"]["smoke"] is False


def test_epoch0_row_can_be_switched_off_and_val_every_epochs(cfg):
    cfg["train"].update(epoch0_validation=False, val_every_epochs=2)                          # 3 epochs: validate 2 and (last) 3
    run_dir = run_training(cfg)
    assert [r["epoch"] for r in read_rows(run_dir)] == [2, 3]
    assert load_checkpoint(run_dir / "ckpt_last.pt")["epoch"] == 3


# --------------------------------------------------------------------------------------- resume
@pytest.mark.parametrize("pause_step, stage, n_params_group", [(2, "warmup", "gate"), (6, "joint", "all")])
def test_resume_continues_global_step_stage_and_rng(cfg, monkeypatch, pause_step, stage, n_params_group):
    cfg["run"]["smoke"] = True                                      # resume="auto" only looks at smoke runs for a smoke config
    cfg["train"]["pause_after_steps"] = pause_step                  # 4 steps per epoch: step 2 = inside warm-up, 6 = inside joint 1
    run_dir = run_training(cfg)
    paused = load_checkpoint(run_dir / "ckpt_last.pt")
    assert paused["global_step"] == pause_step and paused["stage"] == stage
    assert paused["step_in_epoch"] == pause_step % 4
    n_all = sum(1 for _ in rebuild(paused).parameters())
    n_gate = sum(1 for _ in rebuild(paused).gate.parameters())
    assert len(paused["optimizer"]["param_groups"][0]["params"]) == (n_gate if n_params_group == "gate" else n_all)
    assert (paused["scheduler"] is None) == (stage == "warmup")

    restored = []
    real = tr.restore_rng_state
    monkeypatch.setattr(tr, "restore_rng_state", lambda s: (restored.append(set(s)), real(s)))
    cfg["train"]["pause_after_steps"] = None
    assert tr.find_latest_checkpoint(cfg) == run_dir / "ckpt_last.pt"                         # what resume="auto" finds
    run_dir2 = run_training(cfg, resume="auto")
    assert run_dir2 == run_dir                                      # the same run continues
    assert restored == [{"python", "numpy", "torch", "cuda"}]       # all RNG states restored
    final = load_checkpoint(run_dir / "ckpt_last.pt")
    assert final["global_step"] == 12 and final["epoch"] == 3 and final["step_in_epoch"] == 0 and final["stage"] == "joint"
    assert [r["epoch"] for r in read_rows(run_dir)] == [0, 1, 2, 3]  # epoch 0 written once only


def test_resume_refuses_changed_settings(cfg):
    cfg["train"]["pause_after_steps"] = 2
    run_dir = run_training(cfg)
    ckpt = str(run_dir / "ckpt_last.pt")
    bad = json.loads(json.dumps(cfg))
    bad["train"]["joint_lr"] = 0.5
    with pytest.raises(ValueError, match="joint_lr"):
        run_training(bad, resume=ckpt)
    bad = json.loads(json.dumps(cfg))
    bad["model"]["tau"] = 3.0
    with pytest.raises(ValueError, match="tau"):
        run_training(bad, resume=ckpt)


# ------------------------------------------------------------------------- best checkpoint rule
def scripted_validate(monkeypatch, js, collapsed):
    """Replace validate() so that call i returns J = js[i] and the collapse flag collapsed[i]."""
    real, calls = tr.validate, []

    def fake(model, loader, device, collapse_cfg=None):
        res = real(model, loader, device, collapse_cfg)
        i = len(calls)
        calls.append(i)
        res["J"] = js[i]
        if collapsed[i]:
            res["mean_w_by_class"][1][0] = 0.95                     # class 1 rows sent to branch 0
        res["flags"] = collapse_flags(res["mean_w"], res["mean_w_by_class"], REAL)
        return res

    monkeypatch.setattr(tr, "validate", fake)
    return calls


def test_best_is_lowest_joint_j_that_is_not_collapsed(cfg, monkeypatch):
    cfg["train"].update(warmup_epochs=1, joint_epochs=3)
    # calls: epoch 0, warm-up, joint 1, joint 2 (collapsed, lowest of all joint J), joint 3
    scripted_validate(monkeypatch, js=[0.5, 0.01, 0.2, 0.1, 0.15], collapsed=[False, False, False, True, False])
    run_dir = run_training(cfg)
    best = load_checkpoint(run_dir / "ckpt_best.pt")
    assert best["best_metric"] == pytest.approx(0.15)               # not 0.01 (warm-up, epoch 0) and not 0.1 (collapsed)
    assert best["epoch"] == 4 and best["global_step"] == 16
    rows = read_rows(run_dir)
    assert [r["collapsed"] for r in rows] == [False, False, False, True, False]
    assert rows[3]["collapse_reasons"] and "branch 0 (identity)" in rows[3]["collapse_reasons"][0]


def test_all_joint_epochs_collapsed_is_a_clear_error(cfg, monkeypatch):
    scripted_validate(monkeypatch, js=[0.5, 0.4, 0.3, 0.2], collapsed=[False, False, True, True])
    with pytest.raises(ValueError, match="collapse"):
        run_training(cfg)


def test_report_fn_gets_epoch_j_and_flags_and_may_stop_the_run(cfg):
    reports = []

    def report(step, val_J, flags):
        reports.append((step, val_J, flags))

    run_dir, best = tr._train(cfg, report_fn=report)
    assert [r[0] for r in reports] == [1, 2, 3]                     # one call per finished epoch, warm-up first, from 1
    step, val_J, flags = reports[0]
    assert {"collapsed", "reasons", "mean_w", "mean_w_by_class"} <= set(flags) and len(flags["mean_w_by_class"]) == 4
    assert best == pytest.approx(min(r[1] for r in reports[1:]))    # best J of the joint stage

    class Stop(Exception):
        pass

    def stopper(step, val_J, flags):
        raise Stop

    with pytest.raises(Stop):
        tr._train({**cfg, "output_root": str(Path(cfg["output_root"]) / "again"),
                   "persist_root": str(Path(cfg["output_root"]) / "again")}, report_fn=stopper)


# ----------------------------------------------------------------------------- refusals
def test_wrong_source_hash_refuses_before_creating_anything(cfg, tmp_path):
    cfg["sources"]["expected_sha256"] = dict(cfg["sources"]["expected_sha256"], salt="0" * 64)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        run_training(cfg)
    assert not (tmp_path / "out").exists()


def test_bad_budgets_and_batch_fail_before_any_folder(cfg, tmp_path):
    with pytest.raises(ValueError, match="multiple of 4"):
        run_training({**cfg, "train": {**cfg["train"], "batch_size": 10}})
    with pytest.raises(ValueError, match="TBD"):
        run_training({**cfg, "train": {**cfg["train"], "joint_epochs": "TBD_AFTER_BENCHMARK"}})
    with pytest.raises(ValueError, match="lambda_s must be 1"):
        run_training({**cfg, "train": {**cfg["train"], "lambda_s": 0.5}})
    with pytest.raises(ValueError, match="multiple of 4"):
        run_training({**cfg, "train": {**cfg["train"], "val_subset": 10}})
    assert not (tmp_path / "out").exists()


# --------------------------------------------------------------------------------- benchmark
def test_measure_step_times_on_cpu(cfg, t3_sources):
    paths, sha = t3_sources
    cfg["trial_val_subset"] = 16
    before = {name: sha256_file(p) for name, p in paths.items()}
    res = measure_step_times(cfg, batch=8, amp=True, steps=2, warmup=1, sources=paths)
    assert set(res) == {"warmup_s_per_step", "joint_s_per_step", "val_full_s", "val_subset_s", "peak_mb", "params"}
    assert all(res[k] > 0 for k in ("warmup_s_per_step", "joint_s_per_step", "val_full_s", "val_subset_s"))
    assert res["peak_mb"] == 0.0                                    # no CUDA: CUDA_VISIBLE_DEVICES=-1 in these tests
    fresh = SoftMoE.load_from_task2(paths, tau=2.0, expected_sha256=sha)
    assert res["params"] == count_parameters(fresh, trainable_only=False)
    again = measure_step_times(cfg, batch=8, amp=False, steps=1, warmup=0)                    # sources=None: find_sources(cfg)
    assert again["params"] == res["params"]
    assert {name: sha256_file(p) for name, p in paths.items()} == before


# ----------------------------------------------------------------------------- locked test set
def test_task3_training_never_opens_the_test_split(cfg, tmp_path, monkeypatch):
    log = tmp_path / "test_access.log"
    monkeypatch.setattr(pets_dataset, "TEST_LOG_PATH", log)
    run_training(cfg)
    assert not log.exists()
    text = Path(tr.__file__).read_text(encoding="utf-8")
    assert "final_test=True" not in text and "pets_test_manifest" not in text
    assert 'split="test"' not in text and "split='test'" not in text
