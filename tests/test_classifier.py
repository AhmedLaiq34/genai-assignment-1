"""Task 2 corruption classifier: model, balanced batches, training / resume, metrics, evaluation, study.

Everything runs on CPU with the tiny SYNTHETIC dataset from tests/conftest.py (fixture `tiny_pets_root`:
64 train images, 16 val images -> 64 val manifest rows). No real data, no real training.
"""
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest
import torch

from genai.common.checkpoint import load_checkpoint
from genai.common.constants import CLASS_NAMES
from genai.models.autoencoder import count_parameters
from genai.models.classifier import CorruptionClassifier
from genai.pets import dataset as pets_dataset
from genai.pets.dataset import resolve_data_paths
from genai.tasks.task1 import config as t1cfg
from genai.tasks.task1.tune import dry_run_overrides
from genai.tasks.task2 import classifier as t2c
from genai.tasks.task2.classifier import (classification_metrics, evaluate_classifier, load_classifier,
                                          save_confusion_matrix, study_classifier, train_classifier)
from genai.tasks.task2.runs import find_latest_checkpoint


def make_cfg(root, tmp_path, **train):
    """The real classifier YAML, shrunk to a tiny CPU model on the synthetic data."""
    cfg = t1cfg.load_config("configs/task2_classifier.yaml", "local")
    cfg.update(device="test", data_root=str(root), output_root=str(tmp_path / "out"),
               persist_root=str(tmp_path / "out"), num_workers=0)
    cfg["model"].update(channels=[4, 8], dropout=0.1)
    cfg["train"].update(batch_size=8, epochs=2, sample_every_epochs=1, val_workers=0, **train)
    cfg["run"]["smoke"] = True
    return cfg


def read_rows(run_dir):
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]


# ------------------------------------------------------------------------------------------ model
def test_output_is_logits_of_shape_n_by_4():
    model = CorruptionClassifier(channels=[4, 8], dropout=0.0).eval()
    x = torch.rand(5, 3, 128, 128)
    out = model(x)
    assert out.shape == (5, 4)
    # logits only: the model contains no Softmax (that lives in evaluation and in the backend)
    assert not any(isinstance(m, torch.nn.Softmax) for m in model.modules())
    assert torch.allclose(model(x), model(x))                          # eval mode: deterministic
    assert count_parameters(model) > 0


def test_from_config_ignores_other_keys_and_records_hparams():
    cfg = {"channels": [4, 8, 16], "dropout": 0.3, "num_classes": 4, "something_else": 1}
    model = CorruptionClassifier.from_config(cfg)
    assert model.hparams == {"channels": [4, 8, 16], "dropout": 0.3, "num_classes": 4}
    assert len(model.features) == 3                                    # one block per entry of channels
    with pytest.raises(ValueError):
        CorruptionClassifier(channels=[])


def test_onnx_export_opset17_matches_torch(tmp_path):
    import onnxruntime as ort
    model = CorruptionClassifier(channels=[4, 8], dropout=0.2).eval()
    path = tmp_path / "t2_classifier.onnx"
    torch.onnx.export(model, torch.zeros(1, 3, 128, 128), str(path), input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    x = torch.rand(3, 3, 128, 128)                                     # another batch size: the batch axis is dynamic
    with torch.no_grad():
        expected = model(x).numpy()
    got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(["logits"], {"input": x.numpy()})[0]
    assert got.shape == (3, 4) and np.abs(got - expected).max() < 1e-4


# ------------------------------------------------------------------------------ balanced batches
def test_every_training_batch_has_exact_class_counts(tiny_pets_root, tmp_path):
    cfg = make_cfg(tiny_pets_root, tmp_path)
    _ds, _sampler, loader = t2c.build_train_loader(cfg, resolve_data_paths(cfg), batch_size=8)
    batches = []
    while len(batches) < 20:                                           # 8 batches per epoch -> 3 epochs
        for _corrupted, _clean, cond, _sev in loader:
            batches.append(cond)
            if len(batches) == 20:
                break
    for cond in batches:
        assert torch.bincount(cond, minlength=4).tolist() == [2, 2, 2, 2]     # exactly B/4 per class


def test_batch_size_not_multiple_of_4_raises(tiny_pets_root, tmp_path):
    cfg = make_cfg(tiny_pets_root, tmp_path)
    cfg["train"]["batch_size"] = 10
    with pytest.raises(ValueError, match="multiple of 4"):
        train_classifier(cfg)
    assert not (tmp_path / "out").exists()                             # failed before creating any folder
    with pytest.raises(ValueError, match="multiple of 4"):
        t2c.build_train_loader(cfg, resolve_data_paths(cfg), batch_size=6)


def test_tbd_epochs_need_max_steps(tiny_pets_root, tmp_path):
    cfg = make_cfg(tiny_pets_root, tmp_path)
    cfg["train"]["epochs"] = "TBD_AFTER_BENCHMARK"
    with pytest.raises(ValueError, match="TBD"):
        train_classifier(cfg)
    cfg["train"]["max_steps"] = 3                                      # a step cap is enough
    run_dir = train_classifier(cfg)
    assert load_checkpoint(run_dir / "ckpt_last.pt")["global_step"] == 3


# ------------------------------------------------------------------------------------- training
def test_training_loss_decreases_on_tiny_subset(tiny_pets_root, tmp_path):
    cfg = make_cfg(tiny_pets_root, tmp_path, train_subset=16, val_subset=16)
    cfg["train"].update(epochs=15, lr=3e-3, scheduler="none", weight_decay=0.0)    # 2 steps per epoch -> 30 steps
    rows = read_rows(train_classifier(cfg))
    assert len(rows) == 15
    first, last = np.mean([r["train_loss"] for r in rows[:2]]), np.mean([r["train_loss"] for r in rows[-2:]])
    assert last < first - 0.05, (first, last)                          # clear decrease, not noise


def test_train_writes_all_files_and_checkpoint_layout(tiny_pets_root, tmp_path):
    cfg = make_cfg(tiny_pets_root, tmp_path, train_subset=32, val_subset=32)
    seen = []
    run_dir = train_classifier(cfg, on_checkpoint=seen.append)
    assert run_dir.parent.name == "task2_classifier" and run_dir.name.endswith("_test_t2cls_smoke")
    for name in ("config.yaml", "ckpt_last.pt", "ckpt_best.pt", "metrics.jsonl",
                 "confusion_val_best.csv", "confusion_val_best.png"):
        assert (run_dir / name).exists()
    assert len(list((run_dir / "samples").glob("confusion_epoch*.png"))) == 2
    assert run_dir / "ckpt_last.pt" in seen and run_dir / "ckpt_best.pt" in seen    # hook called after each save

    rows = read_rows(run_dir)
    assert [r["epoch"] for r in rows] == [1, 2] and rows[-1]["global_step"] == 8
    assert {"val_accuracy", "val_macro_precision", "val_macro_recall", "val_macro_f1", "per_class"} <= set(rows[0])
    assert set(rows[0]["per_class"]) == set(CLASS_NAMES)
    assert set(rows[0]["per_class"]["clean"]) == {"precision", "recall", "f1", "support"}
    cm = pd.read_csv(run_dir / "confusion_val_best.csv", index_col=0)
    assert list(cm.columns) == list(CLASS_NAMES) and list(cm.index) == list(CLASS_NAMES)

    # best checkpoint = highest macro-F1 (best_metric is the macro-F1)
    best = load_checkpoint(run_dir / "ckpt_best.pt")
    assert best["best_metric"] == pytest.approx(max(r["val_macro_f1"] for r in rows))
    for key in ("model", "optimizer", "scheduler", "scaler", "epoch", "global_step", "best_metric", "config",
                "seed", "split_sha256", "manifest_sha256", "rng"):
        assert key in best

    # the checkpoint config rebuilds the model and the weights load
    config = best["config"]
    assert config["component"] == "classifier" and config["run_id"] == run_dir.name
    assert config["model"] == {"channels": [4, 8], "dropout": 0.1, "num_classes": 4}
    rebuilt = CorruptionClassifier.from_config(config["model"])
    rebuilt.load_state_dict(best["model"])
    model, ckpt = load_classifier(run_dir / "ckpt_best.pt")
    assert not model.training and ckpt["global_step"] == best["global_step"]
    x = torch.rand(2, 3, 128, 128)
    assert torch.allclose(model(x), rebuilt.eval()(x))


def test_resume_continues_global_step_and_restores_rng(tiny_pets_root, tmp_path, monkeypatch):
    cfg = make_cfg(tiny_pets_root, tmp_path, pause_after_steps=6)      # 8 steps per epoch: stop inside epoch 1
    run_dir = train_classifier(cfg)
    paused = load_checkpoint(run_dir / "ckpt_last.pt")
    assert paused["global_step"] == 6 and paused["epoch"] == 0 and paused["step_in_epoch"] == 6
    assert not (run_dir / "metrics.jsonl").exists()                    # no epoch was validated yet
    assert find_latest_checkpoint(cfg) == run_dir / "ckpt_last.pt"     # what resume="auto" finds

    restored = []
    real = t2c.restore_rng_state
    monkeypatch.setattr(t2c, "restore_rng_state", lambda s: (restored.append(set(s)), real(s)))
    cfg["train"]["pause_after_steps"] = None
    run_dir2 = train_classifier(cfg, resume=str(run_dir / "ckpt_last.pt"))
    assert run_dir2 == run_dir                                         # same run continues
    assert restored == [{"python", "numpy", "torch", "cuda"}]          # all RNG states restored
    final = load_checkpoint(run_dir / "ckpt_last.pt")
    assert final["global_step"] == 16 and final["epoch"] == 2 and final["step_in_epoch"] == 0
    assert [r["epoch"] for r in read_rows(run_dir)] == [1, 2]

    bad = json.loads(json.dumps(cfg))
    bad["train"]["lr"] = 0.5
    with pytest.raises(ValueError, match="lr"):                        # a changed config cannot resume
        train_classifier(bad, resume=str(run_dir / "ckpt_last.pt"))


# -------------------------------------------------------------------------------------- metrics
def test_classification_metrics_agree_with_sklearn():
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3])
    y_pred = np.array([0, 0, 1, 0, 1, 1, 0, 2, 2, 3, 2, 3, 2, 3])
    m = classification_metrics(y_true, y_pred)

    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    assert m["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert m["macro_precision"] == pytest.approx(p)
    assert m["macro_recall"] == pytest.approx(r)
    assert m["macro_f1"] == pytest.approx(f)
    p4, r4, f4, s4 = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0)
    for k, name in enumerate(CLASS_NAMES):
        pc = m["per_class"][name]
        assert (pc["precision"], pc["recall"], pc["f1"], pc["support"]) == pytest.approx((p4[k], r4[k], f4[k], s4[k]))
    counts = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    assert m["confusion_counts"] == counts.tolist()
    norm = np.array(m["confusion_normalised"])
    assert norm.shape == (4, 4) and np.allclose(norm.sum(axis=1), 1.0)
    assert np.allclose(norm, counts / counts.sum(axis=1, keepdims=True))
    json.dumps(m)                                                      # everything is JSON-serialisable


def test_metrics_row_without_support_stays_zero():
    m = classification_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))   # classes 2 and 3 never occur
    norm = np.array(m["confusion_normalised"])
    assert norm[2].tolist() == [0.0] * 4 and norm[3].tolist() == [0.0] * 4
    assert m["per_class"]["occlusion"]["support"] == 0
    assert np.allclose(norm[:2].sum(axis=1), 1.0)


def test_save_confusion_matrix_writes_csv_and_png(tmp_path):
    cm = np.array([[0.9, 0.1, 0, 0], [0, 1, 0, 0], [0, 0, 0.5, 0.5], [0.25, 0, 0, 0.75]])
    save_confusion_matrix(cm, tmp_path / "cm.csv", tmp_path / "sub" / "cm.png")
    df = pd.read_csv(tmp_path / "cm.csv", index_col=0)
    assert list(df.index) == list(CLASS_NAMES) and list(df.columns) == list(CLASS_NAMES)
    assert np.allclose(df.to_numpy(), cm)
    assert (tmp_path / "sub" / "cm.png").stat().st_size > 0


# ------------------------------------------------------------------------------------ evaluation
def test_evaluate_val_outputs_and_test_stays_locked(tiny_pets_root, tmp_path, monkeypatch):
    log = tmp_path / "test_access.log"
    monkeypatch.setattr(pets_dataset, "TEST_LOG_PATH", log)
    cfg = make_cfg(tiny_pets_root, tmp_path, train_subset=16, val_subset=16)
    run_dir = train_classifier(cfg)
    out = evaluate_classifier(cfg, str(run_dir / "ckpt_best.pt"))      # default: val manifest
    assert out.parent.name == "task2" and out.name == f"classifier_{run_dir.name}_val"
    assert not log.exists()                                            # the test split was never opened
    for name in ("classification_report.json", "confusion_normalised.csv", "confusion_normalised.png",
                 "confusion_counts.csv", "predictions.csv"):
        assert (out / name).exists()
    report = json.loads((out / "classification_report.json").read_text())
    assert report["n_rows"] == 64 and report["split"] == "val"
    assert {"accuracy", "macro_precision", "macro_recall", "macro_f1", "per_class",
            "confusion_counts", "confusion_normalised"} <= set(report)
    preds = pd.read_csv(out / "predictions.csv")
    assert len(preds) == 64 and {"image_id", "cond", "true_class", "predicted_class", "p_clean", "p_occlusion"} <= set(preds.columns)
    assert report["accuracy"] == pytest.approx((preds["true_class"] == preds["predicted_class"]).mean())
    assert preds["true_class"].value_counts().to_dict() == {0: 16, 1: 16, 2: 16, 3: 16}   # all 4 conditions


def test_training_and_study_code_never_open_the_test_split():
    text = Path(t2c.__file__).read_text(encoding="utf-8")
    assert "final_test=True" not in text and "pets_test_manifest" not in text
    assert 'split="test"' not in text and "split='test'" not in text


# ----------------------------------------------------------------------------------------- study
def test_search_space_refuses_batch_not_multiple_of_4():
    space = [{"name": "batch", "type": "categorical", "choices": [4, 6]}]
    with pytest.raises(ValueError, match="multiples of 4"):
        t2c._check_search_space(space)
    with pytest.raises(ValueError, match="conv_channels"):
        t2c._check_search_space([{"name": "conv_channels", "type": "categorical", "choices": ["1-2"]}])
    t2c._check_search_space(t1cfg.load_config("configs/task2_classifier.yaml", "local")["tuned_params"])   # the YAML is valid


def test_make_trial_config_maps_values():
    cfg = t1cfg.load_config("configs/task2_classifier.yaml", "local")
    cfg.update(epochs_per_trial=3, trial_train_subset=32)
    values = {"lr": 0.01, "batch": 8, "conv_channels": "16-32-64", "dropout": 0.25, "weight_decay": 1e-3}
    tcfg = t2c.make_trial_config(cfg, values, 7)
    assert tcfg["train"]["lr"] == 0.01 and tcfg["train"]["batch_size"] == 8 and tcfg["train"]["weight_decay"] == 1e-3
    assert tcfg["model"]["channels"] == [16, 32, 64] and tcfg["model"]["dropout"] == 0.25
    assert tcfg["train"]["epochs"] == 3 and tcfg["train"]["train_subset"] == 32
    assert cfg["train"]["epochs"] != 3 and cfg["model"]["channels"] == [16, 32, 64, 128]     # base config untouched
    assert tcfg["run_id"].endswith("t2_classifier-trial7") and Path(tcfg["output_root"]).name == "trials"


def test_study_dry_run(tiny_pets_root, tmp_path):
    cfg = dry_run_overrides(make_cfg(tiny_pets_root, tmp_path), tmp_path / "dry")
    cfg["tuned_params"] = [dict(p) for p in cfg["tuned_params"]]
    for p in cfg["tuned_params"]:
        if p["name"] == "batch":
            p["choices"] = [4, 8]
        if p["name"] == "conv_channels":
            p["choices"] = ["16-32-64"]                                # keep the dry run fast
    cfg["trial_train_subset"] = 16
    cfg["train"]["val_subset"] = 16
    cfg.update(n_trials="TBD_AFTER_BENCHMARK", epochs_per_trial="TBD_AFTER_BENCHMARK")   # the guard (the YAML holds real budgets, D44)
    with pytest.raises(ValueError, match="TBD"):                       # budgets stay TBD unless given
        study_classifier(cfg)
    cfg.update(n_trials=2, epochs_per_trial=1)
    out = study_classifier(cfg)
    assert cfg["study"] == "t2_classifier_dryrun" and str(tmp_path / "dry") in str(out)

    trials = pd.read_csv(out / "trials.csv")
    assert len(trials) == 2 and trials["state"].eq("COMPLETE").all()
    assert {"params_lr", "params_batch", "params_conv_channels", "params_dropout", "params_weight_decay"} <= set(trials.columns)
    assert trials["params_batch"].isin([4, 8]).all() and trials["value"].between(0, 1).all()   # macro-F1 values

    db = tmp_path / "dry" / "optuna" / "t2_classifier_dryrun.db"
    assert db.exists()
    study = optuna.load_study(study_name=cfg["study"], storage=f"sqlite:///{db.as_posix()}")
    assert study.direction == optuna.study.StudyDirection.MAXIMIZE and len(study.trials) == 2
    assert len(list((tmp_path / "dry" / "trials" / "runs" / "task2_classifier").glob("*_smoke"))) == 2   # trial runs kept apart
    assert not (tmp_path / "dry" / "runs").exists()                    # nothing in the folder resume="auto" searches
    study_classifier(cfg)                                              # continues the same study: no new trials
    study = optuna.load_study(study_name=cfg["study"], storage=f"sqlite:///{db.as_posix()}")
    assert len(study.trials) == 2
