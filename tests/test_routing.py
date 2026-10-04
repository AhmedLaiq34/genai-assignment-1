"""Task 2 hard routing and evaluation: fixture models (random weights), stand-in specialists, tiny synthetic data."""
import json

import pandas as pd
import pytest
import torch
from torch import nn

from genai.models.autoencoder import UniversalAE
from genai.models.classifier import CorruptionClassifier
from genai.pets import dataset as pets_dataset
from genai.tasks.task2 import evaluation
from genai.tasks.task2.evaluate import run_evaluation
from genai.tasks.task2.evaluation import evaluate_task2
from genai.tasks.task2.routing import HardRoutedSystem, load_task2_models
from t2_fixtures import make_fixture_checkpoints


# ----------------------------------------------------------------------------- stand-in models
class FixedClassifier(nn.Module):
    """Classifier stand-in: always votes for the class ids in `forced` (one per image of the batch)."""

    def __init__(self, forced):
        super().__init__()
        self.forced = torch.as_tensor(forced)

    def forward(self, x):
        return nn.functional.one_hot(self.forced[:len(x)], 4).float() * 10.0


class SequenceClassifier(nn.Module):
    """Classifier stand-in for a whole evaluation: returns the labels in manifest order, shifted by `shift`
    (shift 0 = perfect classifier, shift 1 = wrong on every single image)."""

    def __init__(self, labels, shift=0):
        super().__init__()
        self.labels, self.shift, self.position = torch.as_tensor(labels), shift, 0

    def forward(self, x):
        ids = (self.labels[self.position:self.position + len(x)] + self.shift) % 4
        self.position += len(x)
        return nn.functional.one_hot(ids, 4).float() * 10.0


class CountingSpecialist(nn.Module):
    """Specialist stand-in: paints its whole output with `value` and counts calls / images seen."""

    def __init__(self, value):
        super().__init__()
        self.value, self.calls, self.images = value, 0, 0

    def forward(self, x):
        self.calls += 1
        self.images += len(x)
        return torch.full_like(x, self.value)


class ForbiddenSpecialist(nn.Module):
    """Specialist stand-in that must never run."""

    def forward(self, x):
        raise AssertionError("a specialist was called for an image that should bypass it")


def stand_in_system(forced_classes, specialists=None):
    specialists = specialists or {"salt": CountingSpecialist(0.1), "blur": CountingSpecialist(0.2),
                                  "occlusion": CountingSpecialist(0.3)}
    return HardRoutedSystem({"classifier": FixedClassifier(forced_classes), **specialists}), specialists


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def ckpts(tmp_path_factory):
    return make_fixture_checkpoints(tmp_path_factory.mktemp("t2_ckpts"))


def make_cfg(root, tmp_path, **eval_cfg):
    return {"data_root": str(root), "output_root": str(tmp_path / "out"), "eval": dict(eval_cfg)}


# ----------------------------------------------------------------------------- loading
def test_load_task2_models_rebuilds_four_models(ckpts):
    models = load_task2_models(ckpts)
    assert set(models) == {"classifier", "salt", "blur", "occlusion"}
    assert isinstance(models["classifier"], CorruptionClassifier)
    assert all(isinstance(models[k], UniversalAE) for k in ("salt", "blur", "occlusion"))
    assert not any(m.training for m in models.values())             # all in eval mode
    assert models["classifier"](torch.rand(2, 3, 128, 128)).shape == (2, 4)


def test_load_rejects_swapped_or_wrong_checkpoints(ckpts):
    with pytest.raises(ValueError, match="cond_id"):                # the salt checkpoint given as blur
        load_task2_models({**ckpts, "blur": ckpts["salt"]})
    with pytest.raises(ValueError, match="component"):              # a specialist given as the classifier
        load_task2_models({**ckpts, "classifier": ckpts["salt"]})
    with pytest.raises(ValueError, match="component"):              # the classifier given as a specialist
        load_task2_models({**ckpts, "occlusion": ckpts["classifier"]})
    with pytest.raises(ValueError, match="missing"):
        load_task2_models({"classifier": ckpts["classifier"]})


# ----------------------------------------------------------------------------- routing
def test_clean_uses_identity_bypass_and_never_calls_a_specialist():
    forbidden = {k: ForbiddenSpecialist() for k in ("salt", "blur", "occlusion")}
    system, _ = stand_in_system([0, 0, 0, 0], forbidden)
    x = torch.rand(4, 3, 128, 128)
    for mode, true_cond in (("oracle", torch.zeros(4, dtype=torch.long)), ("predicted", None)):
        restored, routes, _probs = system.route(x, true_cond, mode)    # would raise if a specialist ran
        assert torch.equal(restored, x) and routes.tolist() == [0, 0, 0, 0]


def test_each_corrupted_class_calls_exactly_its_specialist_once():
    system, spec = stand_in_system([2, 2, 2])                       # three blurred images
    x = torch.rand(3, 3, 128, 128)
    restored, routes, _ = system.route(x, mode="predicted")
    assert routes.tolist() == [2, 2, 2]
    assert (spec["salt"].calls, spec["blur"].calls, spec["occlusion"].calls) == (0, 1, 0)
    assert spec["blur"].images == 3 and torch.all(restored == 0.2)


def test_mixed_batch_is_scattered_back_to_the_right_positions():
    routes_in = [0, 1, 2, 3, 3, 2, 1, 0]
    system, spec = stand_in_system(routes_in)
    x = torch.rand(8, 3, 128, 128)
    restored, routes, probs = system.route(x, mode="predicted")
    assert routes.tolist() == routes_in and probs.shape == (8, 4)
    for i, r in enumerate(routes_in):
        expected = x[i] if r == 0 else torch.full_like(x[i], {1: 0.1, 2: 0.2, 3: 0.3}[r])
        assert torch.equal(restored[i], expected), f"image {i} (route {r}) was written to the wrong place"
    assert all(s.calls == 1 and s.images == 2 for s in spec.values())    # each specialist ran once, on 2 images


def test_oracle_and_predicted_routes_differ_when_the_classifier_is_wrong():
    true_cond = torch.tensor([0, 1, 2, 3])
    system, _ = stand_in_system([1, 1, 1, 1])                       # the classifier always says "salt"
    x = torch.rand(4, 3, 128, 128)
    out_o, routes_o, _ = system.route(x, true_cond, "oracle")
    out_p, routes_p, _ = system.route(x, mode="predicted")
    assert routes_o.tolist() == [0, 1, 2, 3] and routes_p.tolist() == [1, 1, 1, 1]
    assert torch.equal(out_o[0], x[0]) and torch.all(out_p[0] == 0.1)    # clean image misrouted to the salt expert
    assert torch.equal(out_o[1], out_p[1])                               # image 1 is routed the same way in both
    with pytest.raises(ValueError, match="true_cond"):
        system.route(x, None, "oracle")
    with pytest.raises(ValueError, match="mode"):
        system.route(x, true_cond, "sideways")


def test_real_fixture_models_route_and_stay_in_range(ckpts):
    system = HardRoutedSystem(load_task2_models(ckpts))
    x = torch.rand(6, 3, 128, 128)
    restored, routes, probs = system.route(x, torch.tensor([0, 1, 2, 3, 1, 2]), "oracle")
    assert restored.shape == x.shape and routes.tolist() == [0, 1, 2, 3, 1, 2]
    assert torch.equal(restored[0], x[0]) and not torch.equal(restored[1], x[1])
    assert restored.min() >= 0 and restored.max() <= 1 and torch.allclose(probs.sum(1), torch.ones(6))


# ----------------------------------------------------------------------------- evaluation
@pytest.fixture(scope="module")
def val_run(tiny_pets_root, ckpts, tmp_path_factory):
    """One evaluate_task2 run on the tiny val manifest (64 rows), with the test log redirected."""
    tmp = tmp_path_factory.mktemp("t2_eval")
    log = tmp / "test_access.log"
    mp = pytest.MonkeyPatch()
    mp.setattr(pets_dataset, "TEST_LOG_PATH", log)
    out = evaluate_task2(make_cfg(tiny_pets_root, tmp), ckpts)
    yield out, log
    mp.undo()


def test_evaluate_writes_every_file_and_column(val_run):
    out, _ = val_run
    assert out.name.endswith("_val")
    for name in ("per_image.csv", "table_cond_severity_oracle.csv", "table_cond_severity_predicted.csv",
                 "classifier_report.json", "confusion_normalised.csv", "confusion_normalised.png",
                 "routing_failures.csv", "misroute_confusion.csv", "summary.json"):
        assert (out / name).exists(), name
    df = pd.read_csv(out / "per_image.csv")
    required = {"image_id", "cond", "severity", "true_class", "predicted_class", "oracle_route",
                "predicted_route", "MAE", "SSIM", "PSNR", "J", "mode", "J_input", "SSIM_input"}
    assert required <= set(df.columns)
    assert len(df) == 128 and set(df["mode"]) == {"oracle", "predicted"}      # 64 manifest rows x 2 modes
    for name in ("oracle", "predicted"):
        table = pd.read_csv(out / f"table_cond_severity_{name}.csv")
        assert {"clean", "salt_pepper", "gaussian_blur", "occlusion", "overall"} <= set(table["cond"])
        assert {"MAE", "SSIM", "PSNR", "J", "J_input", "SSIM_input", "count"} <= set(table.columns)
    report = json.loads((out / "classifier_report.json").read_text())
    assert {"accuracy", "macro_f1", "per_class", "confusion_normalised"} <= set(report)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n_rows"] == 64 and summary["smoke"] is True
    assert {"oracle", "predicted", "classifier_accuracy", "n_misroutes", "checkpoints", "input_baseline"} <= set(summary)
    by_cond = summary["oracle_vs_input_by_condition"]                          # the "do nothing" comparison
    assert set(by_cond) == {"clean", "salt_pepper", "gaussian_blur", "occlusion"}
    assert by_cond["clean"]["J_input"] == 0 and by_cond["salt_pepper"]["J_input"] > 0
    assert all(len(c["sha256"]) == 64 for c in summary["checkpoints"].values())


def test_both_modes_use_identical_inputs_and_clean_is_untouched(val_run):
    out, _ = val_run
    df = pd.read_csv(out / "per_image.csv")
    oracle, pred = df[df["mode"] == "oracle"].reset_index(drop=True), df[df["mode"] == "predicted"].reset_index(drop=True)
    assert oracle["image_id"].tolist() == pred["image_id"].tolist() and oracle["cond"].tolist() == pred["cond"].tolist()
    same_route = oracle["oracle_route"] == oracle["predicted_route"]
    assert same_route.any()                                                    # the check below is not empty
    for metric in ("MAE", "SSIM", "PSNR", "J"):
        assert (oracle.loc[same_route, metric] == pred.loc[same_route, metric]).all()
    clean_oracle = oracle[oracle["cond"] == "clean"]                           # identity bypass: output == target
    assert (clean_oracle["MAE"] == 0).all() and (clean_oracle["route_used"] == "identity").all()


def test_routing_failures_are_the_misroutes(val_run):
    out, _ = val_run
    df = pd.read_csv(out / "per_image.csv")
    oracle = df[df["mode"] == "oracle"]
    n_wrong = int((oracle["oracle_route"] != oracle["predicted_route"]).sum())
    failures = pd.read_csv(out / "routing_failures.csv")
    assert len(failures) == n_wrong
    assert (failures["oracle_route"] != failures["predicted_route"]).all()
    assert failures["J_drop"].tolist() == sorted(failures["J_drop"], reverse=True)
    assert (failures["J_drop"] - (failures["predicted_J"] - failures["oracle_J"])).abs().max() < 1e-9
    confusion = pd.read_csv(out / "misroute_confusion.csv", index_col=0)
    assert int(confusion.to_numpy().sum()) == n_wrong and (confusion.to_numpy().diagonal() == 0).all()
    assert (out / ("routing_failures_worst.png" if n_wrong else "routing_failures_worst_NOTE.txt")).exists()


def test_test_manifest_stays_unopened_without_final_test(val_run):
    _out, log = val_run
    assert not log.exists()


def test_final_test_flag_opens_the_test_manifest_and_logs_it(tiny_pets_root, ckpts, tmp_path, monkeypatch):
    log = tmp_path / "test_access.log"
    monkeypatch.setattr(pets_dataset, "TEST_LOG_PATH", log)
    out = evaluate_task2(make_cfg(tiny_pets_root, tmp_path), ckpts, final_test=True)   # synthetic test split
    assert out.name.endswith("_test") and log.exists()
    assert len(pd.read_csv(out / "per_image.csv")) == 2 * 80                 # 8 test images x 10 rows x 2 modes


def test_max_rows_limits_the_evaluation(tiny_pets_root, ckpts, tmp_path):
    out = evaluate_task2(make_cfg(tiny_pets_root, tmp_path, max_rows=8, batch_size=3), ckpts)
    assert len(pd.read_csv(out / "per_image.csv")) == 16


def _patched_models(monkeypatch, ckpts, ds_labels, shift):
    """evaluate_task2 with the real specialists but a stand-in classifier that is perfect (shift 0) or always wrong."""
    models = load_task2_models(ckpts)
    models["classifier"] = SequenceClassifier(ds_labels, shift)
    monkeypatch.setattr(evaluation, "load_task2_models", lambda paths, device="cpu": {k: m.to(device) for k, m in models.items()})


def _val_labels(root):
    from genai.pets.dataset import PetsManifestDataset
    ds = PetsManifestDataset(root / "manifests" / "pets_val_manifest.jsonl", "val", False, root)
    return [r["cond_id"] for r in ds.rows]


def test_zero_misroutes_does_not_crash(tiny_pets_root, ckpts, tmp_path, monkeypatch):
    _patched_models(monkeypatch, ckpts, _val_labels(tiny_pets_root), shift=0)
    out = evaluate_task2(make_cfg(tiny_pets_root, tmp_path), ckpts)
    assert len(pd.read_csv(out / "routing_failures.csv")) == 0
    assert (out / "routing_failures_worst_NOTE.txt").exists() and not (out / "routing_failures_worst.png").exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n_misroutes"] == 0 and summary["classifier_accuracy"] == 1.0
    assert summary["oracle"] == summary["predicted"]                          # same routes -> same numbers
    assert (pd.read_csv(out / "misroute_confusion.csv", index_col=0).to_numpy() == 0).all()


def test_all_misrouted_gives_failure_grid(tiny_pets_root, ckpts, tmp_path, monkeypatch):
    _patched_models(monkeypatch, ckpts, _val_labels(tiny_pets_root), shift=1)
    out = evaluate_task2(make_cfg(tiny_pets_root, tmp_path, n_worst=3), ckpts)
    failures = pd.read_csv(out / "routing_failures.csv")
    assert len(failures) == 64 and (out / "routing_failures_worst.png").exists()
    assert json.loads((out / "summary.json").read_text())["classifier_accuracy"] == 0.0
    # a clean image sent to an expert (true class 0 -> predicted 1) is a failure like any other
    assert ((failures["true_class"] == 0) & (failures["predicted_class"] == 1)).any()


# ----------------------------------------------------------------------------- dispatcher
def test_dispatcher_runs_with_checkpoints_from_the_config(tiny_pets_root, ckpts, tmp_path):
    cfg = make_cfg(tiny_pets_root, tmp_path, max_rows=8)
    cfg["eval"]["checkpoints"] = {k: str(v) for k, v in ckpts.items()}
    out = run_evaluation(cfg, None)
    assert (out / "summary.json").exists()


def test_dispatcher_names_the_missing_checkpoint(tiny_pets_root, ckpts, tmp_path):
    cfg = make_cfg(tiny_pets_root, tmp_path)
    cfg["eval"]["checkpoints"] = {k: str(v) for k, v in ckpts.items() if k != "blur"}
    with pytest.raises(ValueError, match="blur"):
        run_evaluation(cfg, None)
