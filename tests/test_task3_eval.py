"""Task 3 evaluation (soft mixture of experts): tiny random fixture models, tiny synthetic data.

The numbers mean nothing (random weights); the tests prove that every output file is written, that the
columns and tables have the promised layout, that all systems are scored on the same tensors, and that the
locked test set is only touched with final_test=True (and then only through a tmp access log).
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from genai.common.checkpoint import build_checkpoint, save_checkpoint, sha256_file
from genai.models.moe import SoftMoE
from genai.pets import dataset as pets_dataset
from genai.tasks.task3 import BRANCH_NAMES
from genai.tasks.task3 import evaluate as t3_eval
from genai.tasks.task3 import sources as t3_sources
from genai.tasks.task3.evaluate import evaluate_task3, load_soft_moe, run_evaluation
from genai.tasks.task3.sources import source_record
from t3_fixtures import make_t3_sources

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import t3_report_assets  # noqa: E402

REAL_TEST_LOG = Path(__file__).resolve().parents[1] / "artifacts" / "test_access.log"
W = ["w_identity", "w_salt", "w_blur", "w_occlusion"]
PER_IMAGE_COLUMNS = (["row", "image_id", "cond", "severity", "true_class"] + W
                     + ["dominant", "max_w", "entropy", "MAE", "SSIM", "PSNR", "J", "SSIM_input", "J_input",
                        "J_t1", "J_t2_oracle", "J_t2_predicted", "t2_predicted_class",
                        "J_branch_identity", "J_branch_salt", "J_branch_blur", "J_branch_occlusion"])


# ----------------------------------------------------------------------------- helpers
def make_t3_checkpoint(out_path, paths, sha, tau: float = 2.0, smoke: bool = True) -> Path:
    """A Task 3 checkpoint in the layout of plan B11, built from fixture sources (what train.py will write).

    The model is the freshly initialised SoftMoE (gate = the classifier, experts = the specialists), so the
    tests know what it must reproduce. `smoke` marks it like a run from smoke / fixture sources.
    """
    model = SoftMoE.load_from_task2(paths, tau=tau, expected_sha256=sha).eval()
    cfg = {"component": "soft_moe", "model": model.model_config(), "stage": "joint",
           "run": {"smoke": smoke}, "run_id": "20260101-0000_test_t3moe" + ("_smoke" if smoke else ""),
           "source_checkpoints": source_record(paths),
           "train": {"lambda_1": 0.8, "lambda_s": 0.2, "lambda_c": 0.1, "lambda_b": 0.01, "joint_lr": 5e-5}}
    save_checkpoint(out_path, build_checkpoint(model, config=cfg, global_step=3))
    return Path(out_path)


def make_cfg(root, tmp_path, **eval_cfg):
    return {"data_root": str(root), "output_root": str(tmp_path / "out"), "eval": dict(eval_cfg)}


def real_log_lines() -> int:
    return len(REAL_TEST_LOG.read_text().splitlines()) if REAL_TEST_LOG.exists() else 0


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def fixture_models(tmp_path_factory):
    """(paths, sha, T3 checkpoint) written once for the whole module."""
    tmp = tmp_path_factory.mktemp("t3_eval_models")
    paths, sha = make_t3_sources(tmp / "sources")
    return paths, sha, make_t3_checkpoint(tmp / "t3_soft_moe.pt", paths, sha)


@pytest.fixture(scope="module")
def val_run(tiny_pets_root, fixture_models, tmp_path_factory):
    """One evaluation on the first 32 val rows, with the test-access log redirected to a tmp file."""
    paths, _sha, ckpt = fixture_models
    tmp = tmp_path_factory.mktemp("t3_eval_run")
    log = tmp / "test_access.log"
    before = real_log_lines()
    mp = pytest.MonkeyPatch()
    mp.setattr(pets_dataset, "TEST_LOG_PATH", log)
    out = evaluate_task3(make_cfg(tiny_pets_root, tmp, max_rows=32, batch_size=16, n_examples=4), ckpt, sources=paths)
    yield out, log, before
    mp.undo()


# ----------------------------------------------------------------------------- loading
def test_load_soft_moe_rebuilds_the_model_without_the_task2_files(fixture_models):
    paths, sha, ckpt_path = fixture_models
    model, ckpt = load_soft_moe(ckpt_path)
    assert isinstance(model, SoftMoE) and not model.training and model.tau == 2.0
    reference = SoftMoE.load_from_task2(paths, tau=2.0, expected_sha256=sha).eval()
    x = torch.rand(3, 3, 128, 128, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.equal(model(x)[0], reference(x)[0])
    assert ckpt["config"]["component"] == "soft_moe"


def test_load_soft_moe_refuses_other_checkpoints(fixture_models):
    paths, _sha, _ckpt = fixture_models
    with pytest.raises(ValueError, match="soft_moe"):
        load_soft_moe(paths["classifier"])                  # a Task 2 checkpoint is not a Task 3 one


# ----------------------------------------------------------------------------- files and columns
def test_every_file_is_written(val_run):
    out, _log, _before = val_run
    assert out.name.endswith("_val") and out.parent.name == "task3"
    for name in ("per_image.csv", "comparison_cond_severity.csv", "weights_by_class_severity.csv",
                 "routing_heatmap.png", "expert_activity.csv", "expert_drift.csv", "examples_dominant.png",
                 "examples_distributed.png", "examples.csv", "summary.json"):
        assert (out / name).exists() and (out / name).stat().st_size > 0, name


def test_per_image_columns_are_exactly_the_planned_ones(val_run):
    out, _log, _before = val_run
    df = pd.read_csv(out / "per_image.csv")
    assert list(df.columns) == PER_IMAGE_COLUMNS
    assert len(df) == 32 and df["row"].tolist() == list(range(32))
    assert set(df["dominant"]) <= set(BRANCH_NAMES)
    assert (df["dominant"] == df[W].idxmax(axis=1).str[2:]).all()
    assert (df["max_w"] - df[W].max(axis=1)).abs().max() < 1e-9
    assert (df["entropy"] >= 0).all() and (df["entropy"] <= torch.log(torch.tensor(4.0)).item() + 1e-6).all()


def test_the_four_weights_sum_to_one(val_run):
    out, _log, _before = val_run
    df = pd.read_csv(out / "per_image.csv")
    assert ((df[W].sum(axis=1) - 1).abs() < 1e-5).all() and (df[W] >= 0).all().all()


def test_clean_rows_are_the_identity_baseline(val_run):
    out, _log, _before = val_run
    df = pd.read_csv(out / "per_image.csv")
    clean = df[df["cond"] == "clean"]
    assert len(clean) > 0 and (clean["J_input"] == 0).all() and (clean["severity"] == "none").all()
    assert (clean["J_t2_oracle"] == 0).all()                       # Task 2 oracle: identity bypass
    # branch 0 of the mixture is the input, so its J is the "do nothing" J on every row
    assert (df["J_branch_identity"] - df["J_input"]).abs().max() < 1e-6


def test_comparison_table_has_the_five_systems(val_run):
    out, _log, _before = val_run
    table = pd.read_csv(out / "comparison_cond_severity.csv")
    for system in ("input", "t1", "t2_oracle", "t2_predicted", "t3"):
        assert f"J_{system}" in table.columns and f"SSIM_{system}" in table.columns, system
    assert {"clean", "salt_pepper", "gaussian_blur", "occlusion", "overall"} <= set(table["cond"])
    overall = table[table["cond"] == "overall"].iloc[0]
    df = pd.read_csv(out / "per_image.csv")
    assert overall["count"] == 32 and abs(overall["J_t3"] - df["J"].mean()) < 1e-9
    assert abs(overall["J_t2_oracle"] - df["J_t2_oracle"].mean()) < 1e-9
    assert abs(overall["J_input"] - df["J_input"].mean()) < 1e-9
    # per condition and severity: a clean row and 3 conditions x severities, plus an "all" row per condition
    assert (table[(table["cond"] == "clean")]["severity"].tolist()) == ["none", "all"]
    assert {"low", "medium", "high"} >= set(table[table["cond"] == "salt_pepper"]["severity"]) - {"all"}


def test_weights_by_class_severity_matches_per_image(val_run):
    out, _log, _before = val_run
    table = pd.read_csv(out / "weights_by_class_severity.csv")
    df = pd.read_csv(out / "per_image.csv")
    assert (table[table["cond"] == "clean"]["severity"] == "none").all() and (table["cond"] == "clean").sum() == 1
    assert table["count"].sum() == len(df) and ((table[W].sum(axis=1) - 1).abs() < 1e-5).all()
    for _, row in table.iterrows():
        group = df[(df["cond"] == row["cond"]) & (df["severity"] == row["severity"])]
        assert len(group) == row["count"] and (group[W].mean() - row[W]).abs().max() < 1e-6


def test_expert_activity_has_one_row_per_branch(val_run):
    out, _log, _before = val_run
    activity = pd.read_csv(out / "expert_activity.csv")
    assert activity["branch"].tolist() == list(BRANCH_NAMES)
    assert {"mean_weight", "argmax_share", "mean_weight_own_class", "max_mean_weight_other_class",
            "inactive", "dominates_unrelated"} <= set(activity.columns)
    assert abs(activity["mean_weight"].sum() - 1) < 1e-5 and abs(activity["argmax_share"].sum() - 1) < 1e-9
    df = pd.read_csv(out / "per_image.csv")
    for k, row in activity.iterrows():
        assert abs(row["mean_weight"] - df[W[k]].mean()) < 1e-6
        assert bool(row["inactive"]) == bool(row["mean_weight"] < 0.02 or row["argmax_share"] == 0)


def test_expert_drift_is_zero_before_any_training(val_run):
    """The fixture T3 model IS the Task 2 system (gate = classifier, experts = specialists): no drift yet."""
    out, _log, _before = val_run
    drift = pd.read_csv(out / "expert_drift.csv")
    assert drift["expert"].tolist() == ["salt", "blur", "occlusion"]
    assert drift["delta"].abs().max() < 1e-5
    assert (drift["n_rows"] > 0).all() and (drift["J_input"] >= 0).all()


# ----------------------------------------------------------------------------- examples
def test_examples_csv_records_rows_and_threshold(val_run):
    out, _log, _before = val_run
    examples = pd.read_csv(out / "examples.csv")
    assert set(examples["figure"]) == {"dominant", "distributed"}
    df = pd.read_csv(out / "per_image.csv").set_index("row")
    for kind, group in examples.groupby("figure"):
        assert 1 <= len(group) <= 4
        assert group["rule"].nunique() == 1 and group["threshold"].nunique() == 1
        assert (group["max_w"] - df.loc[group["row"], "max_w"].to_numpy()).abs().max() < 1e-9
        if group["rule"].iloc[0] == "threshold":              # the rows really satisfy the stated threshold
            op = (lambda a, b: a >= b) if kind == "dominant" else (lambda a, b: a <= b)
            assert op(group["max_w"], group["threshold"]).all()


def test_example_choice_rules():
    """Threshold rule: 2 rows per true class; fallback: the rows with the most extreme max weight."""
    rows = []
    for row in range(80):                                      # 20 rows per class: 5 spread, 5 middle, 10 dominant
        k = row // 4
        rows.append({"row": row, "true_class": row % 4, "max_w": 0.50 if k < 5 else 0.80 if k < 10 else 0.97})
    df = pd.DataFrame(rows)
    chosen, threshold, rule = t3_eval._pick_examples(df, "dominant", 8)
    assert rule == "threshold" and threshold == 0.95 and len(chosen) == 8
    assert chosen["max_w"].ge(0.95).all() and chosen["true_class"].value_counts().eq(2).all()
    chosen, threshold, rule = t3_eval._pick_examples(df, "distributed", 8)
    assert rule == "threshold" and threshold == 0.60 and chosen["max_w"].le(0.60).all()

    few = df[df["max_w"] > 0.7].copy()                         # no row is below 0.60: fallback to the lowest
    chosen, threshold, rule = t3_eval._pick_examples(few, "distributed", 8)
    assert rule == "fallback" and len(chosen) == 8
    assert chosen["max_w"].max() == pytest.approx(threshold)
    assert chosen["max_w"].max() <= few.sort_values("max_w")["max_w"].iloc[7] + 1e-12


# ----------------------------------------------------------------------------- summary
def test_summary_has_gate_vs_classifier_and_hashes(val_run, fixture_models):
    out, _log, _before = val_run
    paths, _sha, ckpt = fixture_models
    s = json.loads((out / "summary.json").read_text())
    assert s["split"] == "val" and s["n_rows"] == 32 and s["smoke"] is True and s["has_t1"] is True
    assert set(s["systems"]) == {"input", "t1", "t2_oracle", "t2_predicted", "t3"}
    assert s["checkpoint"]["t3_checkpoint_sha256"] == sha256_file(ckpt) and len(s["manifest_sha256"]) == 64
    assert set(s["sources"]) == {"classifier", "salt", "blur", "occlusion", "t1"}
    assert s["sources"]["classifier"]["sha256"] == sha256_file(paths["classifier"])
    assert s["tau"] == 2.0 and s["train_params"]["lambda_c"] == 0.1 and s["sources_checked_against_checkpoint"] is True
    gate = s["gate_vs_classifier"]
    # at initialisation the gate IS the classifier: same accuracy, same decisions
    assert gate["t3_gate_accuracy"] == gate["t2_classifier_accuracy"] and gate["gate_agrees_with_t2_classifier"] == 1.0
    assert set(s["collapse"]) >= {"collapsed", "reasons", "min_mean_weight", "max_foreign_weight"}
    assert set(s["expert_activity"]) == set(BRANCH_NAMES)
    assert set(s["thresholds"]) == {"dominant_min_weight", "distributed_max_weight",
                                    "inactive_mean_weight", "unrelated_mean_weight"}


def test_collapse_rule_flags_a_dead_branch_and_a_foreign_takeover():
    rows = [{"true_class": c, "w_identity": 0.0, "w_salt": 0.0, "w_blur": 0.0, "w_occlusion": 0.0}
            for c in (0, 1, 2, 3) for _ in range(4)]
    df = pd.DataFrame(rows)
    df["w_salt"] = 1.0                                         # everything goes to the salt branch
    flags = t3_eval._collapse_flags(df, {})
    assert flags["collapsed"] and any("identity" in r and "mean weight" in r for r in flags["reasons"])
    assert any("> 0.9 on gaussian_blur" in r for r in flags["reasons"])
    balanced = pd.DataFrame([{"true_class": c, **{w: (0.85 if k == c else 0.05) for k, w in enumerate(W)}}
                             for c in (0, 1, 2, 3) for _ in range(4)])
    assert not t3_eval._collapse_flags(balanced, {})["collapsed"]
    assert t3_eval._collapse_flags(balanced, {"collapse": {"max_foreign_weight": 0.04}})["collapsed"]


# ----------------------------------------------------------------------------- Task 1 is optional
def test_t1_column_is_absent_without_a_t1_checkpoint(tiny_pets_root, fixture_models, tmp_path):
    paths, _sha, ckpt = fixture_models
    no_t1 = {**paths, "t1": None}
    out = evaluate_task3(make_cfg(tiny_pets_root, tmp_path, max_rows=8, n_examples=2), ckpt, sources=no_t1)
    df = pd.read_csv(out / "per_image.csv")
    assert "J_t1" not in df.columns and "J_t2_oracle" in df.columns and len(df) == 8
    table = pd.read_csv(out / "comparison_cond_severity.csv")
    assert not any("t1" in c for c in table.columns)
    s = json.loads((out / "summary.json").read_text())
    assert s["has_t1"] is False and "t1" not in s["systems"] and "t1" not in s["sources"]


# ----------------------------------------------------------------------------- sources
def test_sources_that_differ_from_the_recorded_hashes_are_refused(tiny_pets_root, fixture_models, tmp_path):
    _paths, _sha, ckpt = fixture_models
    other, _ = make_t3_sources(tmp_path / "other", seed=5)      # same file names, different weights
    with pytest.raises(ValueError, match="sha256 mismatch"):
        evaluate_task3(make_cfg(tiny_pets_root, tmp_path, max_rows=4), ckpt, sources=other)


def test_run_evaluation_finds_the_sources_from_the_config(tiny_pets_root, fixture_models, tmp_path):
    paths, _sha, ckpt = fixture_models
    cfg = make_cfg(tiny_pets_root, tmp_path, max_rows=8, n_examples=2)
    cfg["sources"] = {"dir": str(paths["classifier"].parent)}
    out = run_evaluation(cfg, str(ckpt))
    assert (out / "summary.json").exists()
    assert json.loads((out / "summary.json").read_text())["has_t1"] is True


def test_run_evaluation_needs_a_checkpoint_and_sources(tiny_pets_root, fixture_models, tmp_path, monkeypatch):
    _paths, _sha, ckpt = fixture_models
    with pytest.raises(ValueError, match="checkpoint"):
        run_evaluation(make_cfg(tiny_pets_root, tmp_path), None)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(t3_sources, "MODELS_CKPT", empty)       # hide the repository's real Task 2 checkpoints
    cfg = make_cfg(empty, tmp_path)                             # no Task 2 files anywhere in the data root
    cfg["sources"] = {"dir": str(empty)}
    with pytest.raises(FileNotFoundError, match="Places searched"):
        evaluate_task3(cfg, ckpt)


# ----------------------------------------------------------------------------- the locked test set
def test_val_run_leaves_the_test_set_alone(val_run):
    _out, log, before = val_run
    assert not log.exists()                                    # no access was logged
    assert real_log_lines() == before                          # and the real log has no new line


def test_final_test_writes_only_to_the_tmp_access_log(tiny_pets_root, fixture_models, tmp_path, monkeypatch):
    paths, _sha, ckpt = fixture_models
    log = tmp_path / "test_access.log"
    monkeypatch.setattr(pets_dataset, "TEST_LOG_PATH", log)
    before = real_log_lines()
    out = evaluate_task3(make_cfg(tiny_pets_root, tmp_path, max_rows=20, n_examples=2), ckpt,
                         final_test=True, sources=paths)       # the synthetic test split of tiny_pets_root
    assert out.name.endswith("_test") and len(log.read_text().splitlines()) == 1
    assert len(pd.read_csv(out / "per_image.csv")) == 20
    assert json.loads((out / "summary.json").read_text())["split"] == "test"
    assert real_log_lines() == before


def test_a_checkpoint_copy_elsewhere_evaluates_the_same(tiny_pets_root, fixture_models, tmp_path):
    """The evaluation reads only the checkpoint and the sources: moving the .pt file changes nothing."""
    paths, _sha, ckpt = fixture_models
    moved = tmp_path / "elsewhere" / "best.pt"
    moved.parent.mkdir()
    shutil.copyfile(ckpt, moved)
    a = evaluate_task3(make_cfg(tiny_pets_root, tmp_path / "a", max_rows=8, n_examples=2), ckpt, sources=paths)
    b = evaluate_task3(make_cfg(tiny_pets_root, tmp_path / "b", max_rows=8, n_examples=2), moved, sources=paths)
    assert pd.read_csv(a / "per_image.csv").equals(pd.read_csv(b / "per_image.csv"))


# ----------------------------------------------------------------------------- report assets (tools/t3_report_assets.py)
def make_fake_run_and_study(tmp_path):
    """A final run's metrics.jsonl (epoch 0 + 1 warm-up + 2 joint epochs), a study folder and the two configs."""
    run = tmp_path / "run"
    run.mkdir()
    rows = [{"epoch": 0, "stage": "joint", "val_J": 0.30, "val_mean_w": [0.25, 0.25, 0.25, 0.25], "val_gate_accuracy": 0.9}]
    for epoch, stage in ((1, "warmup"), (2, "joint"), (3, "joint")):
        rows.append({"epoch": epoch, "stage": stage, "train_loss": 0.2 / epoch, "train_recon": 0.1 / epoch,
                     "train_ce": 0.05 / epoch, "train_balance": 1e-4 / epoch, "val_J": 0.29 - 0.01 * epoch,
                     "val_mean_w": [0.3, 0.2, 0.2, 0.3], "val_gate_accuracy": 0.95})
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    study = tmp_path / "studies" / "t3_moe"
    study.mkdir(parents=True)
    pd.DataFrame({"number": [0, 1, 2, 3], "value": [0.20, 0.18, None, 0.17],
                  "state": ["COMPLETE", "COMPLETE", "PRUNED", "COMPLETE"],
                  "params_joint_lr": [5e-5, 1e-4, 3e-5, 1.1e-5], "params_tau": [1.0, 2.0, 4.0, 3.0],
                  "params_lambda_c": [0.1, 0.2, 0.3, 0.4], "params_lambda_b": [0.01, 0.02, 0.03, 0.04],
                  "params_recon_l1_share": [0.8, 0.7, 0.6, 0.5],
                  "user_attrs_pruned_reason": [None, None, "collapse", None]}).to_csv(study / "trials.csv", index=False)
    (study / "optimization_history.png").write_bytes(b"png")
    (study / "study_run.json").write_text(json.dumps({"timeout_hit": True}), encoding="utf-8")
    space = [{"name": "joint_lr", "type": "float", "low": 1e-5, "high": 2e-4, "log": True},
             {"name": "tau", "type": "float", "low": 0.5, "high": 5.0, "log": True}]
    (tmp_path / "study.yaml").write_text(json.dumps({"tuned_params": space}), encoding="utf-8")        # json is valid yaml
    (tmp_path / "final.yaml").write_text(json.dumps(
        {"model": {"tau": 3.0}, "train": {"joint_lr": 1.1e-5, "lambda_1": 0.5, "lambda_s": 0.5, "lambda_c": 0.4,
                                          "lambda_b": 0.04, "warmup_epochs": 2, "joint_epochs": 12},
         "collapse": {"min_mean_weight": 0.02}, "final_config_source": {"study": "t3_moe", "best_trial": 3}}),
        encoding="utf-8")
    return run, study


def test_report_assets_make_every_table_and_figure(val_run, fixture_models, tmp_path):
    out, _log, _before = val_run
    _paths, _sha, ckpt = fixture_models
    run, study = make_fake_run_and_study(tmp_path)
    tables, figures = tmp_path / "tables", tmp_path / "figures"
    assets = t3_report_assets.build_assets(out, tables, figures, run_dir=run, study_dir=study,
                                           config_path=tmp_path / "study.yaml", final_config=tmp_path / "final.yaml",
                                           checkpoint=ckpt)
    for name in ("comparison_cond_severity", "weights_by_class_severity", "expert_activity", "expert_drift",
                 "gate_vs_classifier", "model_facts", "study_summary", "final_config"):
        assert (tables / f"{name}.csv").exists() and (tables / f"{name}.tex").exists(), name
    for name in ("t3_architecture.png", "routing_heatmap.png", "weights_distribution.png", "examples_dominant.png",
                 "examples_distributed.png", "training_curves.png", "j_comparison.png", "t3_optuna_history.png"):
        assert (figures / name).exists() and (figures / name).stat().st_size > 0, name
    assert any("t3_optuna_importance" in s for s in assets.skipped)           # the fake study has no such plot
    assert r"\begin{tabular}" in (tables / "comparison_cond_severity.tex").read_text()

    study_table = dict(pd.read_csv(tables / "study_summary.csv").to_numpy())
    assert tuple(int(study_table[k]) for k in ("complete", "pruned", "failed")) == (3, 1, 0)
    assert json.loads(study_table["pruned reasons"]) == {"collapse": 1} and int(study_table["best trial"]) == 3
    assert json.loads(study_table["range-edge flags (best value within 10 % of a range end)"]) == {"joint_lr": "low"}
    assert study_table["trial 0 (PDF start values) state"] == "COMPLETE"
    assert str(study_table["timeout hit (stopped before the trial budget)"]) == "True"      # read from study_run.json
    final = pd.read_csv(tables / "final_config.csv").set_index("parameter")
    assert float(final.loc["tau", "final model"]) == 3.0 and float(final.loc["tau", "trial 0 (PDF start)"]) == 1.0
    assert float(final.loc["lambda_s", "trial 0 (PDF start)"]) == pytest.approx(0.2)
    facts = dict(pd.read_csv(tables / "model_facts.csv").to_numpy())
    assert int(facts["parameters, total"]) > 0 and facts["T3 checkpoint sha256"] == sha256_file(ckpt)


def test_report_assets_skip_what_is_missing(val_run, tmp_path):
    out, _log, _before = val_run
    assets = t3_report_assets.build_assets(out, tmp_path / "t", tmp_path / "f", run_dir=None, study_dir=tmp_path / "none",
                                           config_path=None, final_config=tmp_path / "none.yaml")
    assert (tmp_path / "f" / "routing_heatmap.png").exists() and (tmp_path / "t" / "comparison_cond_severity.csv").exists()
    assert not (tmp_path / "f" / "training_curves.png").exists() and len(assets.skipped) >= 3


def test_report_assets_refuse_a_test_folder_without_allow_test(val_run, tmp_path):
    out, _log, _before = val_run
    as_test = tmp_path / "20260101-000000_test"
    shutil.copytree(out, as_test)
    with pytest.raises(AssertionError, match="allow-test"):
        t3_report_assets.build_assets(as_test, tmp_path / "t", tmp_path / "f")
    assert not (tmp_path / "t").exists()
    t3_report_assets.build_assets(as_test, tmp_path / "t", tmp_path / "f", allow_test=True)
    assert (tmp_path / "t" / "test" / "comparison_cond_severity.csv").exists()      # test assets stay apart
    assert (tmp_path / "f" / "test" / "j_comparison.png").exists() and not (tmp_path / "t" / "study_summary.csv").exists()
    with pytest.raises(AssertionError):                                              # neither a val nor a test folder
        t3_report_assets.build_assets(tmp_path, tmp_path / "t2", tmp_path / "f2", allow_test=True)


def test_range_edge_flags():
    space = [{"name": "lr", "type": "float", "low": 1e-5, "high": 1e-3, "log": True},
             {"name": "share", "type": "float", "low": 0.3, "high": 0.95, "log": False},
             {"name": "batch", "type": "categorical", "choices": [32, 64]}]
    assert t3_report_assets.range_edge_flags(space, {"lr": 1.2e-5, "share": 0.94, "batch": 32}) == {"lr": "low", "share": "high"}
    assert t3_report_assets.range_edge_flags(space, {"lr": 1e-4, "share": 0.6, "batch": 64}) == {}
