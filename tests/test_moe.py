"""Tests for the soft mixture of experts (models/moe.py) and the source finder (tasks/task3/sources.py).

All models are tiny and randomly initialised (tests/t3_fixtures.py): the numbers mean nothing, the tests
prove the formulas, the initialisation from the Task 2 checkpoints and the provenance checks.
"""
import shutil

import pytest
import torch

from genai.common.checkpoint import sha256_file
from genai.models.moe import SoftMoE, describe, parameter_hash
from genai.tasks.task3 import BRANCH_NAMES, EXPERT_NAMES, SOURCE_FILES
from genai.tasks.task3.sources import find_sources, source_record, verify_sources
from genai.tasks.task2.routing import load_component
from t3_fixtures import make_t3_sources


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    """Fixture checkpoints written once for the whole module (tests that damage files use their own copy)."""
    return make_t3_sources(tmp_path_factory.mktemp("t3_sources"))


@pytest.fixture()
def model(sources):
    """A fresh SoftMoE (tau 1) built from the fixture sources, in eval mode."""
    paths, sha = sources
    return SoftMoE.load_from_task2(paths, tau=1.0, expected_sha256=sha).eval()


def images(n=4, seed=0):
    return torch.rand(n, 3, 128, 128, generator=torch.Generator().manual_seed(seed))


# ======================================================================= the mixture formula
@pytest.mark.parametrize("tau", [0.5, 1.0, 5.0])
def test_weights_sum_to_one(sources, tau):
    paths, sha = sources
    m = SoftMoE.load_from_task2(paths, tau=tau, expected_sha256=sha).eval()
    with torch.no_grad():
        x_hat, w, logits = m(images(6))
    assert w.shape == (6, 4) and logits.shape == (6, 4) and x_hat.shape == (6, 3, 128, 128)
    assert w.dtype == torch.float32 and logits.dtype == torch.float32
    assert torch.allclose(w.sum(dim=1), torch.ones(6), atol=1e-6)
    assert (w >= 0).all()
    assert torch.allclose(w, torch.softmax(logits / tau, dim=1), atol=1e-7)


def test_x_hat_matches_manual_loop(model):
    x = images(3, seed=1)
    with torch.no_grad():
        x_hat, w, logits, branches = model(x, return_branches=True)
        # The same thing written out branch by branch: w0 x + w1 A_salt(x) + w2 A_blur(x) + w3 A_occ(x).
        manual = w[:, 0, None, None, None] * x
        for k, name in enumerate(EXPERT_NAMES, start=1):
            manual = manual + w[:, k, None, None, None] * model.experts[name](x)
    assert branches.shape == (3, 4, 3, 128, 128)
    assert torch.equal(branches[:, 0], x)                     # branch 0 is the (corrupted) input itself
    for k, name in enumerate(EXPERT_NAMES, start=1):          # branches 1..3 are the experts' outputs
        assert torch.allclose(branches[:, k], model.experts[name](x), atol=1e-6)
    assert torch.allclose(x_hat, manual, atol=1e-6)


def test_output_in_unit_range(model):
    with torch.no_grad():
        x_hat, _, _ = model(images(5, seed=2))
    assert x_hat.min() >= 0.0 and x_hat.max() <= 1.0


def test_tau_changes_sharpness_not_logits(sources):
    paths, sha = sources
    sharp = SoftMoE.load_from_task2(paths, tau=0.5, expected_sha256=sha).eval()
    soft = SoftMoE.load_from_task2(paths, tau=5.0, expected_sha256=sha).eval()
    x = images(4, seed=3)
    with torch.no_grad():
        _, w_sharp, l_sharp = sharp(x)
        _, w_soft, l_soft = soft(x)
    assert torch.equal(l_sharp, l_soft)                        # the gate is the same
    assert w_sharp.max(dim=1).values.mean() > w_soft.max(dim=1).values.mean()


# ======================================================================= initialisation from Task 2
def test_gate_equals_classifier(sources, model):
    paths, _ = sources
    classifier, _ = load_component("classifier", paths["classifier"])
    x = images(5, seed=4)
    with torch.no_grad():
        _, _, logits = model(x)
        expected = classifier(x)
    assert torch.allclose(logits, expected, atol=1e-6)
    assert parameter_hash(model.gate) == parameter_hash(classifier)   # weights and BatchNorm buffers


def test_experts_equal_specialists(sources, model):
    paths, _ = sources
    x = images(3, seed=5)
    for name in EXPERT_NAMES:
        specialist, _ = load_component(name, paths[name])
        assert parameter_hash(model.experts[name]) == parameter_hash(specialist)
        with torch.no_grad():
            assert torch.allclose(model.experts[name](x), specialist(x), atol=1e-6)


def test_load_does_not_modify_source_files(sources):
    paths, sha = sources
    before = {n: sha256_file(paths[n]) for n in paths}
    SoftMoE.load_from_task2(paths, tau=1.0, expected_sha256=sha)
    assert {n: sha256_file(paths[n]) for n in paths} == before


def test_wrong_expert_checkpoint_refused(sources):
    """A salt checkpoint handed in as "blur" is refused by the component checks of task2.routing."""
    paths, _ = sources
    swapped = dict(paths, blur=paths["salt"])
    with pytest.raises(ValueError, match="blur"):
        SoftMoE.load_from_task2(swapped, tau=1.0, expected_sha256=None)


# ======================================================================= modes: train, eval, freezing
def test_train_keeps_experts_in_eval(model):
    model.train()
    assert model.training and model.gate.training
    assert not model.experts.training
    assert all(not m.training for m in model.experts.modules())
    model.eval()
    assert not model.gate.training and not model.experts.training
    model.train(True)
    assert model.gate.training and all(not m.training for m in model.experts.modules())


def test_fresh_model_has_experts_in_eval(sources):
    paths, sha = sources
    m = SoftMoE.load_from_task2(paths, tau=1.0, expected_sha256=sha)     # no .eval() call
    assert m.gate.training
    assert all(not mod.training for mod in m.experts.modules())


def test_expert_buffers_unchanged_by_forward_in_train_mode(model):
    """B4: a forward pass in train mode must not move the experts' BatchNorm statistics."""
    model.train()
    before = parameter_hash(model.experts)
    for _ in range(2):
        model(images(4, seed=6))
    assert parameter_hash(model.experts) == before


def test_freeze_experts_gives_no_grad(model):
    model.train()
    assert not model.experts_frozen
    model.freeze_experts()
    assert model.experts_frozen
    assert all(not p.requires_grad for p in model.experts.parameters())
    x_hat, w, logits = model(images(4, seed=7))
    (x_hat.mean() + logits.mean()).backward()
    assert all(p.grad is None for p in model.experts.parameters())      # experts: no gradient at all
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.gate.parameters())
    # and the other way round
    model.unfreeze_experts()
    assert not model.experts_frozen
    model.zero_grad()
    x_hat, _, _ = model(images(4, seed=7))
    x_hat.mean().backward()
    assert all(p.grad is not None for p in model.experts.parameters())


def test_frozen_experts_outputs_carry_no_graph(model):
    model.freeze_experts()
    model.train()
    x_hat, w, _, branches = model(images(2, seed=8), return_branches=True)
    assert not branches.requires_grad          # the experts ran under no_grad
    assert w.requires_grad and x_hat.requires_grad    # the gate still gets its gradient through w


# ======================================================================= rebuilding
def test_model_config_layout(model):
    cfg = model.model_config()
    assert set(cfg) == {"gate", "experts", "tau", "branches"}
    assert cfg["branches"] == list(BRANCH_NAMES)
    assert set(cfg["experts"]) == set(EXPERT_NAMES)
    assert cfg["tau"] == 1.0 and cfg["gate"]["num_classes"] == 4
    cfg["gate"]["channels"].append(999)                                    # a copy: the model is not affected
    assert 999 not in model.model_config()["gate"]["channels"]


def test_from_config_state_dict_round_trip(model):
    model.train()
    for _ in range(2):
        model(images(4, seed=9))                                           # move the gate's BN statistics
    rebuilt = SoftMoE.from_config(model.model_config())
    assert parameter_hash(rebuilt) != parameter_hash(model)                # random init differs
    rebuilt.load_state_dict(model.state_dict())
    assert parameter_hash(rebuilt) == parameter_hash(model)
    model.eval(), rebuilt.eval()
    x = images(3, seed=10)
    with torch.no_grad():
        for a, b in zip(model(x), rebuilt(x)):
            assert torch.equal(a, b)


def test_from_config_rejects_bad_input(model):
    cfg = model.model_config()
    with pytest.raises(ValueError):
        SoftMoE.from_config({**cfg, "experts": {"salt": cfg["experts"]["salt"]}})
    with pytest.raises(ValueError):
        SoftMoE.from_config({**cfg, "tau": 0.0})


def test_parameter_hash_and_describe(model):
    h = parameter_hash(model)
    assert len(h) == 64 and h == parameter_hash(model)
    with torch.no_grad():
        next(model.gate.parameters()).add_(1.0)
    assert parameter_hash(model) != h
    text = describe(model)
    for word in ("gate", *EXPERT_NAMES, "total", "tau=1.0"):
        assert word in text


# ======================================================================= provenance: sha256 checks
def test_verify_sources_ok(sources):
    paths, sha = sources
    result = verify_sources(paths, expected=sha)
    assert {n: result[n] for n in sha} == sha
    assert result["t1"] == sha256_file(paths["t1"])                        # hashed too, nothing to compare with
    assert verify_sources(paths, expected=sha, t1_sha=result["t1"])["t1"] == result["t1"]


def test_wrong_sha_refused(sources, tmp_path):
    paths, sha = sources
    wrong = dict(sha, blur="0" * 64)
    with pytest.raises(ValueError) as err:
        verify_sources(paths, expected=wrong)
    message = str(err.value)
    assert SOURCE_FILES["blur"] in message and "0" * 64 in message and sha["blur"] in message
    # load_from_task2 checks before loading anything
    with pytest.raises(ValueError, match=SOURCE_FILES["blur"]):
        SoftMoE.load_from_task2(paths, tau=1.0, expected_sha256=wrong)
    # a file that was changed on disk (a copy, the shared fixture stays intact) is refused as well
    copy_paths = {n: shutil.copy(p, tmp_path / p.name) for n, p in paths.items()}
    with open(copy_paths["salt"], "ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError, match=SOURCE_FILES["salt"]):
        SoftMoE.load_from_task2(copy_paths, tau=1.0, expected_sha256=sha)
    # the real hashes of the Task 2 checkpoints do not match random fixture files
    with pytest.raises(ValueError):
        SoftMoE.load_from_task2(paths, tau=1.0)


def test_wrong_t1_sha_refused(sources):
    paths, sha = sources
    with pytest.raises(ValueError, match=SOURCE_FILES["t1"]):
        verify_sources(paths, expected=sha, t1_sha="f" * 64)


def test_missing_file_refused(sources, tmp_path):
    paths, sha = sources
    gone = dict(paths, occlusion=tmp_path / "does_not_exist.pt")
    with pytest.raises(ValueError, match=SOURCE_FILES["occlusion"]):
        verify_sources(gone, expected=sha)
    with pytest.raises(ValueError, match=SOURCE_FILES["occlusion"]):
        SoftMoE.load_from_task2(gone, tau=1.0, expected_sha256=sha)
    with pytest.raises(ValueError, match=SOURCE_FILES["salt"]):
        verify_sources({k: v for k, v in paths.items() if k != "salt"}, expected=sha)
    # an absent Task 1 file is fine unless its hash is demanded
    no_t1 = dict(paths, t1=None)
    assert "t1" not in verify_sources(no_t1, expected=sha)
    with pytest.raises(ValueError, match=SOURCE_FILES["t1"]):
        verify_sources(no_t1, expected=sha, t1_sha="a" * 64)


def test_source_record(sources):
    paths, _ = sources
    record = source_record(dict(paths, t1=None))
    assert set(record) == {"classifier", "salt", "blur", "occlusion"}
    assert record["salt"] == {"file": SOURCE_FILES["salt"], "sha256": sha256_file(paths["salt"])}
    assert "t1" in source_record(paths)


# ======================================================================= find_sources
def test_find_sources_in_nested_input_folder(no_repo_models, tmp_path, sources):
    """Kaggle layout: <data_root>/datasets/<user>/<dataset>/<files>."""
    paths, _ = sources
    nested = tmp_path / "input" / "datasets" / "u" / "t3-sources"
    nested.mkdir(parents=True)
    for p in paths.values():
        shutil.copy(p, nested / p.name)
    found = find_sources({"data_root": str(tmp_path / "input")})
    assert set(found) == set(SOURCE_FILES)
    for name in SOURCE_FILES:
        assert found[name] == nested / SOURCE_FILES[name]


def test_find_sources_explicit_dir_wins(no_repo_models, tmp_path, sources):
    paths, _ = sources
    explicit = tmp_path / "explicit"
    other = tmp_path / "root" / "t3"
    for folder in (explicit, other):
        folder.mkdir(parents=True)
        for p in paths.values():
            shutil.copy(p, folder / p.name)
    found = find_sources({"sources": {"dir": str(explicit)}, "data_root": str(tmp_path / "root")})
    assert all(p.parent == explicit for p in found.values())


@pytest.fixture()
def no_repo_models(tmp_path, monkeypatch):
    """Point the repository fallback (models/checkpoints) at an empty folder: the real files must not be found."""
    import genai.tasks.task3.sources as src
    monkeypatch.setattr(src, "MODELS_CKPT", tmp_path / "empty_models_folder")


def test_find_sources_t1_optional(no_repo_models, tmp_path, sources):
    paths, _ = sources
    for name in ("classifier", "salt", "blur", "occlusion"):
        shutil.copy(paths[name], tmp_path / paths[name].name)
    cfg = {"sources": {"dir": str(tmp_path)}}
    assert find_sources(cfg)["t1"] is None
    with pytest.raises(FileNotFoundError, match=SOURCE_FILES["t1"]):
        find_sources(cfg, need_t1=True)


def test_find_sources_error_lists_every_place(no_repo_models, tmp_path):
    empty = tmp_path / "empty_root"
    empty.mkdir()
    cfg = {"sources": {"dir": str(tmp_path / "nowhere")}, "data_root": str(empty)}
    with pytest.raises(FileNotFoundError) as err:
        find_sources(cfg)
    message = str(err.value)
    assert str(tmp_path / "nowhere") in message and str(empty) in message
    assert "models" in message and SOURCE_FILES["classifier"] in message


def test_find_sources_has_no_hidden_data_root(no_repo_models):
    """Without cfg['data_root'] no data folder is searched; the message says so (decision D46)."""
    with pytest.raises(FileNotFoundError, match="data_root"):
        find_sources({})


def test_find_sources_falls_back_to_models_folder(tmp_path, sources, monkeypatch):
    paths, _ = sources
    import genai.tasks.task3.sources as src
    for p in paths.values():
        shutil.copy(p, tmp_path / p.name)
    monkeypatch.setattr(src, "MODELS_CKPT", tmp_path)
    found = find_sources({"data_root": str(tmp_path / "empty_input")})   # does not exist: reported, then models/
    assert found["classifier"] == tmp_path / SOURCE_FILES["classifier"]
