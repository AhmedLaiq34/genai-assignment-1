"""Tests for genai.common.tracking and timing."""
import pytest
import torch

from genai.common import timing, tracking


def test_tracker_none_is_noop(monkeypatch):
    monkeypatch.delenv("TRACKER", raising=False)
    tracking.init_run({"a": 1}, "run1")
    tracking.log({"loss": 1.0}, step=0)
    tracking.log_images("g", torch.rand(3, 8, 8), step=0)
    tracking.finish()
    tracking.finish()  # safe to call twice


def test_tracker_mlflow_local_store(monkeypatch, tmp_path):
    pytest.importorskip("mlflow")
    monkeypatch.setenv("TRACKER", "mlflow")
    monkeypatch.chdir(tmp_path)  # image artifacts default to ./mlartifacts
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///" + (tmp_path / "mlflow.db").as_posix())
    tracking.init_run({"lr": 1e-3, "nested": {"k": 2}}, "run1", group="test")
    tracking.log({"loss": 0.5}, step=1)
    tracking.log_images("grid", torch.rand(2, 3, 8, 8), step=1)
    tracking.finish()
    assert (tmp_path / "mlflow.db").exists()


def test_timer_and_time_block():
    with timing.Timer() as t:
        sum(range(1000))
    assert t.seconds >= 0 and t.ms == t.seconds * 1000
    out, ms = timing.time_block(lambda a, b: a + b, 1, 2)
    assert out == 3 and ms >= 0
