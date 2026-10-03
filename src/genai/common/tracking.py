"""Thin experiment-tracking wrapper. Implements CONTRACTS 3.12.

Backend chosen by env var TRACKER = wandb | mlflow | none (default none).
'none' needs no account and no network: every call is a no-op.
Only this file knows about wandb / mlflow, so switching touches one place.
Optional env vars: WANDB_MODE=offline, MLFLOW_TRACKING_URI.
"""
from __future__ import annotations

import os

_backend = "none"   # backend of the active run


def _flatten(d: dict, prefix: str = "") -> dict:
    """Nested config dict -> flat {'a.b': value} (MLflow params must be flat)."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _to_hwc_uint8(grid):
    """Tensor (C,H,W) or (N,C,H,W) in [0,1] -> numpy HxWxC uint8 (N images side by side)."""
    import torch
    g = grid.detach().cpu().float().clamp(0, 1)
    if g.dim() == 4:                       # concatenate the batch along width
        g = torch.cat(list(g), dim=2)
    if g.size(0) == 1:                     # grayscale -> 3 channels for viewers
        g = g.repeat(3, 1, 1)
    return (g.permute(1, 2, 0).numpy() * 255).round().astype("uint8")


def init_run(cfg: dict, run_id: str, project: str = "genai-a1", group: str | None = None):
    """Start a run. cfg is logged as the run config; run_id is the run name."""
    global _backend
    _backend = os.environ.get("TRACKER", "none").lower()
    if _backend == "wandb":
        import wandb
        wandb.init(project=project, group=group, name=run_id, config=cfg)
    elif _backend == "mlflow":
        import mlflow
        # MLflow's plain file store is deprecated, so the default is a local
        # SQLite file (no server, no account). Override with MLFLOW_TRACKING_URI.
        if "MLFLOW_TRACKING_URI" not in os.environ:
            from genai.common import paths
            paths.ARTIFACTS.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri("sqlite:///" + (paths.ARTIFACTS / "mlflow.db").as_posix())
        mlflow.set_experiment(group or project)
        mlflow.start_run(run_name=run_id)
        mlflow.log_params({k: str(v)[:250] for k, v in _flatten(cfg).items()})
    else:
        _backend = "none"


def log(metrics: dict, step: int) -> None:
    """Log scalar metrics at a step."""
    if _backend == "wandb":
        import wandb
        wandb.log(metrics, step=step)
    elif _backend == "mlflow":
        import mlflow
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)


def log_images(name: str, tensor_grid, step: int) -> None:
    """Log an image grid tensor (C,H,W) or (N,C,H,W), values in [0,1]."""
    if _backend == "none":
        return
    img = _to_hwc_uint8(tensor_grid)
    if _backend == "wandb":
        import wandb
        wandb.log({name: wandb.Image(img)}, step=step)
    elif _backend == "mlflow":
        import mlflow
        mlflow.log_image(img, key=name, step=step)


def finish() -> None:
    """End the active run (safe to call when none is active)."""
    global _backend
    if _backend == "wandb":
        import wandb
        wandb.finish()
    elif _backend == "mlflow":
        import mlflow
        mlflow.end_run()
    _backend = "none"
