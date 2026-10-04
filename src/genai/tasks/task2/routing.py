"""Hard routing for Task 2 (PDF page 5): classifier -> argmax -> one specialist (or identity).

Two parts:

* load_task2_models(paths)  rebuilds the four trained networks from their checkpoints and checks that
  every checkpoint really is what its name says (a salt checkpoint handed in as "blur" is an error).
* HardRoutedSystem          the routing itself.

    x_hat = x                 if the route is 0 (clean: IDENTITY BYPASS, no specialist is called)
            A_salt(x)         if the route is 1
            A_blur(x)         if the route is 2
            A_occlusion(x)    if the route is 3

Where does the route come from?
  oracle mode     the TRUE class id (the manifest's cond_id). Shows what the specialists can do.
  predicted mode  argmax of the classifier's softmax. Shows the whole operational system.

Inside a batch every image is routed on its own: the images are grouped by route, every specialist
runs ONCE on its group, and the results are written back to the original positions.
"""
from __future__ import annotations

import torch

from genai.common.checkpoint import load_checkpoint, sha256_file
from genai.models.autoencoder import UniversalAE
from genai.models.classifier import CorruptionClassifier
from genai.tasks.task2 import EXPERT_FOR_CLASS, SPECIALIST_COND_ID

COMPONENT_NAMES = ("classifier", "salt", "blur", "occlusion")   # keys of the checkpoint / model dicts


# ======================================================================= loading + checks
def check_checkpoint(name: str, ckpt: dict) -> None:
    """Raise a clear ValueError if checkpoint `ckpt` is not the one that belongs to `name`.

    name is "classifier", "salt", "blur" or "occlusion". The checks use what training stored in
    ckpt["config"]: the component name and, for a specialist, its corruption and cond_id.
    """
    if name not in COMPONENT_NAMES:
        raise ValueError(f"unknown component {name!r}; expected one of {COMPONENT_NAMES}")
    cfg = ckpt.get("config") or {}
    component = cfg.get("component")

    if name == "classifier":
        if component != "classifier":
            raise ValueError(f"the 'classifier' checkpoint has component {component!r}, expected 'classifier'")
        return

    if component != "specialist":
        raise ValueError(f"the {name!r} checkpoint has component {component!r}, expected 'specialist'")
    spec = cfg.get("specialist") or {}
    expected_id = SPECIALIST_COND_ID[name]
    if spec.get("cond_id") != expected_id:
        raise ValueError(f"the {name!r} checkpoint was trained for cond_id {spec.get('cond_id')!r} "
                         f"(corruption {spec.get('corruption')!r}) but {name!r} needs cond_id {expected_id}; "
                         f"was a checkpoint passed under the wrong name?")
    if spec.get("corruption") != name:
        raise ValueError(f"the {name!r} checkpoint was trained for corruption {spec.get('corruption')!r}, "
                         f"expected {name!r}")


def load_component(name: str, ckpt_path) -> tuple:
    """Rebuild ONE network from its checkpoint. Returns (model in eval mode, checkpoint dict).

    The classifier is a CorruptionClassifier, every specialist is a UniversalAE; both are rebuilt from
    the `model` section stored in the checkpoint config, then the weights are loaded.
    """
    ckpt = load_checkpoint(ckpt_path)
    check_checkpoint(name, ckpt)
    model_cls = CorruptionClassifier if name == "classifier" else UniversalAE
    model = model_cls.from_config(ckpt["config"]["model"])
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt            # eval(): dropout off, BatchNorm uses running statistics


def load_task2_models(paths: dict, device="cpu") -> dict:
    """paths = {"classifier": ..., "salt": ..., "blur": ..., "occlusion": ...} -> the four models.

    Every model is in eval mode on `device`. Missing keys and mismatching checkpoints raise ValueError.
    """
    missing = [name for name in COMPONENT_NAMES if not paths.get(name)]
    if missing:
        raise ValueError(f"load_task2_models needs four checkpoint paths; missing: {missing}")
    return {name: load_component(name, paths[name])[0].to(device) for name in COMPONENT_NAMES}


def describe_checkpoints(paths: dict) -> dict:
    """Facts about the four checkpoints for summary.json: path, sha256, run id, step, smoke flag."""
    info = {}
    for name in COMPONENT_NAMES:
        ckpt = load_checkpoint(paths[name])
        cfg = ckpt.get("config") or {}
        run_id = cfg.get("run_id", "")
        info[name] = {
            "path": str(paths[name]),
            "sha256": sha256_file(paths[name]),
            "run_id": run_id,
            "global_step": ckpt.get("global_step"),
            "smoke": bool((cfg.get("run") or {}).get("smoke")) or run_id.endswith("_smoke"),
        }
    return info


# ======================================================================= the routed system
class HardRoutedSystem:
    """Classifier + three specialists with hard routing (see the module docstring).

    models: the dict returned by load_task2_models (anything with the same keys also works, which
            is how the tests plug in counting stand-ins).
    """

    def __init__(self, models: dict, device="cpu"):
        self.models = models
        self.device = torch.device(device)

    @torch.no_grad()
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Images (N,3,128,128) -> class probabilities (N,4): softmax of the classifier's logits."""
        logits = self.models["classifier"](x.to(self.device))
        return torch.softmax(logits, dim=1)

    @torch.no_grad()
    def restore_by_route(self, x: torch.Tensor, route_ids: torch.Tensor) -> torch.Tensor:
        """Restore every image with the expert of its own route id (0 clean, 1 salt, 2 blur, 3 occlusion).

        Route 0 is the identity bypass: those images are copied through and NO specialist sees them.
        Each of the other routes calls its specialist once, on just the images that were sent to it.
        """
        x = x.to(self.device)
        route_ids = route_ids.to(self.device)
        out = x.clone()                                    # identity for everything not restored below
        for class_id in (1, 2, 3):                         # class 0 is deliberately absent: bypass
            rows = (route_ids == class_id).nonzero(as_tuple=True)[0]
            if len(rows) == 0:
                continue                                   # nobody was routed here: skip the specialist
            specialist = self.models[EXPERT_FOR_CLASS[class_id]]
            out[rows] = specialist(x[rows])                # run once for the whole group, scatter back
        return out

    @torch.no_grad()
    def route(self, x: torch.Tensor, true_cond=None, mode: str = "predicted") -> tuple:
        """Restore a batch. Returns (restored, route_ids, probs).

        mode="oracle":    route_ids = true_cond (needed: the known class id of every image).
        mode="predicted": route_ids = argmax of the classifier probabilities.
        probs are always the classifier's softmax output (N,4), so oracle mode can still be compared
        with what the classifier would have said.
        """
        probs = self.classify(x)
        if mode == "oracle":
            if true_cond is None:
                raise ValueError("oracle routing needs true_cond (the manifest's cond_id of every image)")
            route_ids = torch.as_tensor(true_cond, dtype=torch.long)
        elif mode == "predicted":
            route_ids = probs.argmax(dim=1)
        else:
            raise ValueError(f"mode must be 'oracle' or 'predicted', got {mode!r}")
        return self.restore_by_route(x, route_ids), route_ids.to(self.device), probs
