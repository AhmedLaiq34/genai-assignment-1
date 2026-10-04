"""Soft mixture of experts for Task 3 (PDF pages 6 and 7): gate + identity branch + three experts.

Implements CONTRACTS 3.2 (branch order), 3.8 (checkpoint config) and 3.9 (ONNX outputs output + weights).
Design: docs/TASK3_PLAN.md, section D1 and decisions B1 to B5.

Data flow (x = corrupted image, N x 3 x 128 x 128 in [0,1])

    gate(x)               4 raw logits per image      (a copy of the Task 2 CorruptionClassifier)
    w = softmax(logits / tau)                          4 weights per image, they sum to 1
    branch 0 = x                                       the identity branch: the input as it is
    branch 1 = A_salt(x)    branch 2 = A_blur(x)    branch 3 = A_occlusion(x)     (the three Task 2 specialists)
    x_hat = w0 * x + w1 * A_salt(x) + w2 * A_blur(x) + w3 * A_occlusion(x)

The weights are positive and sum to 1, so x_hat is a convex combination of four images in [0,1]
and stays in [0,1]: there is no clamp. All four branches always run (soft mixture, not routing).

Two modes that matter for training
    * the gate follows the usual train() / eval() switch (BatchNorm statistics, dropout);
    * the experts are ALWAYS in eval() mode (decision B4): each specialist's BatchNorm statistics were
      learned on its own corruption only, and in the mixture every expert sees all four classes, so
      train-mode BatchNorm would overwrite those statistics. While the experts are frozen (warm-up
      stage) they also run under torch.no_grad().

Public names: SoftMoE, parameter_hash, describe.
"""
from __future__ import annotations

import copy
import hashlib

import torch
from torch import nn

from genai.models.autoencoder import UniversalAE, count_parameters
from genai.models.classifier import CorruptionClassifier
from genai.tasks.task3 import BRANCH_NAMES, EXPERT_NAMES, TASK2_SHA256


class SoftMoE(nn.Module):
    """Gate + identity branch + three experts, mixed with softmax weights (see the module docstring).

    Args:
        gate_cfg:    constructor arguments of the gate = a CorruptionClassifier
                     (channels, dropout, num_classes), the `model` section of the classifier checkpoint.
        expert_cfgs: {"salt": cfg, "blur": cfg, "occlusion": cfg}, each cfg the constructor arguments
                     of a UniversalAE (the `model` section of the specialist checkpoint).
        tau:         softmax temperature, a fixed number for the whole run (decision B3). tau > 1 makes
                     the weights softer, tau < 1 sharper; tau = 1 is the plain softmax.
    """

    def __init__(self, gate_cfg: dict, expert_cfgs: dict, tau: float = 1.0):
        super().__init__()
        if set(expert_cfgs) != set(EXPERT_NAMES):
            raise ValueError(f"expert_cfgs needs exactly the keys {EXPERT_NAMES}, got {tuple(expert_cfgs)}")
        if not float(tau) > 0:
            raise ValueError(f"tau must be positive, got {tau}")

        self.gate = CorruptionClassifier.from_config(gate_cfg)
        # nn.ModuleDict keeps the keys in insertion order; the order of EXPERT_NAMES is the branch order 1, 2, 3.
        self.experts = nn.ModuleDict({name: UniversalAE.from_config(expert_cfgs[name]) for name in EXPERT_NAMES})
        self.tau = float(tau)                       # a plain Python float: it becomes a constant in the ONNX graph
        # Remember what is needed to rebuild the model (the gate and the experts store their own hparams).
        self.hparams = dict(gate=copy.deepcopy(self.gate.hparams),
                            experts={n: copy.deepcopy(self.experts[n].hparams) for n in EXPERT_NAMES},
                            tau=self.tau)
        self.experts.eval()                         # B4: the experts never leave eval mode

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor, return_branches: bool = False):
        """x: N x 3 x 128 x 128 in [0,1].

        Returns (x_hat, w, logits) or, with return_branches=True, (x_hat, w, logits, branches):
            x_hat     N x 3 x 128 x 128  float32, the mixture
            w         N x 4              float32, softmax(logits / tau), order = BRANCH_NAMES
            logits    N x 4              float32, the gate's raw logits (the cross-entropy loss uses these)
            branches  N x 4 x 3 x 128 x 128  float32, branch 0 is x itself
        Everything after the networks is computed in float32, also when the networks run under fp16 autocast.
        """
        logits = self.gate(x).float()                          # N x 4 (cast: autocast would give fp16)
        w = torch.softmax(logits / self.tau, dim=1)            # N x 4, rows sum to 1

        # The three experts. While they are frozen nobody needs their gradients: skip building the graph.
        # Each output is cast to float32 so that the stack below does not mix fp16 and fp32.
        if self.experts_frozen:
            with torch.no_grad():
                outputs = [self.experts[name](x).float() for name in EXPERT_NAMES]
        else:
            outputs = [self.experts[name](x).float() for name in EXPERT_NAMES]

        branches = torch.stack([x.float()] + outputs, dim=1)   # N x 4 x 3 x 128 x 128, branch 0 = identity
        # Weighted sum over the branch axis: w is N x 4, spread it to N x 4 x 1 x 1 x 1.
        x_hat = (w[:, :, None, None, None] * branches).sum(dim=1)
        if return_branches:
            return x_hat, w, logits, branches
        return x_hat, w, logits

    # ------------------------------------------------------------------ freezing and modes
    def freeze_experts(self) -> None:
        """Warm-up stage: the experts get no gradients and are not updated."""
        for p in self.experts.parameters():
            p.requires_grad_(False)

    def unfreeze_experts(self) -> None:
        """Joint stage: the experts' weights are trained together with the gate."""
        for p in self.experts.parameters():
            p.requires_grad_(True)

    @property
    def experts_frozen(self) -> bool:
        """True when no expert parameter requires a gradient."""
        return not any(p.requires_grad for p in self.experts.parameters())

    def train(self, mode: bool = True) -> "SoftMoE":
        """Switch the GATE to train / eval mode; the experts stay in eval mode (decision B4)."""
        self.training = mode
        self.gate.train(mode)
        self.experts.eval()
        return self

    # ------------------------------------------------------------------ rebuilding and loading
    def model_config(self) -> dict:
        """The `model` section stored in the checkpoint config (decision B11); enough for from_config.

            {"gate": {channels, dropout, num_classes},
             "experts": {"salt": {...UniversalAE args}, "blur": {...}, "occlusion": {...}},
             "tau": float,
             "branches": ["identity", "salt", "blur", "occlusion"]}
        """
        return {"gate": copy.deepcopy(self.hparams["gate"]),
                "experts": copy.deepcopy(self.hparams["experts"]),
                "tau": self.tau,
                "branches": list(BRANCH_NAMES)}

    @classmethod
    def from_config(cls, model_cfg: dict) -> "SoftMoE":
        """Build an (untrained, randomly initialised) SoftMoE from a model_config() dict."""
        return cls(model_cfg["gate"], model_cfg["experts"], model_cfg.get("tau", 1.0))

    @classmethod
    def load_from_task2(cls, paths: dict, tau: float, expected_sha256: dict | None = TASK2_SHA256) -> "SoftMoE":
        """Build the model from the four Task 2 checkpoints (the PDF's initialisation).

            paths = {"classifier": ..., "salt": ..., "blur": ..., "occlusion": ...}   (extra keys are ignored)

        1. verify_sources: the four files are hashed and compared with expected_sha256 BEFORE anything is
           loaded. A missing file or a wrong hash raises ValueError. There is no fallback to random weights.
           (expected_sha256=None switches the check off; only for tools that never train or promote.)
        2. task2.routing.load_component rebuilds each network and checks that the checkpoint is what its
           name says (a salt checkpoint passed as "blur" is an error).
        3. The weights are copied into the new SoftMoE: the classifier into the gate, each specialist
           into its expert. The Task 2 files are only read.
        The returned model is a fresh module: the gate is in train mode (call .eval() for inference),
        the experts are in eval mode, nothing is frozen.
        """
        # Imported here, not at the top: task2.routing and sources import this package's names too.
        from genai.tasks.task2.routing import load_component
        from genai.tasks.task3.sources import verify_sources

        if expected_sha256 is not None:
            verify_sources(paths, expected_sha256)

        gate_src, _ = load_component("classifier", paths["classifier"])
        experts_src = {name: load_component(name, paths[name])[0] for name in EXPERT_NAMES}

        model = cls(gate_src.hparams, {name: experts_src[name].hparams for name in EXPERT_NAMES}, tau)
        model.gate.load_state_dict(gate_src.state_dict())
        for name in EXPERT_NAMES:
            model.experts[name].load_state_dict(experts_src[name].state_dict())
        return model


def parameter_hash(module: nn.Module) -> str:
    """sha256 over the whole state_dict (parameters AND buffers such as BatchNorm statistics), in key order.

    Used to prove that something did not change (the experts during warm-up) or did (the gate).
    """
    h = hashlib.sha256()
    for key, tensor in module.state_dict().items():
        h.update(key.encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        # The raw bytes of the tensor (viewed as uint8, which works for every dtype).
        h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def describe(model: SoftMoE) -> str:
    """Parameter counts per part, printed at the start of a run."""
    gate_n = count_parameters(model.gate, trainable_only=False)
    expert_n = {name: count_parameters(model.experts[name], trainable_only=False) for name in EXPERT_NAMES}
    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)
    lines = [f"SoftMoE tau={model.tau} branches={list(BRANCH_NAMES)}",
             f"  gate (CorruptionClassifier {model.gate.hparams}): {gate_n:,} parameters"]
    for name in EXPERT_NAMES:
        lines.append(f"  expert {name}: {expert_n[name]:,} parameters")
    lines.append(f"  total: {total:,} parameters, trainable now: {trainable:,} "
                 f"(experts {'frozen' if model.experts_frozen else 'trainable'})")
    return "\n".join(lines)
