"""Corruption classifier for Task 2: image -> 4 logits (clean, salt_pepper, gaussian_blur, occlusion).

Implements CONTRACTS 3.1 (input float32 N x 3 x 128 x 128 in [0,1]; any normalisation happens
INSIDE forward, so the ONNX input is a plain [0,1] image) and 3.9 (ONNX output = logits N x 4;
the softmax lives in evaluation and in the backend).

Data flow (example: channels=[16, 32, 64, 128], 128x128 input)

    input   3 x 128 x 128   in [0,1]
    x*2-1   same shape, in [-1,1]                          (fixed normalisation, no parameters)
    block 1 16 x 64 x 64    (3x3 conv, stride 2, BatchNorm, ReLU)
    block 2 32 x 32 x 32
    block 3 64 x 16 x 16
    block 4 128 x 8 x 8
    global average pooling  -> 128 numbers
    dropout -> linear       -> 4 logits

One block per entry of `channels`; every block halves the height and width.
"""
from __future__ import annotations

import torch
from torch import nn

from genai.common.constants import NUM_CLASSES


def _block(c_in: int, c_out: int) -> nn.Sequential:
    """Halve the spatial size: 3x3 conv, stride 2, padding 1 (H -> H/2), then BatchNorm + ReLU."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
    )


class CorruptionClassifier(nn.Module):
    """Small CNN that predicts which corruption an image has.

    Args:
        channels:     output channels of each conv block, e.g. [16, 32, 64, 128]
                      (the number of entries is the number of stride-2 blocks).
        dropout:      dropout probability before the final linear layer (0 = off).
        num_classes:  4 (clean, salt_pepper, gaussian_blur, occlusion).
    """

    def __init__(self, channels: list, dropout: float = 0.0, num_classes: int = NUM_CLASSES):
        super().__init__()
        channels = [int(c) for c in channels]
        if not channels or min(channels) < 1:
            raise ValueError(f"channels must be a non-empty list of positive ints, got {channels}")
        # Remember the arguments: they are saved in the checkpoint so the model can be rebuilt.
        self.hparams = dict(channels=channels, dropout=float(dropout), num_classes=int(num_classes))

        blocks, c_prev = [], 3                       # 3 = RGB input channels
        for c in channels:
            blocks.append(_block(c_prev, c))
            c_prev = c
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                 # global average pooling: N x C x H x W -> N x C x 1 x 1
            nn.Flatten(),                            # -> N x C
            nn.Dropout(dropout),
            nn.Linear(c_prev, num_classes),          # -> N x 4 logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: N x 3 x 128 x 128 in [0,1]  ->  logits N x num_classes (no softmax)."""
        x = x * 2.0 - 1.0                            # [0,1] -> [-1,1], done here so callers pass plain images
        return self.head(self.features(x))

    @classmethod
    def from_config(cls, model_cfg: dict) -> "CorruptionClassifier":
        """Build from a config/checkpoint dict (keys: channels, dropout, num_classes); other keys are ignored."""
        keys = ("channels", "dropout", "num_classes")
        return cls(**{k: model_cfg[k] for k in keys if k in model_cfg})


def describe(model: CorruptionClassifier) -> str:
    """Short summary printed at the start of a run (uses count_parameters from the autoencoder module)."""
    from genai.models.autoencoder import count_parameters
    return f"CorruptionClassifier {model.hparams}\n  trainable parameters: {count_parameters(model):,}"


Classifier = CorruptionClassifier   # scaffold name kept as an alias
