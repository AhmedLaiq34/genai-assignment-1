"""Convolutional autoencoder with a dense bottleneck (Task 1 and the Task 2 specialists).

Implements CONTRACTS 3.1 (input/output float32 N x 3 x 128 x 128 in [0,1], sigmoid output)
and 3.8 (the checkpoint stores the constructor arguments so the model can be rebuilt).

Data flow (example: base_channels=32, depth=4, bottleneck_dim=256, 128x128 input)

    input   3 x 128 x 128
    encoder 32x64x64 -> 64x32x32 -> 128x16x16 -> 256x8x8     (stride-2 convs, channels double)
    flatten 256*8*8 = 16384 numbers
    dense   16384 -> 256                                     <- the genuine bottleneck (latent z)
    dense   256 -> 16384, reshape to 256 x 8 x 8
    decoder 128x16x16 -> 64x32x32 -> 32x64x64 -> 3x128x128   (stride-2 transposed convs)
    sigmoid -> values in [0,1]

There are NO skip connections: everything the decoder knows about the image has to pass
through the `bottleneck_dim` numbers in z. All architecture choices are constructor
arguments, so the same class serves Task 1 and the three Task 2 specialists.
"""
from __future__ import annotations

import torch
from torch import nn

from genai.common.constants import IMG_SIZE


def _down_block(c_in: int, c_out: int) -> nn.Sequential:
    """Halve the spatial size: 4x4 conv, stride 2, padding 1 (H -> H/2), then BatchNorm + ReLU."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


def _up_block(c_in: int, c_out: int) -> nn.Sequential:
    """Double the spatial size: 4x4 transposed conv, stride 2, padding 1 (H -> 2H)."""
    return nn.Sequential(
        nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class UniversalAE(nn.Module):
    """Conv encoder -> flatten -> dense bottleneck -> dense -> conv decoder (no skips).

    Args:
        in_ch:           image channels (3 for RGB).
        base_channels:   channels of the first encoder stage; stage i has base_channels * 2**i.
        depth:           number of stride-2 stages. The spatial size shrinks by 2**depth
                         (128 -> 8 for depth 4).
        bottleneck_dim:  size of the latent vector z (the "compressed representation").
        dropout:         dropout probability applied to the flattened features before the
                         bottleneck and to the expanded features after it (0 = off).
        img_size:        input height/width (128 by contract).
    """

    def __init__(self, in_ch: int = 3, base_channels: int = 32, depth: int = 4,
                 bottleneck_dim: int = 256, dropout: float = 0.0, img_size: int = IMG_SIZE):
        super().__init__()
        if img_size % (2 ** depth) != 0:
            raise ValueError(f"img_size {img_size} must be divisible by 2**depth = {2 ** depth}")
        # Remember the arguments: they are saved in the checkpoint so the model can be rebuilt.
        self.hparams = dict(in_ch=in_ch, base_channels=base_channels, depth=depth,
                            bottleneck_dim=bottleneck_dim, dropout=dropout, img_size=img_size)
        self.in_ch, self.bottleneck_dim, self.img_size = in_ch, bottleneck_dim, img_size

        channels = [base_channels * 2 ** i for i in range(depth)]   # e.g. [32, 64, 128, 256]
        self.final_spatial = img_size // 2 ** depth                  # e.g. 8
        self.final_channels = channels[-1]
        flat_dim = self.final_channels * self.final_spatial ** 2     # e.g. 16384

        # ---- encoder: stride-2 convs, channels grow ----
        enc, c_prev = [], in_ch
        for c in channels:
            enc.append(_down_block(c_prev, c))
            c_prev = c
        self.encoder = nn.Sequential(*enc)

        # ---- dense bottleneck ----
        self.to_latent = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(flat_dim, bottleneck_dim))
        self.from_latent = nn.Sequential(nn.Linear(bottleneck_dim, flat_dim), nn.ReLU(inplace=True),
                                         nn.Dropout(dropout))

        # ---- decoder: stride-2 transposed convs, channels shrink; the last one outputs RGB ----
        dec = []
        for i in range(depth - 1, 0, -1):          # channels[-1] -> ... -> channels[0]
            dec.append(_up_block(channels[i], channels[i - 1]))
        dec.append(nn.ConvTranspose2d(channels[0], in_ch, kernel_size=4, stride=2, padding=1))
        self.decoder = nn.Sequential(*dec)

    # The three steps are separate methods so tests (and curious students) can inspect z.
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Image batch (N,3,H,W) -> latent vectors (N, bottleneck_dim)."""
        return self.to_latent(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Latent vectors (N, bottleneck_dim) -> images (N,3,H,W) in [0,1]."""
        h = self.from_latent(z).view(-1, self.final_channels, self.final_spatial, self.final_spatial)
        return torch.sigmoid(self.decoder(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))      # the ONLY path from input to output

    @classmethod
    def from_config(cls, model_cfg: dict) -> "UniversalAE":
        """Build from a config/checkpoint dict; unknown keys are ignored."""
        keys = ("in_ch", "base_channels", "depth", "bottleneck_dim", "dropout", "img_size")
        return cls(**{k: model_cfg[k] for k in keys if k in model_cfg})


Autoencoder = UniversalAE   # scaffold name kept as an alias


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Number of (trainable) parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def compression_info(model: UniversalAE) -> dict:
    """Input size vs latent size: how strongly the bottleneck compresses the image."""
    n_in = model.in_ch * model.img_size ** 2
    return {"input_values": n_in, "bottleneck_dim": model.bottleneck_dim,
            "compression_ratio": n_in / model.bottleneck_dim}


def describe(model: UniversalAE) -> str:
    """One-paragraph shape / compression summary (printed at the start of a run)."""
    x = torch.zeros(1, model.in_ch, model.img_size, model.img_size, device=next(model.parameters()).device)
    with torch.no_grad():
        was_training = model.training
        model.eval()
        feat = model.encoder(x)
        z = model.encode(x)
        y = model(x)
        model.train(was_training)
    info = compression_info(model)
    return (f"UniversalAE {model.hparams}\n"
            f"  input {tuple(x.shape[1:])} -> encoder features {tuple(feat.shape[1:])} -> "
            f"latent z {tuple(z.shape[1:])} -> output {tuple(y.shape[1:])}\n"
            f"  compression: {info['input_values']} input values -> {info['bottleneck_dim']} latent values "
            f"(ratio {info['compression_ratio']:.1f}:1)\n"
            f"  trainable parameters: {count_parameters(model):,}")
