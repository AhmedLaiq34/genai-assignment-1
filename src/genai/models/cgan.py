"""Style-conditioned U-Net generator and PatchGAN discriminator for Task 4 (photo -> sketch).

Implements CONTRACTS 3.7 (FS2K tensors: photo N x 3 x 128 x 128, sketch N x 1 x 128 x 128, both in [-1,1],
style ids 0/1/2, a learned `nn.Embedding(3, d)` inside BOTH networks) and 3.9 (the generator's ONNX
interface: inputs `photo` float32 [N,3,128,128] and `style` int64 [N], output `sketch` [N,1,128,128] in [-1,1]).
Plan: docs/TASK4_PLAN.md C6 (generator), C7 (discriminator), C8 (ONNX safety), C9 (weight initialisation).

Generator G(photo, style)                       (base channels c, style embedding size d)

    photo 3x128x128 + style embedding tiled to d x 128 x 128 ........ 3+d channels
    encoder (4x4 conv, stride 2):  e1 64x64 c (no norm) | e2 32x32 2c | e3 16x16 4c | e4 8x8 8c | e5 4x4 8c
    bottleneck: e5 + the style embedding tiled again to 4x4, then a 3x3 conv -> 8c at 4x4
    decoder (4x4 transposed conv, stride 2), every stage is concatenated with the matching encoder feature:
        u1 8x8 8c  + e4  |  u2 16x16 4c + e3  |  u3 32x32 2c + e2   (these three use dropout)
        u4 64x64 c + e1  (no dropout)
        output: transposed conv 2c -> 1 channel at 128x128, then tanh

Discriminator D(photo, sketch, style): the 70x70 PatchGAN. The input is photo(3) + sketch(1) + the style
embedding tiled to d channels, so D judges "is this a real sketch OF THIS STYLE for this photo".
Four convs (c, 2c, 4c, 8c channels) and a last conv to 1 channel give a 14 x 14 map of logits (no sigmoid).

Rules that keep the generator exportable to ONNX (plan C8): only Conv, ConvTranspose, InstanceNorm,
LeakyReLU/ReLU, Tanh, Concat, Gather (the embedding) and Expand are used; no BatchNorm (so train and eval
behave the same apart from dropout); tiling is `emb[:, :, None, None].expand(-1, -1, H, W)`.
"""
from __future__ import annotations

import torch
from torch import nn

NUM_STYLES = 3


# ------------------------------------------------------------------------------------ helpers
def _tile(emb: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Style vectors (N, d) -> a feature map (N, d, H, W) where every pixel holds the same vector."""
    return emb[:, :, None, None].expand(-1, -1, height, width)


def _down(c_in: int, c_out: int, norm: bool = True) -> nn.Sequential:
    """Halve the spatial size: 4x4 conv, stride 2, padding 1; InstanceNorm (optional) + LeakyReLU(0.2)."""
    layers = [nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=not norm)]
    if norm:
        layers.append(nn.InstanceNorm2d(c_out, affine=True))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def _up(c_in: int, c_out: int, dropout: float = 0.0) -> nn.Sequential:
    """Double the spatial size: 4x4 transposed conv, stride 2, padding 1; InstanceNorm, dropout, ReLU."""
    layers = [nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=False),
              nn.InstanceNorm2d(c_out, affine=True)]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def init_weights(module: nn.Module) -> None:
    """Weight initialisation for `model.apply(init_weights)` (plan C9).

    Conv / ConvTranspose weights ~ N(0, 0.02); InstanceNorm scale ~ N(1, 0.02) and shift 0.
    The style embedding keeps PyTorch's default N(0, 1): with 0.02 the three styles would start almost
    identical and the networks could learn to ignore them (a small deviation from "everything N(0, 0.02)").
    """
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.InstanceNorm2d) and module.affine:
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


# --------------------------------------------------------------------------------- generator
class Generator(nn.Module):
    """U-Net generator G(photo, style) -> sketch (see the module docstring for the layer list).

    Args:
        base_channels: c, channels of the first stage (the deepest stages have 8c).
        style_dim:     d, size of the learned style embedding.
        dropout:       dropout probability in the first three decoder stages (0 = off).
        num_styles:    number of style ids (3 for FS2K).
    """

    def __init__(self, base_channels: int, style_dim: int, dropout: float, num_styles: int = NUM_STYLES):
        super().__init__()
        c, d = int(base_channels), int(style_dim)
        # The constructor arguments are kept so that the model can be rebuilt (and printed).
        self.hparams = dict(base_channels=c, style_dim=d, dropout=float(dropout), num_styles=int(num_styles))
        self.style_embed = nn.Embedding(num_styles, d)       # G's OWN embedding table

        # encoder: 128 -> 64 -> 32 -> 16 -> 8 -> 4
        self.down1 = _down(3 + d, c, norm=False)             # input = photo + tiled style
        self.down2 = _down(c, 2 * c)
        self.down3 = _down(2 * c, 4 * c)
        self.down4 = _down(4 * c, 8 * c)
        self.down5 = _down(8 * c, 8 * c)

        # bottleneck at 4x4: features + tiled style -> 3x3 conv (keeps 4x4)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(8 * c + d, 8 * c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(8 * c, affine=True),
            nn.ReLU(inplace=True))

        # decoder: 4 -> 8 -> 16 -> 32 -> 64, each input is [previous stage, skip] (hence the doubled channels)
        self.up1 = _up(8 * c, 8 * c, dropout)                # then concat with down4 (8c)  -> 16c
        self.up2 = _up(16 * c, 4 * c, dropout)               # then concat with down3 (4c)  -> 8c
        self.up3 = _up(8 * c, 2 * c, dropout)                # then concat with down2 (2c)  -> 4c
        self.up4 = _up(4 * c, c, 0.0)                        # then concat with down1 (c)   -> 2c; no dropout here
        self.out = nn.ConvTranspose2d(2 * c, 1, kernel_size=4, stride=2, padding=1)   # 64 -> 128, 1 channel

        self.apply(init_weights)

    def forward(self, photo: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """photo (N,3,128,128) in [-1,1], style int64 (N,) -> sketch (N,1,128,128) in [-1,1]."""
        emb = self.style_embed(style)                                        # (N, d)
        e1 = self.down1(torch.cat([photo, _tile(emb, photo.shape[2], photo.shape[3])], dim=1))   # 64x64
        e2 = self.down2(e1)                                                  # 32x32
        e3 = self.down3(e2)                                                  # 16x16
        e4 = self.down4(e3)                                                  # 8x8
        e5 = self.down5(e4)                                                  # 4x4
        b = self.bottleneck(torch.cat([e5, _tile(emb, e5.shape[2], e5.shape[3])], dim=1))       # 4x4
        u = torch.cat([self.up1(b), e4], dim=1)                              # 8x8,   16c
        u = torch.cat([self.up2(u), e3], dim=1)                              # 16x16, 8c
        u = torch.cat([self.up3(u), e2], dim=1)                              # 32x32, 4c
        u = torch.cat([self.up4(u), e1], dim=1)                              # 64x64, 2c
        return torch.tanh(self.out(u))                                       # 128x128, 1 channel

    @classmethod
    def from_config(cls, model_cfg: dict) -> "Generator":
        """Build from the config / checkpoint `model` dict (unknown keys are ignored)."""
        return cls(base_channels=model_cfg["base_channels"], style_dim=model_cfg["style_dim"],
                   dropout=model_cfg.get("dropout", 0.0), num_styles=model_cfg.get("num_styles", NUM_STYLES))


# ----------------------------------------------------------------------------- discriminator
class Discriminator(nn.Module):
    """70x70 PatchGAN D(photo, sketch, style) -> logits (N, 1, 14, 14) (no sigmoid, no dropout).

    Args:
        base_channels: c, channels of the first conv (the last hidden conv has 8c).
        style_dim:     d, size of D's own style embedding.
        num_styles:    number of style ids (3 for FS2K).
    """

    def __init__(self, base_channels: int, style_dim: int, num_styles: int = NUM_STYLES):
        super().__init__()
        c, d = int(base_channels), int(style_dim)
        self.hparams = dict(base_channels=c, style_dim=d, num_styles=int(num_styles))
        self.style_embed = nn.Embedding(num_styles, d)       # D's OWN embedding table (not shared with G)
        self.net = nn.Sequential(
            _down(3 + 1 + d, c, norm=False),                                         # 128 -> 64
            _down(c, 2 * c),                                                         # 64 -> 32
            _down(2 * c, 4 * c),                                                     # 32 -> 16
            nn.Conv2d(4 * c, 8 * c, kernel_size=4, stride=1, padding=1, bias=False),  # 16 -> 15
            nn.InstanceNorm2d(8 * c, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8 * c, 1, kernel_size=4, stride=1, padding=1))                 # 15 -> 14: one logit per patch
        self.apply(init_weights)

    def forward(self, photo: torch.Tensor, sketch: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """photo (N,3,H,W), sketch (N,1,H,W), style int64 (N,) -> logits (N,1,14,14) for H = W = 128."""
        emb = _tile(self.style_embed(style), photo.shape[2], photo.shape[3])
        return self.net(torch.cat([photo, sketch, emb], dim=1))

    @classmethod
    def from_config(cls, model_cfg: dict) -> "Discriminator":
        """Build from the config / checkpoint `model` dict (dropout is a generator-only key and is ignored)."""
        return cls(base_channels=model_cfg["base_channels"], style_dim=model_cfg["style_dim"],
                   num_styles=model_cfg.get("num_styles", NUM_STYLES))


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Number of (trainable) parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def describe(generator: Generator, discriminator: Discriminator) -> str:
    """Two-line summary printed at the start of a run."""
    return (f"Generator {generator.hparams}: {count_parameters(generator):,} parameters\n"
            f"Discriminator {discriminator.hparams}: {count_parameters(discriminator):,} parameters")
