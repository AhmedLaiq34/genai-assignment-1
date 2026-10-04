"""Task 4 networks (models/cgan.py): shapes, ranges, style conditioning, skips, config round trip.

Tiny networks (base_channels 8, style_dim 4) keep these CPU tests fast. Plan: docs/TASK4_PLAN.md section E.
"""
import pytest
import torch
from torch import nn

from genai.models.cgan import Discriminator, Generator, count_parameters, init_weights
from genai.tasks.task4.train import d_step, g_step

MODEL_CFG = {"base_channels": 8, "style_dim": 4, "dropout": 0.25, "num_styles": 3}


def make_batch(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    photo = torch.rand(n, 3, 128, 128, generator=g) * 2 - 1
    sketch = torch.rand(n, 1, 128, 128, generator=g) * 2 - 1
    style = torch.arange(n) % 3
    return photo, sketch, style


@pytest.fixture()
def nets():
    torch.manual_seed(0)
    return Generator.from_config(MODEL_CFG), Discriminator.from_config(MODEL_CFG)


@pytest.mark.parametrize("n", [1, 3])
def test_shapes_and_range(nets, n):
    G, D = nets
    photo, sketch, style = make_batch(n)
    G.eval()
    out = G(photo, style)
    assert out.shape == (n, 1, 128, 128) and out.dtype == torch.float32
    assert out.min() >= -1.0 and out.max() <= 1.0            # tanh output
    logits = D(photo, sketch, style)
    assert logits.shape == (n, 1, 14, 14)                    # 70x70 PatchGAN at 128 px


def test_style_changes_output_of_g_and_d(nets):
    G, D = nets
    G.eval()
    photo, sketch, _ = make_batch(1)
    outs = [G(photo, torch.tensor([k])) for k in range(3)]
    logits = [D(photo, sketch, torch.tensor([k])) for k in range(3)]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert (outs[a] - outs[b]).abs().max() > 1e-6        # G ignores no style
        assert (logits[a] - logits[b]).abs().max() > 1e-6    # D ignores no style


def test_separate_embedding_tables(nets):
    G, D = nets
    assert G.style_embed is not D.style_embed
    assert G.style_embed.weight.shape == (3, 4) and D.style_embed.weight.shape == (3, 4)


def test_embedding_gradients_after_one_step(nets):
    """After one d_step the style embedding of D has a gradient, after one g_step that of G has one."""
    G, D = nets
    G.train()
    D.train()
    opt_g = torch.optim.Adam(G.parameters(), lr=1e-4)
    opt_d = torch.optim.Adam(D.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    batch = make_batch(3)
    d_step(G, D, opt_d, batch, scaler)
    assert D.style_embed.weight.grad is not None and D.style_embed.weight.grad.abs().sum() > 0
    g_step(G, D, opt_g, batch, 100.0, scaler)
    assert G.style_embed.weight.grad is not None and G.style_embed.weight.grad.abs().sum() > 0


def test_skip_connections_are_wired(nets):
    """Zero the skip half of the input of the output layer: the output must change."""
    G, _ = nets
    G.eval()
    photo, _, style = make_batch(2)
    with torch.no_grad():
        before = G(photo, style)

    def zero_skip(module, args):
        x = args[0].clone()
        x[:, G.hparams["base_channels"]:] = 0.0              # the last c channels are the skip from down1
        return (x,)

    handle = G.out.register_forward_pre_hook(zero_skip)
    with torch.no_grad():
        after = G(photo, style)
    handle.remove()
    assert (before - after).abs().max() > 1e-6


def test_skip_channel_counts():
    """The decoder inputs are [previous stage, skip]: 8c+8c, 4c+4c, 2c+2c and c+c channels."""
    G = Generator(base_channels=8, style_dim=4, dropout=0.1)
    assert G.up2[0].in_channels == 16 * 8 and G.up3[0].in_channels == 8 * 8
    assert G.up4[0].in_channels == 4 * 8 and G.out.in_channels == 2 * 8


def test_no_batchnorm_and_dropout_placement():
    G = Generator(base_channels=8, style_dim=4, dropout=0.3)
    D = Discriminator(base_channels=8, style_dim=4)
    for net in (G, D):
        assert not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in net.modules())
    has_dropout = lambda block: any(isinstance(m, nn.Dropout) for m in block.modules())  # noqa: E731
    assert all(has_dropout(b) for b in (G.up1, G.up2, G.up3))
    assert not has_dropout(G.up4)
    assert not any(isinstance(m, nn.Dropout) for m in D.modules())
    assert not any(isinstance(m, nn.Dropout) for m in Generator(8, 4, 0.0).modules())   # dropout 0 -> none


def test_dropout_only_in_train_mode(nets):
    G, _ = nets
    photo, _, style = make_batch(2)
    G.eval()
    with torch.no_grad():
        assert torch.equal(G(photo, style), G(photo, style))          # deterministic in eval
    G.train()
    with torch.no_grad():
        assert not torch.equal(G(photo, style), G(photo, style))      # dropout active in train


def test_from_config_round_trip(nets):
    G, D = nets
    cfg = {**MODEL_CFG, "unknown_key": 1}                     # unknown keys are ignored
    G2, D2 = Generator.from_config(cfg), Discriminator.from_config(cfg)
    assert G2.hparams == G.hparams == {"base_channels": 8, "style_dim": 4, "dropout": 0.25, "num_styles": 3}
    assert D2.hparams == D.hparams
    G2.load_state_dict(G.state_dict())                        # same architecture -> weights load
    D2.load_state_dict(D.state_dict())
    G.eval()
    G2.eval()
    photo, _, style = make_batch(2)
    assert torch.equal(G(photo, style), G2(photo, style))
    # config without num_styles / dropout falls back to the defaults
    assert Generator.from_config({"base_channels": 8, "style_dim": 4}).hparams["num_styles"] == 3


def test_init_weights_scale():
    conv = nn.Conv2d(3, 64, 4)
    conv.apply(init_weights)
    assert abs(conv.weight.std().item() - 0.02) < 0.005 and conv.bias.abs().sum() == 0


def test_parameter_counts_are_positive(nets):
    assert count_parameters(nets[0]) > count_parameters(nets[1]) > 0
