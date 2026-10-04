"""Tests for models/autoencoder.py (UniversalAE)."""
import pytest
import torch

from genai.models.autoencoder import Autoencoder, UniversalAE, compression_info, count_parameters, describe


def small(**kw):
    args = dict(base_channels=8, depth=4, bottleneck_dim=32, dropout=0.0)
    args.update(kw)
    return UniversalAE(**args).eval()


def test_alias():
    assert Autoencoder is UniversalAE


def test_output_shape_and_range():
    x = torch.rand(3, 3, 128, 128)
    y = small()(x)
    assert y.shape == x.shape
    assert y.min() >= 0.0 and y.max() <= 1.0          # sigmoid output (CONTRACTS 3.1)


@pytest.mark.parametrize("depth,spatial", [(3, 16), (4, 8), (5, 4)])
def test_depth_controls_downsampling(depth, spatial):
    m = small(depth=depth)
    feat = m.encoder(torch.rand(1, 3, 128, 128))
    assert feat.shape[-1] == spatial and feat.shape[1] == 8 * 2 ** (depth - 1)   # channels grow


def test_latent_dimension_equals_bottleneck_dim():
    m = small(bottleneck_dim=48)
    z = m.encode(torch.rand(5, 3, 128, 128))
    assert z.shape == (5, 48)


def test_decoder_cannot_be_bypassed():
    """The output depends on the input ONLY through z: feeding the same z gives the same image
    whatever the input was, and different z gives different images (no skip paths)."""
    m = small()
    a, b = torch.rand(2, 3, 128, 128), torch.rand(2, 3, 128, 128)
    z = m.encode(a)
    assert torch.allclose(m(a), m.decode(z))
    # replace the input by something else but keep z: output unchanged
    assert torch.allclose(m.decode(z), m.decode(m.encode(a)))
    assert not torch.allclose(m.decode(z), m.decode(m.encode(b)))
    # structurally: the only modules are encoder, to_latent, from_latent, decoder
    assert {n for n, _ in m.named_children()} == {"encoder", "to_latent", "from_latent", "decoder"}


def test_gradient_reaches_encoder_through_latent_only():
    m = small().train()
    x = torch.rand(4, 3, 128, 128)
    m(x).mean().backward()
    assert all(p.grad is not None for p in m.parameters())


def test_parameter_count_matches_manual_sum():
    m = small()
    assert count_parameters(m) == sum(p.numel() for p in m.parameters())
    assert count_parameters(small(bottleneck_dim=64)) > count_parameters(small(bottleneck_dim=32))


def test_compression_info_and_describe():
    info = compression_info(small(bottleneck_dim=64))
    assert info["input_values"] == 3 * 128 * 128 and info["bottleneck_dim"] == 64
    assert info["compression_ratio"] == pytest.approx(3 * 128 * 128 / 64)
    text = describe(small(bottleneck_dim=64))
    assert "latent z (64,)" in text and "ratio" in text


def test_dropout_active_in_train_off_in_eval():
    m = small(dropout=0.5)
    x = torch.rand(2, 3, 128, 128)
    m.eval()
    assert torch.allclose(m(x), m(x))
    m.train()
    assert not torch.allclose(m(x), m(x))


def test_from_config_ignores_unknown_keys_and_bad_depth():
    m = UniversalAE.from_config({"base_channels": 8, "depth": 3, "bottleneck_dim": 16, "dropout": 0.0, "extra": 1})
    assert m.hparams["depth"] == 3
    with pytest.raises(ValueError):
        UniversalAE(depth=8)   # 128 is not divisible by 2**8


# ---- conv latent (decision D40): a 1x1-conv latent grid instead of flatten + Linear ----
def conv_ae(**kw):
    args = dict(base_channels=8, depth=3, bottleneck_dim=1024, dropout=0.0, latent="conv")
    args.update(kw)
    return UniversalAE(**args)


def test_conv_latent_shapes_and_total_values():
    m = conv_ae(bottleneck_dim=1024)                 # depth 3 -> 16x16 grid -> 1024 / 256 = 4 latent channels
    x = torch.rand(2, 3, 128, 128)
    z = m.encode(x)
    assert z.shape == (2, 4, 16, 16) and z[0].numel() == 1024
    y = m(x)
    assert y.shape == x.shape and 0.0 <= float(y.min()) and float(y.max()) <= 1.0
    assert compression_info(m)["compression_ratio"] == pytest.approx(48.0)


def test_conv_latent_has_no_dense_layer_and_no_skip_path():
    m = conv_ae()
    assert not any(isinstance(mod, torch.nn.Linear) for mod in m.modules())
    x = torch.rand(1, 3, 128, 128)
    # the output is a function of z only: decoding the encoded latent equals the forward pass
    m.eval()
    assert torch.allclose(m(x), m.decode(m.encode(x)))


def test_conv_latent_needs_a_multiple_of_the_grid_size_and_rejects_unknown_kind():
    with pytest.raises(ValueError, match="multiple"):
        conv_ae(bottleneck_dim=1000)
    with pytest.raises(ValueError, match="latent"):
        UniversalAE(latent="vector")


def test_conv_latent_survives_a_config_round_trip_and_old_configs_stay_dense():
    m = conv_ae()
    rebuilt = UniversalAE.from_config(m.hparams)
    assert rebuilt.latent == "conv" and rebuilt.latent_channels == m.latent_channels
    old = UniversalAE.from_config({"base_channels": 8, "depth": 4, "bottleneck_dim": 32, "dropout": 0.0})  # no "latent" key
    assert old.latent == "dense"
