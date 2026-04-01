"""Tests for SciNet encoder-decoder model."""
import pytest
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


def test_model_creation():
    from atlas.scinet.model import SciNet
    model = SciNet(input_dim=3, bottleneck_dim=2, output_dim=1)
    assert model.bottleneck_dim == 2


def test_forward_pass():
    from atlas.scinet.model import SciNet
    import torch
    model = SciNet(input_dim=3, bottleneck_dim=2, output_dim=1)
    x = torch.randn(10, 3)
    y_pred = model(x)
    assert y_pred.shape == (10, 1)


def test_encode():
    from atlas.scinet.model import SciNet
    import torch
    model = SciNet(input_dim=3, bottleneck_dim=2, output_dim=1)
    x = torch.randn(10, 3)
    z = model.encode(x)
    assert z.shape == (10, 2)


def test_decode():
    from atlas.scinet.model import SciNet
    import torch
    model = SciNet(input_dim=3, bottleneck_dim=2, output_dim=1)
    z = torch.randn(10, 2)
    y = model.decode(z)
    assert y.shape == (10, 1)


def test_different_bottleneck_dims():
    from atlas.scinet.model import SciNet
    import torch
    for k in [1, 2, 3, 5]:
        model = SciNet(input_dim=4, bottleneck_dim=k, output_dim=1)
        x = torch.randn(5, 4)
        z = model.encode(x)
        assert z.shape == (5, k)


def test_custom_hidden_layers():
    from atlas.scinet.model import SciNet
    import torch
    model = SciNet(input_dim=3, bottleneck_dim=2, output_dim=1,
                   encoder_hidden=[64, 32], decoder_hidden=[32, 64])
    x = torch.randn(10, 3)
    z = model.encode(x)
    assert z.shape == (10, 2)


def test_output_dim_array():
    from atlas.scinet.model import SciNet
    import torch
    model = SciNet(input_dim=3, bottleneck_dim=2, output_dim=100)
    x = torch.randn(10, 3)
    y = model(x)
    assert y.shape == (10, 100)
