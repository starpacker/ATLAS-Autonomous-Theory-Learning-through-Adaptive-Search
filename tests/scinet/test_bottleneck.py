import pytest
import numpy as np
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


def test_find_optimal_k_simple():
    from atlas.scinet.bottleneck import find_optimal_k, KSelectionResult
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (300, 2)).astype(np.float32)
    y = X[:, 0:1].astype(np.float32)
    result = find_optimal_k(X, y, k_range=[1, 2, 3], epochs_per_k=50)
    assert isinstance(result, KSelectionResult)
    assert result.best_k == 1


def test_find_optimal_k_higher_dim():
    from atlas.scinet.bottleneck import find_optimal_k
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (300, 3)).astype(np.float32)
    y1 = np.sin(X[:, 0] * 3) + X[:, 1]
    y2 = np.cos(X[:, 2] * 2)
    y = np.column_stack([y1, y2]).astype(np.float32)
    result = find_optimal_k(X, y, k_range=[1, 2, 3], epochs_per_k=80)
    assert result.best_k >= 2


def test_extract_bottleneck_vectors():
    from atlas.scinet.bottleneck import extract_bottleneck_vectors
    from atlas.scinet.model import SciNet
    from atlas.scinet.trainer import train_scinet, TrainConfig
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (100, 2)).astype(np.float32)
    y = X[:, 0:1].astype(np.float32)
    model = SciNet(input_dim=2, bottleneck_dim=2, output_dim=1)
    train_scinet(model, X, y, TrainConfig(epochs=20))
    Z = extract_bottleneck_vectors(model, X)
    assert Z.shape == (100, 2)
    assert isinstance(Z, np.ndarray)
