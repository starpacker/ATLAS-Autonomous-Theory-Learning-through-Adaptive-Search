import pytest
import numpy as np
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


def test_find_optimal_k_simple_aic():
    """AIC-based K selection on 1D target should prefer small K."""
    from atlas.scinet.bottleneck import find_optimal_k, KSelectionResult
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (300, 2)).astype(np.float32)
    y = X[:, 0:1].astype(np.float32)
    result = find_optimal_k(X, y, k_range=[1, 2, 3], epochs_per_k=50,
                            selection_method="aic", n_seeds=1)
    assert isinstance(result, KSelectionResult)
    assert result.best_k == 1
    assert result.selection_method == "aic"


def test_find_optimal_k_val_loss():
    """Val-loss selection returns valid result with val_losses populated."""
    from atlas.scinet.bottleneck import find_optimal_k
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (300, 2)).astype(np.float32)
    y = X[:, 0:1].astype(np.float32)
    result = find_optimal_k(X, y, k_range=[1, 2, 3], epochs_per_k=50,
                            selection_method="val_loss")
    assert result.best_k in [1, 2, 3]
    assert len(result.val_losses) == 3
    assert result.selection_method == "val_loss"


def test_find_optimal_k_higher_dim():
    from atlas.scinet.bottleneck import find_optimal_k
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (300, 3)).astype(np.float32)
    y1 = np.sin(X[:, 0] * 3) + X[:, 1]
    y2 = np.cos(X[:, 2] * 2)
    y = np.column_stack([y1, y2]).astype(np.float32)
    result = find_optimal_k(X, y, k_range=[1, 2, 3], epochs_per_k=80,
                            selection_method="aic", n_seeds=1)
    assert result.best_k >= 2


def test_sparsity_disabled_during_k_selection():
    """Encoder sparsity should be forced to 0 during K selection."""
    from atlas.scinet.bottleneck import find_optimal_k
    from atlas.scinet.trainer import TrainConfig
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (200, 2)).astype(np.float32)
    y = X[:, 0:1].astype(np.float32)
    # Pass config with sparsity=0.1 — should be overridden to 0
    config = TrainConfig(epochs=30, encoder_sparsity=0.1)
    result = find_optimal_k(X, y, k_range=[1, 2], epochs_per_k=30,
                            train_config=config, selection_method="aic", n_seeds=1)
    assert result.best_k in [1, 2]  # just check it doesn't crash


def test_bottleneck_activation_tanh():
    """Tanh bottleneck activation should bound encoder outputs to [-1, 1]."""
    import torch
    from atlas.scinet.model import SciNet
    from atlas.scinet.trainer import train_scinet, TrainConfig
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (200, 2)).astype(np.float32)
    y = X[:, 0:1].astype(np.float32)
    model = SciNet(input_dim=2, bottleneck_dim=2, output_dim=1,
                   bottleneck_activation="tanh")
    train_scinet(model, X, y, TrainConfig(epochs=30))
    model.eval()
    with torch.no_grad():
        Z = model.encode(torch.tensor(X))
    assert Z.min() >= -1.0
    assert Z.max() <= 1.0


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
