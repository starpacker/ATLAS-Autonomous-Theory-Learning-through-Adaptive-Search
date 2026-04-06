import pytest
import numpy as np
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


def test_train_config():
    from atlas.scinet.trainer import TrainConfig
    cfg = TrainConfig()
    assert cfg.epochs == 200
    assert cfg.lr == 1e-3
    assert cfg.batch_size == 64


def test_train_reduces_loss():
    from atlas.scinet.trainer import train_scinet, TrainConfig
    from atlas.scinet.model import SciNet
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (200, 2)).astype(np.float32)
    y = (X[:, 0] + X[:, 1]).reshape(-1, 1).astype(np.float32)
    model = SciNet(input_dim=2, bottleneck_dim=2, output_dim=1)
    result = train_scinet(model, X, y, TrainConfig(epochs=50, lr=1e-3))
    assert result.final_loss < result.initial_loss
    assert result.final_loss < 0.1


def test_train_result_has_history():
    from atlas.scinet.trainer import train_scinet, TrainConfig
    from atlas.scinet.model import SciNet
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (100, 2)).astype(np.float32)
    y = X[:, 0:1].astype(np.float32)
    model = SciNet(input_dim=2, bottleneck_dim=1, output_dim=1)
    result = train_scinet(model, X, y, TrainConfig(epochs=20))
    assert len(result.loss_history) == 20


def test_train_config_new_fields():
    from atlas.scinet.trainer import TrainConfig
    cfg = TrainConfig()
    assert cfg.use_cosine_schedule is True
    assert cfg.min_lr == 1e-5
    assert cfg.val_fraction == 0.0


def test_train_with_validation_split():
    from atlas.scinet.trainer import train_scinet, TrainConfig
    from atlas.scinet.model import SciNet
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (200, 2)).astype(np.float32)
    y = (X[:, 0] + X[:, 1]).reshape(-1, 1).astype(np.float32)
    model = SciNet(input_dim=2, bottleneck_dim=2, output_dim=1)
    result = train_scinet(model, X, y, TrainConfig(epochs=30, val_fraction=0.2))
    assert result.val_loss_history is not None
    assert len(result.val_loss_history) == 30
    assert result.final_val_loss is not None
    assert result.final_val_loss > 0


def test_train_with_cosine_schedule():
    """Cosine schedule should still converge and reduce loss."""
    from atlas.scinet.trainer import train_scinet, TrainConfig
    from atlas.scinet.model import SciNet
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (200, 2)).astype(np.float32)
    y = (X[:, 0] * 2).reshape(-1, 1).astype(np.float32)
    model = SciNet(input_dim=2, bottleneck_dim=2, output_dim=1)
    result = train_scinet(model, X, y,
                          TrainConfig(epochs=50, use_cosine_schedule=True))
    assert result.final_loss < result.initial_loss
