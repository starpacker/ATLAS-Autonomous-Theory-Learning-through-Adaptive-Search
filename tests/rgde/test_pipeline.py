# tests/rgde/test_pipeline.py
import pytest
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from atlas.rgde.pipeline import RGDEConfig, RGDEResult, run_rgde


def test_rgde_config_defaults():
    cfg = RGDEConfig()
    assert cfg.k_range == [1, 2, 3, 4, 5]
    assert cfg.scinet_epochs == 200


def test_rgde_result_structure():
    result = RGDEResult(success=False, dsl_type=None, decoder_formula=None,
                        r2_before=0.3, r2_after=-1.0, evaluation=None, k_selected=None)
    assert not result.success


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_rgde_simple_1d():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.uniform(0, 1, (n, 2)).astype(np.float32)
    y = (X[:, 0] ** 2).reshape(-1, 1).astype(np.float32)
    config = RGDEConfig(k_range=[1, 2], scinet_epochs=50, sr_niterations=5, sr_maxsize=8)
    result = run_rgde(X, y, var_names=["knob_0", "knob_1"], r2_before=0.1,
                      env_id="ENV_T2", config=config)
    assert result.k_selected is not None
