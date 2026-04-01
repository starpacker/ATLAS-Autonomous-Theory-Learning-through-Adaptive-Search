import pytest
import numpy as np
from atlas.rgde.encoder_sr import run_encoder_sr, EncoderSRResult
try:
    import pysr
    HAS_PYSR = True
except ImportError:
    HAS_PYSR = False

def test_encoder_sr_result_structure():
    result = EncoderSRResult(formulas={}, r_squared_per_dim={}, success=False)
    assert not result.success

def test_encoder_sr_without_pysr():
    X = np.random.randn(50, 2)
    Z = np.random.randn(50, 2)
    result = run_encoder_sr(X, Z, var_names=["knob_0", "knob_1"], niterations=5, maxsize=5)
    assert isinstance(result, EncoderSRResult)

@pytest.mark.skipif(not HAS_PYSR, reason="PySR not installed")
def test_encoder_sr_on_known_mapping():
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (100, 2))
    Z = np.column_stack([X[:, 0] * 2, X[:, 1] + 1])
    result = run_encoder_sr(X, Z, var_names=["knob_0", "knob_1"], niterations=20, maxsize=10)
    assert result.success
    assert len(result.formulas) == 2
