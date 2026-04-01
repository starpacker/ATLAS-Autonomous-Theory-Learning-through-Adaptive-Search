"""Tests for experiment-centric verification with global MDL."""
import numpy as np
from atlas.multi_agent.verifier import (
    VerificationResult, compute_global_mdl_delta, is_statistically_significant,
)


def test_global_mdl_decrease_adopted():
    """Extension that decreases total MDL should be adoptable."""
    per_env_deltas = {
        "ENV_01": {"mu": -5.0, "sigma": 1.0},
        "ENV_02": {"mu": -3.0, "sigma": 0.5},
        "ENV_04": {"mu": -10.0, "sigma": 2.0},
        "ENV_09": {"mu": 0.5, "sigma": 0.3},   # slight degradation
        "ENV_10": {"mu": 0.2, "sigma": 0.2},
    }
    result = compute_global_mdl_delta(per_env_deltas)
    assert result.delta_total_mdl < 0
    assert result.should_adopt


def test_global_mdl_increase_rejected():
    """Extension that increases total MDL should be rejected."""
    per_env_deltas = {
        "ENV_01": {"mu": 2.0, "sigma": 0.5},
        "ENV_02": {"mu": 3.0, "sigma": 1.0},
        "ENV_04": {"mu": -0.5, "sigma": 0.5},
    }
    result = compute_global_mdl_delta(per_env_deltas)
    assert result.delta_total_mdl > 0
    assert not result.should_adopt


def test_statistical_significance():
    assert is_statistically_significant(-20.0, 5.0)   # |delta| >> noise
    assert not is_statistically_significant(-1.0, 5.0)  # |delta| < 2*noise


def test_verification_result_structure():
    result = VerificationResult(
        delta_total_mdl=-15.0,
        pooled_noise=3.0,
        per_env_results={"ENV_01": {"mu": -5.0}},
        should_adopt=True,
        reason="Global MDL decreased significantly",
    )
    assert result.should_adopt


def test_noisy_but_net_positive():
    """Noisy per-experiment results but net MDL decrease should adopt."""
    per_env_deltas = {
        f"ENV_{i:02d}": {"mu": -2.0 + np.random.default_rng(i).normal(0, 0.5),
                         "sigma": 1.0}
        for i in range(1, 13)
    }
    result = compute_global_mdl_delta(per_env_deltas)
    # Most experiments benefit -> net negative MDL
    assert result.delta_total_mdl < 0
