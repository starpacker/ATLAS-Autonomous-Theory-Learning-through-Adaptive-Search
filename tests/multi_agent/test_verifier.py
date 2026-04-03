"""Tests for experiment-centric verification with global MDL."""
import numpy as np
import pytest
from atlas.multi_agent.verifier import (
    VerificationResult, compute_global_mdl_delta, is_statistically_significant,
    verify_proposal_sr,
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


# ── SR-based verification ──────────────────────────────────────────────────

def test_verify_proposal_sr_no_pysr():
    """Without PySR installed, verify_proposal_sr returns None gracefully."""
    X = np.random.default_rng(42).normal(size=(50, 2))
    y = X[:, 0] + X[:, 1]
    try:
        result = verify_proposal_sr(
            X, y, X, y, var_names=["x0", "x1"],
            n_seeds=1, sr_niterations=5,
        )
        # If PySR IS installed, result should be a dict with expected keys
        if result is not None:
            assert "mu" in result
            assert "sigma" in result
            assert "n_seeds" in result
            assert result["method"] == "sr"
    except ImportError:
        pytest.skip("PySR not available")


def test_verify_proposal_sr_with_concepts():
    """SR-based verification with concept columns should return valid delta."""
    rng = np.random.default_rng(42)
    X_train = rng.uniform(0, 1, size=(50, 2))
    X_test = rng.uniform(0, 1, size=(20, 2))
    # True relation: y = x0 + cos(x0)^2
    y_train = X_train[:, 0] + np.cos(X_train[:, 0]) ** 2
    y_test = X_test[:, 0] + np.cos(X_test[:, 0]) ** 2
    # Concept column that should help: cos(x0)^2
    concept_train = {"cos2_x0": np.cos(X_train[:, 0]) ** 2}
    concept_test = {"cos2_x0": np.cos(X_test[:, 0]) ** 2}
    try:
        result = verify_proposal_sr(
            X_train, y_train, X_test, y_test,
            var_names=["x0", "x1"],
            concept_columns=concept_train,
            concept_columns_test=concept_test,
            n_seeds=1, sr_niterations=5,
        )
        if result is not None:
            assert "mu" in result
            assert "sigma" in result
    except ImportError:
        pytest.skip("PySR not available")


def test_global_mdl_with_mixed_methods():
    """compute_global_mdl_delta works fine with extra keys like 'method'."""
    per_env_deltas = {
        "ENV_01": {"mu": -5.0, "sigma": 1.0, "method": "sr"},
        "ENV_02": {"mu": -3.0, "sigma": 0.5, "method": "estimate"},
    }
    result = compute_global_mdl_delta(per_env_deltas)
    assert result.delta_total_mdl == -8.0
    assert result.should_adopt
