"""Tests for D1-D5 diagnostic tests."""
import numpy as np
from atlas.analysis.diagnostics import (
    DiagnosticResult, diagnose_stochasticity, diagnose_discreteness,
    diagnose_residual_structure, run_all_diagnostics,
)
from atlas.data.dataset import ExperimentDataset


def test_d1_stochastic_positive():
    result = diagnose_stochasticity(repeated_outputs=[
        np.array([1.0, 0.0, 0.0, 1.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
    ])
    assert result.triggered
    assert result.diagnostic_id == "D1"


def test_d1_stochastic_negative():
    same = np.array([1.0, 2.0, 3.0])
    result = diagnose_stochasticity(repeated_outputs=[same, same, same])
    assert not result.triggered


def test_d2_discrete_positive():
    outputs = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    result = diagnose_discreteness(outputs, max_clusters=10)
    assert result.triggered
    assert result.diagnostic_id == "D2"
    assert result.details["n_clusters"] == 2


def test_d2_discrete_negative():
    rng = np.random.default_rng(42)
    outputs = rng.uniform(0, 1, 1000)
    result = diagnose_discreteness(outputs, max_clusters=10)
    assert not result.triggered


def test_d4_residual_structure_positive():
    x = np.linspace(0, 10, 200)
    residuals = np.sin(5 * x)
    result = diagnose_residual_structure(residuals)
    assert result.triggered
    assert result.diagnostic_id == "D4"


def test_d4_residual_structure_negative():
    rng = np.random.default_rng(42)
    residuals = rng.normal(0, 1, 200)
    result = diagnose_residual_structure(residuals)
    assert not result.triggered


def test_run_all_returns_list():
    ds = ExperimentDataset("ENV_TEST", ["knob_0"], ["detector_0"])
    for i in range(50):
        ds.add({"knob_0": i / 50}, {"detector_0": float(i)})
    results = run_all_diagnostics(ds, best_r_squared=0.99, residuals=np.zeros(50))
    assert isinstance(results, list)
    for r in results:
        assert isinstance(r, DiagnosticResult)
