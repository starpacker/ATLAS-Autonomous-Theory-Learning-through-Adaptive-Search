"""Tests for ENV-04."""
import numpy as np
import pytest
from atlas.environments.env_04 import Env04
from atlas.types import KnobType


@pytest.fixture
def env():
    return Env04(seed=42)


def _knobs(knob_0=0.3, knob_1=0.5, knob_2=0.5, knob_3=1_000_000):
    return {"knob_0": knob_0, "knob_1": knob_1, "knob_2": knob_2, "knob_3": knob_3}


# --- Schema tests ---

def test_schema_env_id(env):
    assert env.get_schema().env_id == "ENV_04"


def test_schema_knob_count(env):
    assert len(env.get_schema().knobs) == 4


def test_schema_detector_count(env):
    assert len(env.get_schema().detectors) == 1


def test_no_physics_names_in_schema(env):
    schema = env.get_schema()
    for knob in schema.knobs:
        assert knob.name.startswith("knob_")
    for det in schema.detectors:
        assert det.name.startswith("detector_")


def test_entity_label(env):
    assert "entity_A" in env.get_schema().entities


def test_continuous_knobs_normalized(env):
    schema = env.get_schema()
    for knob in schema.knobs:
        if knob.knob_type == KnobType.CONTINUOUS:
            assert knob.range_min == 0.0
            assert knob.range_max == 1.0


def test_integer_knob_range(env):
    schema = env.get_schema()
    int_knobs = [k for k in schema.knobs if k.knob_type == KnobType.INTEGER]
    assert len(int_knobs) == 1
    assert int_knobs[0].range_min == 1
    assert int_knobs[0].range_max == 1_000_000


# --- Output shape ---

def test_output_is_array_1000(env):
    result = env.run(_knobs())
    arr = result["detector_0"]
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1000,)


# --- High-intensity: smooth interference pattern ---

def test_high_intensity_smooth_pattern(env):
    """High intensity should give a smooth pattern dominated by low frequencies."""
    result = env.run(_knobs(knob_3=1_000_000))
    arr = result["detector_0"]
    fft_mag = np.abs(np.fft.rfft(arr))
    low_power = np.sum(fft_mag[1:20] ** 2)
    high_power = np.sum(fft_mag[20:] ** 2)
    assert low_power > high_power, "High-intensity pattern should be dominated by low frequencies"


def test_high_intensity_normalized(env):
    result = env.run(_knobs(knob_3=1_000_000))
    arr = result["detector_0"]
    assert np.max(arr) <= 1.0 + 1e-9
    assert np.min(arr) >= -1e-9


# --- Low-intensity: discrete sparse hits ---

def test_low_intensity_sparse(env):
    """Low intensity (N=100) should produce sparse output (mostly zeros)."""
    result = env.run(_knobs(knob_3=100))
    arr = result["detector_0"]
    nonzero_frac = np.count_nonzero(arr) / len(arr)
    assert nonzero_frac < 0.10, f"Expected sparse output, got {nonzero_frac:.2%} nonzero"


def test_low_intensity_integer_counts(env):
    """Low-intensity output should contain non-negative integer counts."""
    result = env.run(_knobs(knob_3=50))
    arr = result["detector_0"]
    assert np.all(arr >= 0)
    assert np.all(arr == np.floor(arr)), "Counts should be integers"


def test_low_intensity_total_count(env):
    """Total hits should equal N."""
    N = 200
    result = env.run(_knobs(knob_3=N))
    assert int(np.sum(result["detector_0"])) == N


# --- Stochasticity ---

def test_low_intensity_stochastic_different_seeds():
    """Different seeds must give different outputs."""
    knobs = _knobs(knob_3=500)
    r1 = Env04(seed=1).run(knobs)
    r2 = Env04(seed=2).run(knobs)
    assert not np.array_equal(r1["detector_0"], r2["detector_0"])


def test_low_intensity_same_seed_reproducible():
    """Same seed must give same output."""
    knobs = _knobs(knob_3=500)
    r1 = Env04(seed=99).run(knobs)
    r2 = Env04(seed=99).run(knobs)
    np.testing.assert_array_equal(r1["detector_0"], r2["detector_0"])


def test_high_intensity_deterministic(env):
    """High-intensity pattern must be fully deterministic."""
    knobs = _knobs(knob_3=1_000_000)
    r1 = env.run(knobs)
    r2 = env.run(knobs)
    np.testing.assert_array_equal(r1["detector_0"], r2["detector_0"])


# --- Statistical convergence ---

def test_statistical_convergence():
    """Many low-intensity runs should converge to high-intensity pattern."""
    knobs_low = _knobs(knob_3=100)
    knobs_high = _knobs(knob_3=1_000_000)

    accumulated = np.zeros(1000)
    for seed in range(500):
        result = Env04(seed=seed).run(knobs_low)
        accumulated += result["detector_0"]

    high_result = Env04(seed=0).run(knobs_high)
    high_pattern = high_result["detector_0"]

    # Normalize both for correlation
    acc_norm = accumulated / np.sum(accumulated)
    hi_norm = high_pattern / np.sum(high_pattern)

    corr = np.corrcoef(acc_norm, hi_norm)[0, 1]
    assert corr > 0.8, f"Statistical convergence failed: correlation={corr:.3f}"


# --- Knob sensitivity ---

def test_knob_0_changes_pattern(env):
    """Changing slit width knob must change the output pattern."""
    r1 = env.run(_knobs(knob_0=0.1, knob_3=1_000_000))
    r2 = env.run(_knobs(knob_0=0.9, knob_3=1_000_000))
    assert not np.allclose(r1["detector_0"], r2["detector_0"])


def test_knob_1_changes_pattern(env):
    """Changing slit separation must change interference fringe spacing."""
    r1 = env.run(_knobs(knob_1=0.1, knob_3=1_000_000))
    r2 = env.run(_knobs(knob_1=0.9, knob_3=1_000_000))
    assert not np.allclose(r1["detector_0"], r2["detector_0"])


# --- Validation ---

def test_invalid_knob_out_of_range(env):
    with pytest.raises(ValueError):
        env.run({"knob_0": 1.5, "knob_1": 0.5, "knob_2": 0.5, "knob_3": 1000})


def test_missing_knob_raises(env):
    with pytest.raises(ValueError):
        env.run({"knob_0": 0.5, "knob_1": 0.5, "knob_2": 0.5})
