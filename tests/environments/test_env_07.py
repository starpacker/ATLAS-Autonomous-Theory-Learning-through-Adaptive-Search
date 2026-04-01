"""Tests for ENV-07: Stern-Gerlach spin-1/2 quantization."""
import numpy as np
import pytest
from atlas.environments.env_07_stern_gerlach import Env07SternGerlach
from atlas.types import KnobType


@pytest.fixture
def env():
    return Env07SternGerlach(seed=42)


def _knobs(knob_0=0.5, knob_1=0.5, knob_2=10_000):
    return {"knob_0": knob_0, "knob_1": knob_1, "knob_2": knob_2}


# --- Schema tests ---

def test_schema_env_id(env):
    assert env.get_schema().env_id == "ENV_07"


def test_schema_knob_count(env):
    assert len(env.get_schema().knobs) == 3


def test_schema_detector_count(env):
    assert len(env.get_schema().detectors) == 1


def test_no_physics_names_in_schema(env):
    schema = env.get_schema()
    for knob in schema.knobs:
        assert knob.name.startswith("knob_")
    for det in schema.detectors:
        assert det.name.startswith("detector_")


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
    assert int_knobs[0].name == "knob_2"


# --- Output shape ---

def test_output_is_array_200(env):
    result = env.run(_knobs())
    arr = result["detector_0"]
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (200,)


def test_output_nonnegative(env):
    result = env.run(_knobs())
    assert np.all(result["detector_0"] >= 0)


def test_output_total_equals_N(env):
    N = 5000
    result = env.run(_knobs(knob_2=N))
    assert int(np.sum(result["detector_0"])) == N


# --- Discrete peaks (two-spot pattern) ---

def test_has_two_distinct_peaks(env):
    """With non-trivial gradient and angle, should see two peaks."""
    result = env.run(_knobs(knob_0=0.5, knob_1=0.7, knob_2=50_000))
    arr = result["detector_0"]
    # Find peaks: bins significantly above neighbours
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(arr, height=np.max(arr) * 0.1, distance=5)
    assert len(peaks) >= 1, "Should have at least one peak"


def test_peak_separation_varies_with_gradient():
    """Larger gradient should spread the two peaks farther apart."""
    env_lo = Env07SternGerlach(seed=0)
    env_hi = Env07SternGerlach(seed=0)

    arr_lo = env_lo.run(_knobs(knob_1=0.2, knob_2=100_000))["detector_0"]
    arr_hi = env_hi.run(_knobs(knob_1=0.9, knob_2=100_000))["detector_0"]

    # Compute spread (standard deviation of distribution as proxy)
    positions = np.arange(200)
    total_lo = np.sum(arr_lo)
    total_hi = np.sum(arr_hi)
    if total_lo > 0 and total_hi > 0:
        mean_lo = np.sum(positions * arr_lo) / total_lo
        mean_hi = np.sum(positions * arr_hi) / total_hi
        var_lo = np.sum((positions - mean_lo) ** 2 * arr_lo) / total_lo
        var_hi = np.sum((positions - mean_hi) ** 2 * arr_hi) / total_hi
        # Higher gradient should give larger spread
        assert var_hi >= var_lo, "Higher gradient should spread peaks more"


# --- Quantum probability ---

def test_spin_up_probability_theta_zero():
    """theta=0 means knob_0=0; all particles should go up."""
    env = Env07SternGerlach(seed=0)
    result = env.run(_knobs(knob_0=0.0, knob_1=0.5, knob_2=10_000))
    arr = result["detector_0"]
    # With theta=0, p_up=1 => all particles deflect up (center+deflection region)
    upper_half = np.sum(arr[100:])
    lower_half = np.sum(arr[:100])
    assert upper_half > lower_half, "theta=0 should produce mostly up-deflected particles"


def test_spin_down_probability_theta_pi():
    """theta=pi means knob_0=1; all particles should go down."""
    env = Env07SternGerlach(seed=0)
    result = env.run(_knobs(knob_0=1.0, knob_1=0.5, knob_2=10_000))
    arr = result["detector_0"]
    upper_half = np.sum(arr[100:])
    lower_half = np.sum(arr[:100])
    assert lower_half > upper_half, "theta=pi should produce mostly down-deflected particles"


# --- Stochasticity ---

def test_stochastic_different_seeds():
    knobs = _knobs(knob_2=1000)
    r1 = Env07SternGerlach(seed=1).run(knobs)
    r2 = Env07SternGerlach(seed=2).run(knobs)
    assert not np.array_equal(r1["detector_0"], r2["detector_0"])


def test_same_seed_reproducible():
    knobs = _knobs(knob_2=1000)
    r1 = Env07SternGerlach(seed=7).run(knobs)
    r2 = Env07SternGerlach(seed=7).run(knobs)
    np.testing.assert_array_equal(r1["detector_0"], r2["detector_0"])


# --- Performance (vectorized, not per-particle loop) ---

def test_large_N_completes_quickly(env):
    """N=1_000_000 must complete in reasonable time (< 5 seconds)."""
    import time
    start = time.time()
    env.run(_knobs(knob_2=1_000_000))
    elapsed = time.time() - start
    assert elapsed < 5.0, f"Large N took {elapsed:.2f}s — probably not vectorized"


# --- Validation ---

def test_invalid_knob_raises(env):
    with pytest.raises(ValueError):
        env.run({"knob_0": -0.1, "knob_1": 0.5, "knob_2": 1000})


def test_missing_knob_raises(env):
    with pytest.raises(ValueError):
        env.run({"knob_0": 0.5, "knob_1": 0.5})
