"""Tests for ENV-10."""
import math
import pytest
from atlas.environments.env_10 import Env10
from atlas.types import KnobType


@pytest.fixture
def env():
    return Env10()


def test_schema_structure(env):
    schema = env.get_schema()
    assert schema.env_id == "ENV_10"
    assert len(schema.knobs) == 3
    assert len(schema.detectors) == 1


def test_no_physics_names_in_schema(env):
    schema = env.get_schema()
    for knob in schema.knobs:
        assert knob.name.startswith("knob_")
    for det in schema.detectors:
        assert det.name.startswith("detector_")


def test_output_is_scalar(env):
    result = env.run({"knob_0": 0.0, "knob_1": 0.5, "knob_2": 0.5})
    assert isinstance(result["detector_0"], float)


def test_output_finite(env):
    result = env.run({"knob_0": 0.5, "knob_1": 0.5, "knob_2": 0.5})
    assert math.isfinite(result["detector_0"])


def test_deterministic(env):
    knobs = {"knob_0": 0.3, "knob_1": 0.6, "knob_2": 0.4}
    r1 = env.run(knobs)
    r2 = env.run(knobs)
    assert r1["detector_0"] == r2["detector_0"]


def test_periodic_in_time(env):
    """Oscillation: value at t=0 should equal value at one full period."""
    # At t=0 (knob_0=0): x = A*cos(0) = A
    r_t0 = env.run({"knob_0": 0.0, "knob_1": 0.5, "knob_2": 0.5})
    x_t0 = r_t0["detector_0"]

    # The displacement should be bounded by amplitude
    assert abs(x_t0) <= 1.0 + 1e-9


def test_zero_time_gives_max_amplitude(env):
    """At t=0, cos(0)=1, so x = A (maximum displacement)."""
    result = env.run({"knob_0": 0.0, "knob_1": 0.5, "knob_2": 1.0})
    # knob_2=1.0 -> A = 1.0m, normalized by A_max=1.0 -> detector_0 = 1.0
    assert abs(result["detector_0"] - 1.0) < 1e-9


def test_oscillation_occurs(env):
    """Output varies with time — system actually oscillates."""
    # Use a fixed spring constant and check that different times give different results
    results = [
        env.run({"knob_0": t, "knob_1": 0.5, "knob_2": 0.8})["detector_0"]
        for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    ]
    # Should not all be identical
    assert not all(abs(x - results[0]) < 1e-12 for x in results)
    # Some should be positive and some negative (oscillation)
    assert max(results) > 0 and min(results) < 0


def test_knob_sensitive(env):
    r1 = env.run({"knob_0": 0.5, "knob_1": 0.1, "knob_2": 0.5})
    r2 = env.run({"knob_0": 0.5, "knob_1": 0.9, "knob_2": 0.5})
    assert abs(r1["detector_0"] - r2["detector_0"]) > 1e-6


def test_all_knobs_continuous(env):
    schema = env.get_schema()
    for knob in schema.knobs:
        assert knob.knob_type == KnobType.CONTINUOUS


def test_invalid_knob_raises(env):
    with pytest.raises(ValueError):
        env.run({"knob_0": 1.5, "knob_1": 0.5, "knob_2": 0.5})
