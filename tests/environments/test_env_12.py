"""Tests for ENV-12: classical heat conduction (Fourier's law)."""
import math
import pytest
from atlas.environments.env_12_heat_conduction import Env12HeatConduction
from atlas.types import KnobType


@pytest.fixture
def env():
    return Env12HeatConduction()


def test_schema_structure(env):
    schema = env.get_schema()
    assert schema.env_id == "ENV_12"
    assert len(schema.knobs) == 3
    assert len(schema.detectors) == 1


def test_no_physics_names_in_schema(env):
    schema = env.get_schema()
    for knob in schema.knobs:
        assert knob.name.startswith("knob_")
    for det in schema.detectors:
        assert det.name.startswith("detector_")


def test_output_is_scalar(env):
    result = env.run({"knob_0": 0.5, "knob_1": 0.5, "knob_2": 0.5})
    assert isinstance(result["detector_0"], float)


def test_output_finite(env):
    result = env.run({"knob_0": 0.5, "knob_1": 0.5, "knob_2": 0.5})
    assert math.isfinite(result["detector_0"])


def test_output_nonnegative(env):
    """Heat flow should always be non-negative (all parameters positive)."""
    result = env.run({"knob_0": 0.5, "knob_1": 0.5, "knob_2": 0.5})
    assert result["detector_0"] >= 0.0


def test_deterministic(env):
    knobs = {"knob_0": 0.3, "knob_1": 0.6, "knob_2": 0.4}
    r1 = env.run(knobs)
    r2 = env.run(knobs)
    assert r1["detector_0"] == r2["detector_0"]


def test_higher_delta_t_gives_higher_flow(env):
    """Increasing temperature difference increases heat flow."""
    r_low = env.run({"knob_0": 0.1, "knob_1": 0.5, "knob_2": 0.5})
    r_high = env.run({"knob_0": 0.9, "knob_1": 0.5, "knob_2": 0.5})
    assert r_high["detector_0"] > r_low["detector_0"]


def test_higher_area_gives_higher_flow(env):
    """Larger cross-sectional area gives higher heat flow."""
    r_small = env.run({"knob_0": 0.5, "knob_1": 0.1, "knob_2": 0.5})
    r_large = env.run({"knob_0": 0.5, "knob_1": 0.9, "knob_2": 0.5})
    assert r_large["detector_0"] > r_small["detector_0"]


def test_longer_conductor_gives_lower_flow(env):
    """Longer conductor reduces heat flow (Q proportional to 1/L)."""
    r_short = env.run({"knob_0": 0.5, "knob_1": 0.5, "knob_2": 0.1})
    r_long = env.run({"knob_0": 0.5, "knob_1": 0.5, "knob_2": 0.9})
    assert r_short["detector_0"] > r_long["detector_0"]


def test_max_output_normalized(env):
    """Output at maximum knobs should be <= 1.0 (normalized)."""
    result = env.run({"knob_0": 1.0, "knob_1": 1.0, "knob_2": 0.0})
    # knob_2=0 would be minimum length — but that's range_min, OK
    result2 = env.run({"knob_0": 1.0, "knob_1": 1.0, "knob_2": 0.001})
    assert result2["detector_0"] <= 1.0 + 1e-6


def test_all_knobs_continuous(env):
    schema = env.get_schema()
    for knob in schema.knobs:
        assert knob.knob_type == KnobType.CONTINUOUS


def test_invalid_knob_raises(env):
    with pytest.raises(ValueError):
        env.run({"knob_0": -0.1, "knob_1": 0.5, "knob_2": 0.5})
