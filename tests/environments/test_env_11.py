"""Tests for ENV-11: classical freefall."""
import math
import pytest
from atlas.environments.env_11_freefall import Env11Freefall
from atlas.types import KnobType


@pytest.fixture
def env():
    return Env11Freefall()


def test_schema_structure(env):
    schema = env.get_schema()
    assert schema.env_id == "ENV_11"
    assert len(schema.knobs) == 2
    assert len(schema.detectors) == 1


def test_no_physics_names_in_schema(env):
    schema = env.get_schema()
    for knob in schema.knobs:
        assert knob.name.startswith("knob_")
    for det in schema.detectors:
        assert det.name.startswith("detector_")


def test_output_is_scalar(env):
    result = env.run({"knob_0": 0.0, "knob_1": 0.5})
    assert isinstance(result["detector_0"], float)


def test_output_finite(env):
    result = env.run({"knob_0": 0.5, "knob_1": 0.5})
    assert math.isfinite(result["detector_0"])


def test_deterministic(env):
    knobs = {"knob_0": 0.3, "knob_1": 0.6}
    r1 = env.run(knobs)
    r2 = env.run(knobs)
    assert r1["detector_0"] == r2["detector_0"]


def test_at_t0_position_is_zero(env):
    """At t=0, position is always 0 regardless of v0."""
    result = env.run({"knob_0": 0.0, "knob_1": 0.8})
    assert abs(result["detector_0"]) < 1e-12


def test_position_increases_initially_with_nonzero_v0(env):
    """With positive initial velocity, position should increase at early times."""
    r_early = env.run({"knob_0": 0.05, "knob_1": 0.5})  # small t, medium v0
    r_later = env.run({"knob_0": 0.0, "knob_1": 0.5})   # t=0
    # At small t with v0>0, y > 0
    assert r_early["detector_0"] > r_later["detector_0"]


def test_higher_velocity_gives_higher_peak(env):
    """Higher v0 leads to higher peak height."""
    r_low = env.run({"knob_0": 0.3, "knob_1": 0.2})
    r_high = env.run({"knob_0": 0.3, "knob_1": 0.8})
    # Both at same early time, higher v0 -> higher position
    assert r_high["detector_0"] > r_low["detector_0"]


def test_zero_velocity_only_falls(env):
    """With v0=0, object falls (position decreases over time)."""
    r0 = env.run({"knob_0": 0.0, "knob_1": 0.0})    # t=0, y=0
    r1 = env.run({"knob_0": 0.1, "knob_1": 0.0})    # t=1s, y<0
    assert r1["detector_0"] < r0["detector_0"]


def test_all_knobs_continuous(env):
    schema = env.get_schema()
    for knob in schema.knobs:
        assert knob.knob_type == KnobType.CONTINUOUS


def test_invalid_knob_raises(env):
    with pytest.raises(ValueError):
        env.run({"knob_0": -0.1, "knob_1": 0.5})
