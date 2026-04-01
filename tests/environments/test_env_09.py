"""Tests for ENV-09: classical 1D elastic collision."""
import math
import pytest
from atlas.environments.env_09_elastic_collision import Env09ElasticCollision
from atlas.types import KnobType


@pytest.fixture
def env():
    return Env09ElasticCollision()


def test_schema_structure(env):
    schema = env.get_schema()
    assert schema.env_id == "ENV_09"
    assert len(schema.knobs) == 3
    assert len(schema.detectors) == 2


def test_no_physics_names_in_schema(env):
    schema = env.get_schema()
    for knob in schema.knobs:
        assert knob.name.startswith("knob_")
    for det in schema.detectors:
        assert det.name.startswith("detector_")


def test_two_detectors_present(env):
    result = env.run({"knob_0": 0.5, "knob_1": 0.5, "knob_2": 0.0})
    assert "detector_0" in result
    assert "detector_1" in result


def test_outputs_are_scalars(env):
    result = env.run({"knob_0": 0.5, "knob_1": 0.5, "knob_2": 0.0})
    assert isinstance(result["detector_0"], float)
    assert isinstance(result["detector_1"], float)


def test_outputs_finite(env):
    result = env.run({"knob_0": 0.5, "knob_1": 0.7, "knob_2": 0.2})
    assert math.isfinite(result["detector_0"])
    assert math.isfinite(result["detector_1"])


def test_deterministic(env):
    knobs = {"knob_0": 0.3, "knob_1": 0.6, "knob_2": 0.4}
    r1 = env.run(knobs)
    r2 = env.run(knobs)
    assert r1["detector_0"] == r2["detector_0"]
    assert r1["detector_1"] == r2["detector_1"]


def test_equal_mass_collision_exchange_velocities(env):
    """Equal masses (knob_0=1.0 -> ratio=1): velocities exchange."""
    # v1_i=10, v2_i=-10 (full range): knob_1=1.0, knob_2=0.0
    result = env.run({"knob_0": 1.0, "knob_1": 1.0, "knob_2": 0.0})
    # m1=m2=1: v1_f = 0*v1 + 2*m2*v2/(2m) = v2_i, v2_f = v1_i
    # v1_i=10, v2_i=-10 -> v1_f=-10/10=-1.0, v2_f=10/10=1.0
    assert abs(result["detector_0"] - (-1.0)) < 1e-9
    assert abs(result["detector_1"] - 1.0) < 1e-9


def test_momentum_conserved(env):
    """Total momentum before = total momentum after."""
    knob_0 = 0.4
    knob_1 = 0.6
    knob_2 = 0.3

    # Reconstruct physical values inline
    from atlas.environments.normalizer import denormalize
    mass_ratio = denormalize(knob_0, 0.01, 1.0)
    v1_i = denormalize(knob_1, -10.0, 10.0)
    v2_i = denormalize(knob_2, -10.0, 10.0)
    m1 = 1.0
    m2 = mass_ratio

    result = env.run({"knob_0": knob_0, "knob_1": knob_1, "knob_2": knob_2})
    v1_f = result["detector_0"] * 10.0
    v2_f = result["detector_1"] * 10.0

    p_before = m1 * v1_i + m2 * v2_i
    p_after = m1 * v1_f + m2 * v2_f
    assert abs(p_before - p_after) < 1e-9


def test_all_knobs_continuous(env):
    schema = env.get_schema()
    for knob in schema.knobs:
        assert knob.knob_type == KnobType.CONTINUOUS


def test_invalid_knob_raises(env):
    with pytest.raises(ValueError):
        env.run({"knob_0": -0.1, "knob_1": 0.5, "knob_2": 0.5})
