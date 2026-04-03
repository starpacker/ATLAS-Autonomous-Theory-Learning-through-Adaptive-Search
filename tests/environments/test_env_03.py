"""Tests for ENV-03."""
import numpy as np
import pytest
from atlas.environments.env_03 import Env03
from atlas.types import KnobType


@pytest.fixture
def env():
    return Env03()


def _knobs(knob_0=0.5, knob_1=0.5, knob_2=0):
    return {"knob_0": knob_0, "knob_1": knob_1, "knob_2": knob_2}


# --- Schema tests ---

def test_schema_env_id(env):
    assert env.get_schema().env_id == "ENV_03"


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


def test_entity_label(env):
    assert "entity_B" in env.get_schema().entities


def test_continuous_knobs_normalized(env):
    schema = env.get_schema()
    for knob in schema.knobs:
        if knob.knob_type == KnobType.CONTINUOUS:
            assert knob.range_min == 0.0
            assert knob.range_max == 1.0


def test_discrete_knob_options(env):
    schema = env.get_schema()
    disc_knobs = [k for k in schema.knobs if k.knob_type == KnobType.DISCRETE]
    assert len(disc_knobs) == 1
    assert set(disc_knobs[0].options) == {0, 1, 2}


# --- Output shape ---

def test_output_is_array_500(env):
    result = env.run(_knobs())
    arr = result["detector_0"]
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (500,)


def test_output_range(env):
    result = env.run(_knobs())
    arr = result["detector_0"]
    assert np.min(arr) >= -1e-9
    assert np.max(arr) <= 1.0 + 1e-9


# --- Diffraction structure ---

def test_has_diffraction_peaks(env):
    """Output should not be flat — peaks create structure."""
    result = env.run(_knobs())
    arr = result["detector_0"]
    # Nonzero std means structure exists
    assert np.std(arr) > 0.01, "Pattern should have diffraction peaks, not flat"


def test_output_not_flat(env):
    """Max/min ratio should be large enough to show peaks."""
    result = env.run(_knobs())
    arr = result["detector_0"]
    assert np.max(arr) > 3 * np.mean(arr), "Expected distinct peaks above background"


# --- Deterministic ---

def test_deterministic(env):
    knobs = _knobs()
    r1 = env.run(knobs)
    r2 = env.run(knobs)
    np.testing.assert_array_equal(r1["detector_0"], r2["detector_0"])


# --- Knob sensitivity ---

def test_voltage_changes_pattern(env):
    """Higher voltage => different peak positions."""
    r1 = env.run(_knobs(knob_0=0.1))
    r2 = env.run(_knobs(knob_0=0.9))
    assert not np.allclose(r1["detector_0"], r2["detector_0"])


def test_crystal_type_changes_pattern(env):
    """Different crystal types should give different diffraction patterns."""
    r0 = env.run(_knobs(knob_2=0))
    r1 = env.run(_knobs(knob_2=1))
    r2 = env.run(_knobs(knob_2=2))
    assert not np.allclose(r0["detector_0"], r1["detector_0"])
    assert not np.allclose(r1["detector_0"], r2["detector_0"])


def test_all_crystal_types_runnable(env):
    for crystal in [0, 1, 2]:
        result = env.run(_knobs(knob_2=crystal))
        assert result["detector_0"].shape == (500,)


# --- Validation ---

def test_invalid_knob_raises(env):
    with pytest.raises(ValueError):
        env.run({"knob_0": 1.5, "knob_1": 0.5, "knob_2": 0})


def test_invalid_discrete_raises(env):
    with pytest.raises(ValueError):
        env.run({"knob_0": 0.5, "knob_1": 0.5, "knob_2": 5})


def test_missing_knob_raises(env):
    with pytest.raises(ValueError):
        env.run({"knob_0": 0.5, "knob_1": 0.5})
