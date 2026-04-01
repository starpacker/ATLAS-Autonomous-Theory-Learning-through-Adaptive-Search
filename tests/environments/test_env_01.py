"""Tests for ENV-01: photoelectric effect."""
from __future__ import annotations

import pytest
import numpy as np

from atlas.environments.env_01_photoelectric import Env01Photoelectric
from atlas.types import KnobType


@pytest.fixture
def env():
    return Env01Photoelectric()


def test_schema_env_id(env):
    assert env.get_schema().env_id == "ENV_01"


def test_schema_knob_count(env):
    assert len(env.get_schema().knobs) == 4


def test_schema_detector_count(env):
    assert len(env.get_schema().detectors) == 1


def test_schema_has_entity(env):
    assert "entity_A" in env.get_schema().entities


def test_no_physics_names_in_schema(env):
    schema = env.get_schema()
    for knob in schema.knobs:
        assert knob.name.startswith("knob_"), f"Knob '{knob.name}' leaks physics names"
    for det in schema.detectors:
        assert det.name.startswith("detector_"), f"Detector '{det.name}' leaks physics names"


def test_output_is_scalar(env):
    result = env.run({"knob_0": 1.0, "knob_1": 1.0, "knob_2": 0, "knob_3": 1.0})
    assert isinstance(result["detector_0"], float)


def test_output_in_valid_range(env):
    """Output must be clipped to [0, 1]."""
    for k0 in np.linspace(0, 1, 10):
        result = env.run({"knob_0": float(k0), "knob_1": 1.0, "knob_2": 0, "knob_3": 0.0})
        val = result["detector_0"]
        assert 0.0 <= val <= 1.0, f"Out of range at knob_0={k0}: {val}"


def test_cutoff_behavior(env):
    """Below the threshold frequency (material 3 has highest work function), current=0."""
    # Material 3 has work function 5.1 eV, threshold ~1.23e15 Hz
    # knob_0 at low values (low frequency) should produce zero current
    low_freq_result = env.run({"knob_0": 0.05, "knob_1": 1.0, "knob_2": 3, "knob_3": 0.0})
    assert low_freq_result["detector_0"] == 0.0, "Below cutoff frequency should give zero current"

    # High frequency should produce nonzero current
    high_freq_result = env.run({"knob_0": 1.0, "knob_1": 1.0, "knob_2": 0, "knob_3": 0.0})
    assert high_freq_result["detector_0"] > 0.0, "Above cutoff frequency should give nonzero current"


def test_frequency_sweep_has_zero_and_nonzero(env):
    """Sweep knob_0 to confirm the cutoff effect: some zero, some nonzero."""
    results = [
        env.run({"knob_0": float(k0), "knob_1": 1.0, "knob_2": 1, "knob_3": 0.0})["detector_0"]
        for k0 in np.linspace(0.0, 1.0, 20)
    ]
    has_zero = any(r == 0.0 for r in results)
    has_nonzero = any(r > 0.0 for r in results)
    assert has_zero, "Expected some zero readings below threshold"
    assert has_nonzero, "Expected some nonzero readings above threshold"


def test_zero_intensity_gives_zero_current(env):
    """Zero intensity means no incoming photons, so current = 0."""
    result = env.run({"knob_0": 1.0, "knob_1": 0.0, "knob_2": 0, "knob_3": 0.0})
    assert result["detector_0"] == 0.0


def test_negative_voltage_reduces_current(env):
    """Negative voltage should reduce or eliminate current."""
    pos = env.run({"knob_0": 0.5, "knob_1": 1.0, "knob_2": 0, "knob_3": 1.0})["detector_0"]
    neg = env.run({"knob_0": 0.5, "knob_1": 1.0, "knob_2": 0, "knob_3": -1.0})["detector_0"]
    assert pos >= neg, "Positive voltage should give >= current than negative voltage"


def test_deterministic(env):
    knobs = {"knob_0": 0.7, "knob_1": 0.8, "knob_2": 1, "knob_3": 0.5}
    r1 = env.run(knobs)
    r2 = env.run(knobs)
    assert r1["detector_0"] == r2["detector_0"]


def test_continuous_knobs_normalized(env):
    schema = env.get_schema()
    for knob in schema.knobs:
        if knob.knob_type == KnobType.CONTINUOUS:
            assert knob.range_min >= -1.0
            assert knob.range_max <= 1.0


def test_discrete_knob_options(env):
    schema = env.get_schema()
    discrete = [k for k in schema.knobs if k.knob_type == KnobType.DISCRETE]
    assert len(discrete) == 1
    assert discrete[0].options == [0, 1, 2, 3]


def test_invalid_knob_value_raises(env):
    with pytest.raises(ValueError):
        env.run({"knob_0": 1.5, "knob_1": 0.5, "knob_2": 0, "knob_3": 0.0})


def test_invalid_discrete_raises(env):
    with pytest.raises(ValueError):
        env.run({"knob_0": 0.5, "knob_1": 0.5, "knob_2": 5, "knob_3": 0.0})
