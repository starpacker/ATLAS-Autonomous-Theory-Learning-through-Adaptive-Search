"""Tests for ENV-02."""
from __future__ import annotations

import pytest
import numpy as np

from atlas.environments.env_02 import Env02
from atlas.types import KnobType


@pytest.fixture
def env():
    return Env02()


def test_schema_env_id(env):
    assert env.get_schema().env_id == "ENV_02"


def test_schema_knob_count(env):
    assert len(env.get_schema().knobs) == 2


def test_schema_has_two_detectors(env):
    assert len(env.get_schema().detectors) == 2


def test_detector_names(env):
    schema = env.get_schema()
    names = {d.name for d in schema.detectors}
    assert "detector_0" in names
    assert "detector_1" in names


def test_schema_has_entity(env):
    assert "entity_A" in env.get_schema().entities


def test_no_physics_names_in_schema(env):
    schema = env.get_schema()
    for knob in schema.knobs:
        assert knob.name.startswith("knob_")
    for det in schema.detectors:
        assert det.name.startswith("detector_")


def test_outputs_are_scalar(env):
    result = env.run({"knob_0": 0.5, "knob_1": 0.5})
    assert isinstance(result["detector_0"], float)
    assert isinstance(result["detector_1"], float)


def test_zero_angle_near_zero_shift(env):
    """At zero scattering angle, wavelength shift should be approximately zero."""
    result = env.run({"knob_0": 0.5, "knob_1": 0.0})
    assert result["detector_0"] < 0.01, f"Near-zero shift expected, got {result['detector_0']}"


def test_max_shift_at_pi(env):
    """At scattering angle=pi, wavelength shift is maximized (normed to 1.0)."""
    result = env.run({"knob_0": 0.5, "knob_1": 1.0})
    assert abs(result["detector_0"] - 1.0) < 0.01, (
        f"Max shift expected ~1.0, got {result['detector_0']}"
    )


def test_shift_monotonically_increasing_with_angle(env):
    """Wavelength shift should increase as scattering angle increases from 0 to pi."""
    shifts = [
        env.run({"knob_0": 0.5, "knob_1": float(k1)})["detector_0"]
        for k1 in np.linspace(0.0, 1.0, 10)
    ]
    for i in range(len(shifts) - 1):
        assert shifts[i] <= shifts[i + 1] + 1e-10, (
            f"Shift not monotonic: {shifts[i]} > {shifts[i+1]}"
        )


def test_intensity_at_zero_angle(env):
    """At zero angle, Klein-Nishina factor is maximum (1.0)."""
    result = env.run({"knob_0": 0.5, "knob_1": 0.0})
    assert abs(result["detector_1"] - 1.0) < 0.01, (
        f"Intensity at zero angle expected ~1.0, got {result['detector_1']}"
    )


def test_intensity_at_half_pi(env):
    """At 90 degrees, Klein-Nishina factor is 0.5."""
    result = env.run({"knob_0": 0.5, "knob_1": 0.5})
    assert abs(result["detector_1"] - 0.5) < 0.05, (
        f"Intensity at pi/2 expected ~0.5, got {result['detector_1']}"
    )


def test_output_in_valid_range(env):
    for k0 in np.linspace(0, 1, 5):
        for k1 in np.linspace(0, 1, 5):
            result = env.run({"knob_0": float(k0), "knob_1": float(k1)})
            assert 0.0 <= result["detector_0"] <= 1.0 + 1e-9
            assert 0.0 <= result["detector_1"] <= 1.0 + 1e-9


def test_shift_independent_of_incident_wavelength(env):
    """Shift depends only on angle, not incident wavelength."""
    r1 = env.run({"knob_0": 0.1, "knob_1": 0.5})
    r2 = env.run({"knob_0": 0.9, "knob_1": 0.5})
    assert abs(r1["detector_0"] - r2["detector_0"]) < 1e-10, (
        "Wavelength shift should not depend on incident wavelength"
    )


def test_deterministic(env):
    knobs = {"knob_0": 0.4, "knob_1": 0.6}
    r1 = env.run(knobs)
    r2 = env.run(knobs)
    assert r1["detector_0"] == r2["detector_0"]
    assert r1["detector_1"] == r2["detector_1"]


def test_continuous_knobs_normalized(env):
    schema = env.get_schema()
    for knob in schema.knobs:
        assert knob.knob_type == KnobType.CONTINUOUS
        assert knob.range_min == 0.0
        assert knob.range_max == 1.0


def test_invalid_knob_raises(env):
    with pytest.raises(ValueError):
        env.run({"knob_0": 1.5, "knob_1": 0.5})
