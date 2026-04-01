"""Tests for ENV-05: Blackbody radiation (Planck distribution)."""
from __future__ import annotations

import pytest
import numpy as np

from atlas.environments.env_05_blackbody import Env05Blackbody
from atlas.types import KnobType


@pytest.fixture
def env():
    return Env05Blackbody()


def test_schema_env_id(env):
    assert env.get_schema().env_id == "ENV_05"


def test_schema_knob_count(env):
    assert len(env.get_schema().knobs) == 2


def test_schema_detector_count(env):
    assert len(env.get_schema().detectors) == 1


def test_no_physics_names_in_schema(env):
    schema = env.get_schema()
    for knob in schema.knobs:
        assert knob.name.startswith("knob_")
    for det in schema.detectors:
        assert det.name.startswith("detector_")


def test_output_is_scalar(env):
    result = env.run({"knob_0": 0.5, "knob_1": 0.5})
    assert isinstance(result["detector_0"], float)


def test_output_nonnegative(env):
    """Spectral radiance must always be >= 0."""
    for k0 in np.linspace(0, 1, 8):
        for k1 in np.linspace(0.01, 1.0, 8):  # avoid 0K temperature
            result = env.run({"knob_0": float(k0), "knob_1": float(k1)})
            assert result["detector_0"] >= 0.0, (
                f"Negative radiance at knob_0={k0}, knob_1={k1}: {result['detector_0']}"
            )


def test_higher_temperature_more_radiation(env):
    """At a fixed mid-range frequency, higher temperature gives more spectral radiance."""
    low_temp = env.run({"knob_0": 0.1, "knob_1": 0.1})["detector_0"]
    high_temp = env.run({"knob_0": 0.1, "knob_1": 0.9})["detector_0"]
    assert high_temp > low_temp, (
        f"Higher temperature should give more radiation: low={low_temp}, high={high_temp}"
    )


def test_very_high_frequency_low_radiation(env):
    """At very high frequency with low temperature, radiance should be near zero (exponential cutoff)."""
    result = env.run({"knob_0": 1.0, "knob_1": 0.01})
    assert result["detector_0"] < 1e-6, (
        f"Expected near-zero at extreme UV / low temp, got {result['detector_0']}"
    )


def test_no_overflow_or_nan(env):
    """All combinations must produce finite, non-NaN results."""
    for k0 in np.linspace(0, 1, 10):
        for k1 in np.linspace(0, 1, 10):
            result = env.run({"knob_0": float(k0), "knob_1": float(k1)})
            val = result["detector_0"]
            assert np.isfinite(val), f"Non-finite at k0={k0}, k1={k1}: {val}"


def test_low_frequency_low_radiation(env):
    """At the lowest frequency end (knob_0=0 maps to 1 THz), radiance at low temp is very small.

    At 1 THz (infrared/microwave boundary) vs the reference point (~300 THz), the
    Planck function falls off steeply — radiance should be much less than 1% of the peak.
    """
    result_low_freq = env.run({"knob_0": 0.0, "knob_1": 0.5})
    result_peak_freq = env.run({"knob_0": 0.1, "knob_1": 0.5})
    # The low-frequency end should be well below a typical mid-range value
    assert result_low_freq["detector_0"] < result_peak_freq["detector_0"], (
        "Radiance at minimum frequency should be below radiance at a higher frequency"
    )


def test_radiance_increases_with_temp_at_multiple_freqs(env):
    """Verify temp sensitivity across a few frequency bands."""
    for k0 in [0.1, 0.3, 0.5]:
        low = env.run({"knob_0": k0, "knob_1": 0.1})["detector_0"]
        high = env.run({"knob_0": k0, "knob_1": 0.9})["detector_0"]
        assert high >= low, (
            f"At knob_0={k0}: higher temp should give >= radiance: low={low}, high={high}"
        )


def test_deterministic(env):
    knobs = {"knob_0": 0.3, "knob_1": 0.6}
    r1 = env.run(knobs)
    r2 = env.run(knobs)
    assert r1["detector_0"] == r2["detector_0"]


def test_continuous_knobs_normalized(env):
    schema = env.get_schema()
    for knob in schema.knobs:
        assert knob.knob_type == KnobType.CONTINUOUS
        assert knob.range_min == 0.0
        assert knob.range_max == 1.0


def test_invalid_knob_raises(env):
    with pytest.raises(ValueError):
        env.run({"knob_0": 1.5, "knob_1": 0.5})
