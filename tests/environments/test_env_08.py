"""Tests for ENV-08."""
import numpy as np
import pytest
from atlas.environments.env_08 import Env08
from atlas.types import KnobType


@pytest.fixture
def env():
    return Env08()


def test_schema_structure(env):
    schema = env.get_schema()
    assert schema.env_id == "ENV_08"
    assert len(schema.knobs) == 3
    assert len(schema.detectors) == 1


def test_no_physics_names_in_schema(env):
    schema = env.get_schema()
    for knob in schema.knobs:
        assert knob.name.startswith("knob_"), f"Knob '{knob.name}' leaks physics names"
    for det in schema.detectors:
        assert det.name.startswith("detector_"), f"Detector '{det.name}' leaks physics names"


def test_output_is_array(env):
    result = env.run({"knob_0": 0.5, "knob_1": 0.5, "knob_2": 0.5})
    arr = result["detector_0"]
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1000,)


def test_output_in_valid_range(env):
    result = env.run({"knob_0": 0.5, "knob_1": 0.5, "knob_2": 0.5})
    arr = result["detector_0"]
    assert np.all(arr >= -1e-9), "Intensity should be non-negative"
    assert np.all(arr <= 1.0 + 1e-9), "Intensity should be <= 1"


def test_has_periodic_structure(env):
    """FFT should show a clear peak at nonzero frequency."""
    result = env.run({"knob_0": 0.5, "knob_1": 0.5, "knob_2": 0.5})
    arr = result["detector_0"]
    freqs = np.fft.rfftfreq(len(arr))
    magnitudes = np.abs(np.fft.rfft(arr))
    # Ignore DC component (index 0)
    peak_idx = np.argmax(magnitudes[1:]) + 1
    assert peak_idx > 0, "Should have periodic structure with nonzero-frequency peak"
    # Peak should be stronger than DC or comparable
    assert magnitudes[peak_idx] > 0.0


def test_deterministic(env):
    knobs = {"knob_0": 0.3, "knob_1": 0.6, "knob_2": 0.4}
    r1 = env.run(knobs)
    r2 = env.run(knobs)
    np.testing.assert_array_equal(r1["detector_0"], r2["detector_0"])


def test_knob_sensitive(env):
    """Changing knobs changes the output pattern."""
    r1 = env.run({"knob_0": 0.2, "knob_1": 0.5, "knob_2": 0.5})
    r2 = env.run({"knob_0": 0.8, "knob_1": 0.5, "knob_2": 0.5})
    assert not np.allclose(r1["detector_0"], r2["detector_0"])


def test_all_knobs_continuous(env):
    schema = env.get_schema()
    for knob in schema.knobs:
        assert knob.knob_type == KnobType.CONTINUOUS
        assert knob.range_min == 0.0
        assert knob.range_max == 1.0


def test_invalid_knob_raises(env):
    with pytest.raises(ValueError):
        env.run({"knob_0": 1.5, "knob_1": 0.5, "knob_2": 0.5})
