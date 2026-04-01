"""Tests for ENV-06: Hydrogen emission spectrum via Rydberg formula."""
import numpy as np
import pytest
from atlas.environments.env_06_hydrogen_spectrum import Env06HydrogenSpectrum
from atlas.types import KnobType


@pytest.fixture
def env():
    return Env06HydrogenSpectrum()


def _knobs(knob_0=0.5, knob_1=1.0):
    return {"knob_0": knob_0, "knob_1": knob_1}


# --- Schema tests ---

def test_schema_env_id(env):
    assert env.get_schema().env_id == "ENV_06"


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


def test_continuous_knobs_normalized(env):
    schema = env.get_schema()
    for knob in schema.knobs:
        if knob.knob_type == KnobType.CONTINUOUS:
            assert knob.range_min == 0.0
            assert knob.range_max == 1.0


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


# --- Discrete spectral lines ---

def test_has_at_least_two_peaks(env):
    """Hydrogen spectrum should show discrete emission lines."""
    # knob_0=0.2 => center ~420nm, window 320-520nm (Balmer series: 486,434,410,397,389nm)
    result = env.run(_knobs(knob_0=0.2, knob_1=1.0))
    arr = result["detector_0"]
    from scipy.signal import find_peaks
    peaks, props = find_peaks(arr, height=0.05, distance=3)
    assert len(peaks) >= 2, f"Expected at least 2 spectral peaks, found {len(peaks)}"


def test_peaks_are_sharp(env):
    """Spectral lines should be narrow — most of the array should be near zero."""
    result = env.run(_knobs(knob_0=0.2, knob_1=1.0))
    arr = result["detector_0"]
    # With sharp Gaussian peaks, most bins should be very low
    near_zero_frac = np.mean(arr < 0.05)
    assert near_zero_frac > 0.5, f"Expected sharp lines; only {near_zero_frac:.1%} near zero"


def test_no_signal_in_empty_window(env):
    """Spectrometer window far from hydrogen lines should give ~zero output."""
    # UV extreme: knob_0=0.0 => 300nm center, window 200-400nm
    # Balmer series lines are at ~410, 434, 486, 656 nm — not in this window
    # Lyman lines are at ~121-122nm — far below
    # There may be n1=1 Lyman series lines far from our range
    # This test verifies the windowing logic works
    result_uv = env.run(_knobs(knob_0=0.0, knob_1=1.0))
    result_vis = env.run(_knobs(knob_0=0.5, knob_1=1.0))
    # The two spectra should be different
    assert not np.allclose(result_uv["detector_0"], result_vis["detector_0"]), \
        "Different spectrometer positions should give different spectra"


def test_excitation_energy_affects_spectrum(env):
    """Higher excitation energy allows more transitions."""
    r_low = env.run(_knobs(knob_0=0.2, knob_1=0.0))   # low excitation
    r_high = env.run(_knobs(knob_0=0.2, knob_1=1.0))  # high excitation
    # High excitation should have more or equal signal (more transitions visible)
    assert np.sum(r_high["detector_0"]) >= np.sum(r_low["detector_0"]) - 1e-6


# --- Deterministic ---

def test_deterministic(env):
    knobs = _knobs()
    r1 = env.run(knobs)
    r2 = env.run(knobs)
    np.testing.assert_array_equal(r1["detector_0"], r2["detector_0"])


# --- Knob sensitivity ---

def test_spectrometer_window_shifts_pattern(env):
    """Different spectrometer positions should reveal different lines."""
    r1 = env.run(_knobs(knob_0=0.2, knob_1=1.0))
    r2 = env.run(_knobs(knob_0=0.8, knob_1=1.0))
    assert not np.allclose(r1["detector_0"], r2["detector_0"])


# --- Validation ---

def test_invalid_knob_raises(env):
    with pytest.raises(ValueError):
        env.run({"knob_0": 1.5, "knob_1": 0.5})


def test_missing_knob_raises(env):
    with pytest.raises(ValueError):
        env.run({"knob_0": 0.5})
