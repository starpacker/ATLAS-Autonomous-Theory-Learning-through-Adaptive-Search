"""ENV-05: Blackbody spectral radiance (Planck distribution)."""
from __future__ import annotations

import numpy as np

from atlas.environments.base import BaseEnvironment
from atlas.environments.normalizer import denormalize
from atlas.environments.registry import register
from atlas.types import KnobSpec, KnobType, DetectorSpec

# Physical constants — private, never exposed through interface
_H = 6.626e-34   # Planck's constant (J·s)
_C = 2.998e8     # Speed of light (m/s)
_K_B = 1.381e-23 # Boltzmann constant (J/K)

_FREQ_MIN = 1e12     # 1 THz
_FREQ_MAX = 3e15     # 3 PHz
_TEMP_MIN = 300.0    # K
_TEMP_MAX = 10000.0  # K

# Normalization reference: peak radiance at ~5000K, ~3e14 Hz
# Compute once at module level to avoid recomputing
_REF_FREQ = 3.0e14   # Hz (reference for normalization)
_REF_TEMP = 5000.0   # K (reference temperature)


def _planck(freq: float, temp: float) -> float:
    """Spectral radiance B(f,T) = 2hf^3/c^2 / (exp(hf/kT) - 1)."""
    x = _H * freq / (_K_B * temp)
    if x > 500.0:
        return 0.0
    numerator = 2.0 * _H * freq ** 3 / (_C ** 2)
    denominator = np.exp(x) - 1.0
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


# Compute reference value for normalization
_NORM_REF = _planck(_REF_FREQ, _REF_TEMP)
if _NORM_REF == 0.0:
    _NORM_REF = 1.0  # fallback to avoid division by zero


@register
class Env05Blackbody(BaseEnvironment):

    @property
    def env_id(self) -> str:
        return "ENV_05"

    @property
    def _knob_specs(self) -> list[KnobSpec]:
        return [
            KnobSpec("knob_0", KnobType.CONTINUOUS, 0.0, 1.0),  # frequency (normalized)
            KnobSpec("knob_1", KnobType.CONTINUOUS, 0.0, 1.0),  # temperature (normalized)
        ]

    @property
    def _detector_specs(self) -> list[DetectorSpec]:
        return [DetectorSpec("detector_0", "scalar")]

    def _compute(self, knobs: dict[str, float]) -> dict[str, float]:
        freq = denormalize(knobs["knob_0"], _FREQ_MIN, _FREQ_MAX)
        temp = denormalize(knobs["knob_1"], _TEMP_MIN, _TEMP_MAX)

        B = _planck(freq, temp)
        B_normed = float(B / _NORM_REF)
        # Clip to reasonable positive range; no hard cap at 1 since very high
        # temperatures at low frequencies can exceed the reference
        B_normed = float(np.clip(B_normed, 0.0, np.inf))

        return {"detector_0": B_normed}
