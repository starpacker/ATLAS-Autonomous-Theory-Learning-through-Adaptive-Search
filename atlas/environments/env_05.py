"""ENV-05 experiment environment."""
from __future__ import annotations

import numpy as np

from atlas.environments.base import BaseEnvironment
from atlas.environments.normalizer import denormalize
from atlas.environments.registry import register
from atlas.types import KnobSpec, KnobType, DetectorSpec

# Internal constants — private, never exposed through interface
_H = 6.626e-34
_C = 2.998e8
_K_B = 1.381e-23

_FREQ_MIN = 1e12
_FREQ_MAX = 3e15
_TEMP_MIN = 300.0
_TEMP_MAX = 10000.0

_REF_FREQ = 3.0e14
_REF_TEMP = 5000.0


def _radiation_fn(freq: float, temp: float) -> float:
    """Internal radiation function."""
    x = _H * freq / (_K_B * temp)
    if x > 500.0:
        return 0.0
    numerator = 2.0 * _H * freq ** 3 / (_C ** 2)
    denominator = np.exp(x) - 1.0
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


# Compute reference value for normalization
_NORM_REF = _radiation_fn(_REF_FREQ, _REF_TEMP)
if _NORM_REF == 0.0:
    _NORM_REF = 1.0  # fallback to avoid division by zero


@register
class Env05(BaseEnvironment):

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

        B = _radiation_fn(freq, temp)
        B_normed = float(B / _NORM_REF)
        # Clip to reasonable positive range; no hard cap at 1 since very high
        # temperatures at low frequencies can exceed the reference
        B_normed = float(np.clip(B_normed, 0.0, np.inf))

        return {"detector_0": B_normed}
