"""ENV-02 experiment environment."""
from __future__ import annotations

import numpy as np

from atlas.environments.base import BaseEnvironment
from atlas.environments.normalizer import denormalize
from atlas.environments.registry import register
from atlas.types import KnobSpec, KnobType, DetectorSpec

# Internal constants — private, never exposed through interface
_H = 6.626e-34
_M_E = 9.109e-31
_C = 2.998e8
_CHAR_LENGTH = _H / (_M_E * _C)

_WAVELENGTH_MIN = 1e-12
_WAVELENGTH_MAX = 100e-12
_ANGLE_MIN = 0.0
_ANGLE_MAX = np.pi


@register
class Env02(BaseEnvironment):

    @property
    def env_id(self) -> str:
        return "ENV_02"

    @property
    def _knob_specs(self) -> list[KnobSpec]:
        return [
            KnobSpec("knob_0", KnobType.CONTINUOUS, 0.0, 1.0),  # incident wavelength (normalized)
            KnobSpec("knob_1", KnobType.CONTINUOUS, 0.0, 1.0),  # scattering angle (normalized)
        ]

    @property
    def _detector_specs(self) -> list[DetectorSpec]:
        return [
            DetectorSpec("detector_0", "scalar"),  # wavelength shift (normalized)
            DetectorSpec("detector_1", "scalar"),  # scattered intensity (simplified Klein-Nishina)
        ]

    @property
    def _entities(self) -> list[str]:
        return ["entity_A"]

    def _compute(self, knobs: dict[str, float]) -> dict[str, float]:
        lam = denormalize(knobs["knob_0"], _WAVELENGTH_MIN, _WAVELENGTH_MAX)
        theta = denormalize(knobs["knob_1"], _ANGLE_MIN, _ANGLE_MAX)

        delta_lambda = _CHAR_LENGTH * (1.0 - np.cos(theta))

        shift_normed = float(delta_lambda / (2.0 * _CHAR_LENGTH))
        shift_normed = float(np.clip(shift_normed, 0.0, 1.0))

        intensity = float((1.0 + np.cos(theta) ** 2) / 2.0)

        return {
            "detector_0": shift_normed,
            "detector_1": intensity,
        }
