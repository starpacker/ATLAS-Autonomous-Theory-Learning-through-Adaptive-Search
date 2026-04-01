"""ENV-02: Compton scattering (wavelength shift + scattered intensity)."""
from __future__ import annotations

import numpy as np

from atlas.environments.base import BaseEnvironment
from atlas.environments.normalizer import denormalize
from atlas.environments.registry import register
from atlas.types import KnobSpec, KnobType, DetectorSpec

# Physical constants — private, never exposed through interface
_H = 6.626e-34          # Planck's constant (J·s)
_M_E = 9.109e-31        # Electron rest mass (kg)
_C = 2.998e8            # Speed of light (m/s)
_COMPTON_WAVELENGTH = _H / (_M_E * _C)  # ~2.426e-12 m

_WAVELENGTH_MIN = 1e-12   # 1 pm
_WAVELENGTH_MAX = 100e-12 # 100 pm
_ANGLE_MIN = 0.0
_ANGLE_MAX = np.pi


@register
class Env02Compton(BaseEnvironment):

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

        # Compton formula: delta_lambda = (h / m_e c) * (1 - cos(theta))
        delta_lambda = _COMPTON_WAVELENGTH * (1.0 - np.cos(theta))

        # Normalize by 2 * COMPTON_WAVELENGTH (maximum possible shift at theta=pi)
        shift_normed = float(delta_lambda / (2.0 * _COMPTON_WAVELENGTH))
        shift_normed = float(np.clip(shift_normed, 0.0, 1.0))

        # Simplified Klein-Nishina: differential cross-section proportional to
        # (1 + cos^2(theta)) / 2, normalized to [0,1]
        intensity = float((1.0 + np.cos(theta) ** 2) / 2.0)

        return {
            "detector_0": shift_normed,
            "detector_1": intensity,
        }
