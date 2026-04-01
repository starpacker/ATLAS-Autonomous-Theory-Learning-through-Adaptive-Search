"""ENV-08: Classical two-source wave interference pattern."""
from __future__ import annotations

import numpy as np

from atlas.environments.base import BaseEnvironment
from atlas.environments.normalizer import denormalize
from atlas.environments.registry import register
from atlas.types import KnobSpec, KnobType, DetectorSpec


@register
class Env08WaterWave(BaseEnvironment):

    @property
    def env_id(self) -> str:
        return "ENV_08"

    @property
    def _knob_specs(self) -> list[KnobSpec]:
        return [
            KnobSpec("knob_0", KnobType.CONTINUOUS, 0.0, 1.0),  # source separation
            KnobSpec("knob_1", KnobType.CONTINUOUS, 0.0, 1.0),  # wavelength
            KnobSpec("knob_2", KnobType.CONTINUOUS, 0.0, 1.0),  # screen distance
        ]

    @property
    def _detector_specs(self) -> list[DetectorSpec]:
        return [DetectorSpec("detector_0", "array_1d", 1000)]

    def _compute(self, knobs: dict[str, float]) -> dict[str, np.ndarray]:
        # Denormalize to physical values
        d = denormalize(knobs["knob_0"], 0.5e-3, 5.0e-3)    # source separation [0.5mm, 5mm]
        lam = denormalize(knobs["knob_1"], 400e-9, 700e-9)   # wavelength [400nm, 700nm]
        L = denormalize(knobs["knob_2"], 0.5, 2.0)           # screen distance [0.5m, 2m]

        # Position array across screen [-5mm, 5mm]
        x = np.linspace(-5e-3, 5e-3, 1000)

        # Classical two-source interference: I(x) = cos^2(pi * d * x / (lambda * L))
        intensity = np.cos(np.pi * d * x / (lam * L)) ** 2

        return {"detector_0": intensity}
