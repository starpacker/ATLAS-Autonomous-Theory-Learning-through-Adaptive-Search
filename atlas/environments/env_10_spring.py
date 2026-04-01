"""ENV-10: Classical spring (simple harmonic motion)."""
from __future__ import annotations

import math

from atlas.environments.base import BaseEnvironment
from atlas.environments.normalizer import denormalize
from atlas.environments.registry import register
from atlas.types import KnobSpec, KnobType, DetectorSpec


@register
class Env10Spring(BaseEnvironment):

    @property
    def env_id(self) -> str:
        return "ENV_10"

    @property
    def _knob_specs(self) -> list[KnobSpec]:
        return [
            KnobSpec("knob_0", KnobType.CONTINUOUS, 0.0, 1.0),  # time
            KnobSpec("knob_1", KnobType.CONTINUOUS, 0.0, 1.0),  # spring constant parameter
            KnobSpec("knob_2", KnobType.CONTINUOUS, 0.0, 1.0),  # amplitude
        ]

    @property
    def _detector_specs(self) -> list[DetectorSpec]:
        return [DetectorSpec("detector_0", "scalar")]

    def _compute(self, knobs: dict[str, float]) -> dict[str, float]:
        # Denormalize to physical values
        t = denormalize(knobs["knob_0"], 0.0, 10.0)       # time [0, 10] s
        k = denormalize(knobs["knob_1"], 0.1, 100.0)      # spring constant [0.1, 100] N/m
        A = denormalize(knobs["knob_2"], 0.01, 1.0)       # amplitude [0.01, 1.0] m

        m = 1.0  # fixed mass = 1 kg

        # SHM: x(t) = A * cos(sqrt(k/m) * t)
        omega = math.sqrt(k / m)
        x = A * math.cos(omega * t)

        # Normalize output to [-1, 1] by dividing by max amplitude
        A_max = 1.0
        return {"detector_0": float(x / A_max)}
