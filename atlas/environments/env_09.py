"""ENV-09 experiment environment."""
from __future__ import annotations

from atlas.environments.base import BaseEnvironment
from atlas.environments.normalizer import denormalize
from atlas.environments.registry import register
from atlas.types import KnobSpec, KnobType, DetectorSpec


@register
class Env09(BaseEnvironment):

    @property
    def env_id(self) -> str:
        return "ENV_09"

    @property
    def _knob_specs(self) -> list[KnobSpec]:
        return [
            KnobSpec("knob_0", KnobType.CONTINUOUS, 0.0, 1.0),  # mass ratio m2/m1
            KnobSpec("knob_1", KnobType.CONTINUOUS, 0.0, 1.0),  # initial velocity of object 1
            KnobSpec("knob_2", KnobType.CONTINUOUS, 0.0, 1.0),  # initial velocity of object 2
        ]

    @property
    def _detector_specs(self) -> list[DetectorSpec]:
        return [
            DetectorSpec("detector_0", "scalar"),  # final velocity of object 1 (normalized)
            DetectorSpec("detector_1", "scalar"),  # final velocity of object 2 (normalized)
        ]

    def _compute(self, knobs: dict[str, float]) -> dict[str, float]:
        # Denormalize: mass ratio in [0.01, 1.0], velocities in [-10, 10] m/s
        mass_ratio = denormalize(knobs["knob_0"], 0.01, 1.0)  # m2/m1
        v1_i = denormalize(knobs["knob_1"], -10.0, 10.0)
        v2_i = denormalize(knobs["knob_2"], -10.0, 10.0)

        m1 = 1.0
        m2 = mass_ratio * m1

        v1_f = ((m1 - m2) * v1_i + 2.0 * m2 * v2_i) / (m1 + m2)
        v2_f = ((m2 - m1) * v2_i + 2.0 * m1 * v1_i) / (m1 + m2)

        # Normalize outputs to [-1, 1] relative to max possible speed (10 m/s)
        v_max = 10.0
        return {
            "detector_0": float(v1_f / v_max),
            "detector_1": float(v2_f / v_max),
        }
