"""ENV-11 experiment environment."""
from __future__ import annotations

from atlas.environments.base import BaseEnvironment
from atlas.environments.normalizer import denormalize
from atlas.environments.registry import register
from atlas.types import KnobSpec, KnobType, DetectorSpec


@register
class Env11(BaseEnvironment):

    @property
    def env_id(self) -> str:
        return "ENV_11"

    @property
    def _knob_specs(self) -> list[KnobSpec]:
        return [
            KnobSpec("knob_0", KnobType.CONTINUOUS, 0.0, 1.0),  # time
            KnobSpec("knob_1", KnobType.CONTINUOUS, 0.0, 1.0),  # initial velocity
        ]

    @property
    def _detector_specs(self) -> list[DetectorSpec]:
        return [DetectorSpec("detector_0", "scalar")]

    def _compute(self, knobs: dict[str, float]) -> dict[str, float]:
        # Denormalize to physical values
        t = denormalize(knobs["knob_0"], 0.0, 10.0)     # time [0, 10] s
        v0 = denormalize(knobs["knob_1"], 0.0, 50.0)    # initial upward velocity [0, 50] m/s

        g = 9.81  # m/s^2

        y = v0 * t - 0.5 * g * t * t

        # Normalize: max height for v0=50 is v0^2/(2g) ≈ 127.4m; use 200m as scale
        y_scale = 200.0
        return {"detector_0": float(y / y_scale)}
