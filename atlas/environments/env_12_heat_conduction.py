"""ENV-12: Classical heat conduction (Fourier's law)."""
from __future__ import annotations

from atlas.environments.base import BaseEnvironment
from atlas.environments.normalizer import denormalize
from atlas.environments.registry import register
from atlas.types import KnobSpec, KnobType, DetectorSpec


@register
class Env12HeatConduction(BaseEnvironment):

    @property
    def env_id(self) -> str:
        return "ENV_12"

    @property
    def _knob_specs(self) -> list[KnobSpec]:
        return [
            KnobSpec("knob_0", KnobType.CONTINUOUS, 0.0, 1.0),  # temperature difference
            KnobSpec("knob_1", KnobType.CONTINUOUS, 0.0, 1.0),  # cross-sectional area
            KnobSpec("knob_2", KnobType.CONTINUOUS, 0.0, 1.0),  # length of conductor
        ]

    @property
    def _detector_specs(self) -> list[DetectorSpec]:
        return [DetectorSpec("detector_0", "scalar")]

    def _compute(self, knobs: dict[str, float]) -> dict[str, float]:
        # Denormalize to physical values
        dT = denormalize(knobs["knob_0"], 1.0, 100.0)    # temperature diff [1, 100] K
        A = denormalize(knobs["knob_1"], 0.001, 0.1)     # area [0.001, 0.1] m^2
        L = denormalize(knobs["knob_2"], 0.01, 1.0)      # length [0.01, 1.0] m

        k = 50.0  # fixed thermal conductivity (e.g., steel) W/(m·K)

        # Fourier's law: Q = k * A * dT / L
        Q = k * A * dT / L

        # Normalize: max Q at dT=100, A=0.1, L=0.01 => Q_max = 50*0.1*100/0.01 = 50000 W
        Q_max = 50000.0
        return {"detector_0": float(Q / Q_max)}
