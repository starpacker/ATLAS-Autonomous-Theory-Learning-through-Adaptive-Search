"""ENV-01 experiment environment."""
from __future__ import annotations

import numpy as np

from atlas.environments.base import BaseEnvironment
from atlas.environments.normalizer import denormalize
from atlas.environments.registry import register
from atlas.types import KnobSpec, KnobType, DetectorSpec

# Internal constants — private, never exposed through interface
_H = 6.626e-34
_E = 1.602e-19
_WORK_FUNCTIONS = [2.3, 4.1, 4.7, 5.1]
_FREQ_MIN = 1e14
_FREQ_MAX = 3e15
_VOLTAGE_MIN = -5.0
_VOLTAGE_MAX = 5.0


@register
class Env01(BaseEnvironment):

    @property
    def env_id(self) -> str:
        return "ENV_01"

    @property
    def _knob_specs(self) -> list[KnobSpec]:
        return [
            KnobSpec("knob_0", KnobType.CONTINUOUS, 0.0, 1.0),   # frequency (normalized)
            KnobSpec("knob_1", KnobType.CONTINUOUS, 0.0, 1.0),   # intensity (normalized)
            KnobSpec("knob_2", KnobType.DISCRETE, 0.0, 3.0, options=[0, 1, 2, 3]),  # material
            KnobSpec("knob_3", KnobType.CONTINUOUS, -1.0, 1.0),  # applied voltage (normalized)
        ]

    @property
    def _detector_specs(self) -> list[DetectorSpec]:
        return [DetectorSpec("detector_0", "scalar")]

    @property
    def _entities(self) -> list[str]:
        return ["entity_A"]

    def _compute(self, knobs: dict[str, float | int]) -> dict[str, float]:
        freq = denormalize(knobs["knob_0"], _FREQ_MIN, _FREQ_MAX)
        intensity = knobs["knob_1"]
        material = int(knobs["knob_2"])
        voltage = denormalize(knobs["knob_3"], _VOLTAGE_MIN, _VOLTAGE_MAX,
                              target_min=-1.0, target_max=1.0)

        e_in = _H * freq
        threshold = _WORK_FUNCTIONS[material] * _E
        surplus = e_in - threshold

        if surplus <= 0:
            return {"detector_0": 0.0}

        effective = surplus + _E * voltage
        if effective <= 0:
            return {"detector_0": 0.0}

        output = intensity * (effective / (_H * _FREQ_MAX))
        output = float(np.clip(output, 0.0, 1.0))
        return {"detector_0": output}
