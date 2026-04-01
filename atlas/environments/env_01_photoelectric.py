"""ENV-01: Photoelectric effect (scalar current output)."""
from __future__ import annotations

import numpy as np

from atlas.environments.base import BaseEnvironment
from atlas.environments.normalizer import denormalize
from atlas.environments.registry import register
from atlas.types import KnobSpec, KnobType, DetectorSpec

# Physical constants — private, never exposed through interface
_H = 6.626e-34          # Planck's constant (J·s)
_E = 1.602e-19          # Elementary charge (C)
_WORK_FUNCTIONS = [2.3, 4.1, 4.7, 5.1]  # eV, indexed by material type
_FREQ_MIN = 1e14        # Hz
_FREQ_MAX = 3e15        # Hz
_VOLTAGE_MIN = -5.0     # V
_VOLTAGE_MAX = 5.0      # V


@register
class Env01Photoelectric(BaseEnvironment):

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

        photon_energy = _H * freq
        W = _WORK_FUNCTIONS[material] * _E
        E_max = photon_energy - W

        if E_max <= 0:
            return {"detector_0": 0.0}

        effective_energy = E_max + _E * voltage
        if effective_energy <= 0:
            return {"detector_0": 0.0}

        current = intensity * (effective_energy / (_H * _FREQ_MAX))
        current = float(np.clip(current, 0.0, 1.0))
        return {"detector_0": current}
