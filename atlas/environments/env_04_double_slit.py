"""ENV-04: Double-slit interference with wave-particle duality."""
from __future__ import annotations

import numpy as np

from atlas.environments.base import BaseEnvironment
from atlas.environments.normalizer import denormalize
from atlas.environments.registry import register
from atlas.types import KnobSpec, KnobType, DetectorSpec

# Physical constants — private, never exposed through interface
_H = 6.626e-34   # Planck's constant (J·s)
_L = 1.0         # Fixed screen distance (m)


@register
class Env04DoubleSlit(BaseEnvironment):

    def __init__(self, seed: int | None = None):
        self._seed = seed

    @property
    def env_id(self) -> str:
        return "ENV_04"

    @property
    def _knob_specs(self) -> list[KnobSpec]:
        return [
            KnobSpec("knob_0", KnobType.CONTINUOUS, 0.0, 1.0),  # slit width
            KnobSpec("knob_1", KnobType.CONTINUOUS, 0.0, 1.0),  # slit separation
            KnobSpec("knob_2", KnobType.CONTINUOUS, 0.0, 1.0),  # wavelength
            KnobSpec("knob_3", KnobType.INTEGER, 1, 1_000_000),  # source intensity
        ]

    @property
    def _detector_specs(self) -> list[DetectorSpec]:
        return [DetectorSpec("detector_0", "array_1d", 1000)]

    @property
    def _entities(self) -> list[str]:
        return ["entity_A"]

    def _compute(self, knobs: dict[str, float | int]) -> dict[str, np.ndarray]:
        a = denormalize(knobs["knob_0"], 10e-6, 200e-6)    # slit width [10um, 200um]
        d = denormalize(knobs["knob_1"], 50e-6, 1000e-6)   # slit separation [50um, 1000um]
        lam = denormalize(knobs["knob_2"], 400e-9, 700e-9) # wavelength [400nm, 700nm]
        N = int(knobs["knob_3"])

        x = np.linspace(-0.02, 0.02, 1000)  # screen positions [-2cm, 2cm]

        alpha = np.pi * a * x / (lam * _L)
        beta = np.pi * d * x / (lam * _L)

        sinc_env = np.where(np.abs(alpha) < 1e-10, 1.0, np.sin(alpha) / alpha)
        interference = np.cos(beta) ** 2
        intensity = sinc_env ** 2 * interference

        if N >= 10_000:
            # High intensity: return smooth normalized pattern
            max_val = np.max(intensity)
            if max_val > 0:
                output = intensity / max_val
            else:
                output = intensity
        else:
            # Low intensity: discrete photon/particle hits via multinomial sampling
            total = np.sum(intensity)
            if total > 0:
                prob = intensity / total
            else:
                prob = np.ones(1000) / 1000.0
            rng = np.random.default_rng(self._seed)
            output = rng.multinomial(N, prob).astype(float)

        return {"detector_0": output}
