"""ENV-07 experiment environment."""
from __future__ import annotations

import numpy as np

from atlas.environments.base import BaseEnvironment
from atlas.environments.normalizer import denormalize
from atlas.environments.registry import register
from atlas.types import KnobSpec, KnobType, DetectorSpec


@register
class Env07(BaseEnvironment):

    def __init__(self, seed: int | None = None):
        self._seed = seed

    @property
    def env_id(self) -> str:
        return "ENV_07"

    @property
    def _knob_specs(self) -> list[KnobSpec]:
        return [
            KnobSpec("knob_0", KnobType.CONTINUOUS, 0.0, 1.0),  # preparation angle
            KnobSpec("knob_1", KnobType.CONTINUOUS, 0.0, 1.0),  # field gradient
            KnobSpec("knob_2", KnobType.INTEGER, 1, 1_000_000),  # particle count
        ]

    @property
    def _detector_specs(self) -> list[DetectorSpec]:
        return [DetectorSpec("detector_0", "array_1d", 200)]

    def _compute(self, knobs: dict[str, float | int]) -> dict[str, np.ndarray]:
        theta = denormalize(knobs["knob_0"], 0.0, np.pi)  # preparation angle [0, pi]
        gradient = knobs["knob_1"]                         # field gradient [0, 1] normalized
        N = int(knobs["knob_2"])

        p_up = np.cos(theta / 2) ** 2

        rng = np.random.default_rng(self._seed)
        n_up = int(rng.binomial(N, p_up))
        n_down = N - n_up

        bins = np.zeros(200, dtype=float)
        center = 100
        deflection = int(gradient * 30)
        spread = 3.0  # pixel standard deviation

        if n_up > 0:
            up_pos = rng.normal(center + deflection, spread, size=n_up).astype(int)
            up_pos = np.clip(up_pos, 0, 199)
            np.add.at(bins, up_pos, 1)

        if n_down > 0:
            down_pos = rng.normal(center - deflection, spread, size=n_down).astype(int)
            down_pos = np.clip(down_pos, 0, 199)
            np.add.at(bins, down_pos, 1)

        return {"detector_0": bins}
