"""Base environment abstract class with input validation."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from atlas.types import EnvSchema, KnobSpec, KnobType, DetectorSpec


class BaseEnvironment(ABC):

    def get_schema(self) -> EnvSchema:
        return EnvSchema(
            env_id=self.env_id,
            knobs=self._knob_specs,
            detectors=self._detector_specs,
            entities=self._entities,
        )

    def run(self, knobs: dict[str, float | int]) -> dict[str, float | np.ndarray]:
        self._validate_knobs(knobs)
        return self._compute(knobs)

    @property
    @abstractmethod
    def env_id(self) -> str: ...

    @property
    @abstractmethod
    def _knob_specs(self) -> list[KnobSpec]: ...

    @property
    @abstractmethod
    def _detector_specs(self) -> list[DetectorSpec]: ...

    @property
    def _entities(self) -> list[str]:
        return []

    @abstractmethod
    def _compute(self, knobs: dict[str, float | int]) -> dict[str, float | np.ndarray]: ...

    def _validate_knobs(self, knobs: dict[str, float | int]) -> None:
        expected = {s.name for s in self._knob_specs}
        provided = set(knobs.keys())
        missing = expected - provided
        if missing:
            raise ValueError(f"Missing knobs: {missing}")
        extra = provided - expected
        if extra:
            raise ValueError(f"Unexpected knobs: {extra}")
        for spec in self._knob_specs:
            val = knobs[spec.name]
            if spec.knob_type == KnobType.DISCRETE:
                if val not in spec.options:
                    raise ValueError(f"Knob '{spec.name}': value {val} not in options {spec.options}")
            else:
                if val < spec.range_min or val > spec.range_max:
                    raise ValueError(f"Knob '{spec.name}': value {val} out of range [{spec.range_min}, {spec.range_max}]")
