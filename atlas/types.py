"""Core data structures used across ATLAS modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class KnobType(Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    INTEGER = "integer"


@dataclass(frozen=True)
class KnobSpec:
    """Specification for a single input knob."""
    name: str
    knob_type: KnobType
    range_min: float
    range_max: float
    options: list[int] | None = None  # only for DISCRETE type


@dataclass(frozen=True)
class DetectorSpec:
    """Specification for a single detector output."""
    name: str
    output_type: str  # "scalar", "array_1d", "list"
    length: int | None = None  # for array_1d


@dataclass(frozen=True)
class EnvSchema:
    """Complete schema for an experiment environment."""
    env_id: str
    knobs: list[KnobSpec]
    detectors: list[DetectorSpec]
    entities: list[str] = field(default_factory=list)


@dataclass
class FitMetrics:
    """Fit quality metrics for a formula on an experiment."""
    r_squared: float
    residual_var: float
    mdl: float
    n_seeds: int = 1


@dataclass
class FormulaRecord:
    """A discovered formula with its provenance and fit metrics."""
    expr_str: str  # serialized expression
    env_id: str
    fit: FitMetrics
    constants: dict[str, float] = field(default_factory=dict)
