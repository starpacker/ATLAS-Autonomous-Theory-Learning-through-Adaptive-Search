"""Knob normalization utilities."""
from __future__ import annotations


def normalize(value: float, phys_min: float, phys_max: float,
              target_min: float = 0.0, target_max: float = 1.0) -> float:
    return target_min + (value - phys_min) / (phys_max - phys_min) * (target_max - target_min)


def denormalize(normed: float, phys_min: float, phys_max: float,
                target_min: float = 0.0, target_max: float = 1.0) -> float:
    return phys_min + (normed - target_min) / (target_max - target_min) * (phys_max - phys_min)
