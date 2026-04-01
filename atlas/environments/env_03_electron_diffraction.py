"""ENV-03: Electron diffraction — de Broglie wavelength through a crystal lattice."""
from __future__ import annotations

import numpy as np

from atlas.environments.base import BaseEnvironment
from atlas.environments.normalizer import denormalize
from atlas.environments.registry import register
from atlas.types import KnobSpec, KnobType, DetectorSpec

# Physical constants — private, never exposed through interface
_H = 6.626e-34    # Planck's constant (J·s)
_M_E = 9.109e-31  # Electron rest mass (kg)
_E = 1.602e-19    # Elementary charge (C)

# Crystal lattice spacings (meters) — three material types
_LATTICE_SPACINGS = [2.0e-10, 3.5e-10, 5.0e-10]  # Angstrom-scale


@register
class Env03ElectronDiffraction(BaseEnvironment):

    @property
    def env_id(self) -> str:
        return "ENV_03"

    @property
    def _knob_specs(self) -> list[KnobSpec]:
        return [
            KnobSpec("knob_0", KnobType.CONTINUOUS, 0.0, 1.0),  # accelerating voltage
            KnobSpec("knob_1", KnobType.CONTINUOUS, 0.0, 1.0),  # lattice spacing scale
            KnobSpec("knob_2", KnobType.DISCRETE, 0.0, 2.0, options=[0, 1, 2]),  # crystal type
        ]

    @property
    def _detector_specs(self) -> list[DetectorSpec]:
        return [DetectorSpec("detector_0", "array_1d", 500)]

    @property
    def _entities(self) -> list[str]:
        return ["entity_B"]

    def _compute(self, knobs: dict[str, float | int]) -> dict[str, np.ndarray]:
        voltage = denormalize(knobs["knob_0"], 1e3, 100e3)  # [1kV, 100kV]
        lattice_scale = denormalize(knobs["knob_1"], 0.5, 2.0)  # scale factor [0.5, 2.0]
        crystal = int(knobs["knob_2"])

        # de Broglie wavelength: lambda = h / sqrt(2 * m * E)
        kinetic_energy = _E * voltage  # joules
        lam = _H / np.sqrt(2.0 * _M_E * kinetic_energy)

        # Lattice spacing with user-controlled scaling
        d = _LATTICE_SPACINGS[crystal] * lattice_scale

        # Angular range for radial detector [0, pi/3]
        theta = np.linspace(0.0, np.pi / 3, 500)

        # Bragg condition: 2d sin(theta) = n*lambda  =>  n = 1..5
        intensity = np.zeros(500)
        peak_width = 0.01  # radians, Gaussian peak width

        for n in range(1, 6):
            # Bragg angle for order n
            sin_val = n * lam / (2.0 * d)
            if abs(sin_val) > 1.0:
                continue  # physically inaccessible order
            theta_bragg = np.arcsin(sin_val)
            # Add a Gaussian peak at the Bragg angle
            peak = np.exp(-0.5 * ((theta - theta_bragg) / peak_width) ** 2)
            # Intensity diminishes with order (structure factor approximation)
            intensity += peak / (n ** 2)

        # Normalize to [0, 1]
        max_val = np.max(intensity)
        if max_val > 0:
            intensity = intensity / max_val

        return {"detector_0": intensity}
