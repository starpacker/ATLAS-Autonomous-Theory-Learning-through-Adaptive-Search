"""ENV-06 experiment environment."""
from __future__ import annotations

import numpy as np

from atlas.environments.base import BaseEnvironment
from atlas.environments.normalizer import denormalize
from atlas.environments.registry import register
from atlas.types import KnobSpec, KnobType, DetectorSpec

# Internal constants — private, never exposed through interface
_R_H = 1.097e7


@register
class Env06(BaseEnvironment):

    @property
    def env_id(self) -> str:
        return "ENV_06"

    @property
    def _knob_specs(self) -> list[KnobSpec]:
        return [
            KnobSpec("knob_0", KnobType.CONTINUOUS, 0.0, 1.0),  # spectrometer center
            KnobSpec("knob_1", KnobType.CONTINUOUS, 0.0, 1.0),  # excitation energy
        ]

    @property
    def _detector_specs(self) -> list[DetectorSpec]:
        return [DetectorSpec("detector_0", "array_1d", 500)]

    def _compute(self, knobs: dict[str, float | int]) -> dict[str, np.ndarray]:
        # Spectrometer window: center wavelength [300nm, 900nm], half-width 100nm
        center_wl = denormalize(knobs["knob_0"], 300e-9, 900e-9)
        half_width = 100e-9

        wl_min = center_wl - half_width
        wl_max = center_wl + half_width
        wl_axis = np.linspace(wl_min, wl_max, 500)

        # Excitation energy determines maximum n level: mapped to [2, 8]
        n_max = int(denormalize(knobs["knob_1"], 2.0, 8.0))
        n_max = max(n_max, 2)  # need at least n=2 for any transition

        spectrum = np.zeros(500)
        peak_sigma = 1e-9  # 1 nm Gaussian width (sharp lines)

        for n1 in range(1, n_max):
            for n2 in range(n1 + 1, n_max + 1):
                inv_lam = _R_H * (1.0 / n1 ** 2 - 1.0 / n2 ** 2)
                lam_line = 1.0 / inv_lam
                # Add Gaussian peak if within the detector window
                if wl_min - 5 * peak_sigma <= lam_line <= wl_max + 5 * peak_sigma:
                    rel_intensity = (n2 - n1) / (n2 ** 3 * n1 ** 3)
                    peak = rel_intensity * np.exp(
                        -0.5 * ((wl_axis - lam_line) / peak_sigma) ** 2
                    )
                    spectrum += peak

        # Normalize to [0, 1] if any signal present
        max_val = np.max(spectrum)
        if max_val > 0:
            spectrum = spectrum / max_val

        return {"detector_0": spectrum}
