"""ExperimentDataset: structured storage for experiment observations."""
from __future__ import annotations

import numpy as np
from atlas.types import KnobType


class ExperimentDataset:
    """Stores (knob_settings, detector_readings) pairs from an experiment."""

    def __init__(self, env_id: str, knob_names: list[str], detector_names: list[str]):
        self.env_id = env_id
        self.knob_names = list(knob_names)
        self.detector_names = list(detector_names)
        self._knobs: list[dict[str, float]] = []
        self._detectors: list[dict[str, float | np.ndarray]] = []

    def __len__(self) -> int:
        return len(self._knobs)

    def add(self, knobs: dict[str, float], detectors: dict[str, float | np.ndarray]) -> None:
        self._knobs.append(dict(knobs))
        self._detectors.append(dict(detectors))

    def get_knobs(self, index: int) -> dict[str, float]:
        """Return the knob settings for a specific sample by index."""
        return dict(self._knobs[index])

    def iter_knobs(self):
        """Iterate over all knob settings (each as a dict)."""
        for k in self._knobs:
            yield dict(k)

    def knob_array(self) -> np.ndarray:
        return np.array([[k[name] for name in self.knob_names] for k in self._knobs])

    def detector_array(self, detector_name: str) -> np.ndarray:
        values = [d[detector_name] for d in self._detectors]
        if not values:
            return np.array([])
        if isinstance(values[0], np.ndarray):
            return np.stack(values)
        return np.array(values)

    def split(self, test_fraction: float = 0.2, seed: int = 42) -> tuple[ExperimentDataset, ExperimentDataset]:
        rng = np.random.default_rng(seed)
        n = len(self)
        n_test = int(n * test_fraction)
        indices = rng.permutation(n)
        test_idx = set(indices[:n_test].tolist())
        train = ExperimentDataset(self.env_id, self.knob_names, self.detector_names)
        test = ExperimentDataset(self.env_id, self.knob_names, self.detector_names)
        for i in range(n):
            target = test if i in test_idx else train
            target.add(self._knobs[i], self._detectors[i])
        return train, test

    @classmethod
    def from_env(cls, env, n_samples_per_knob: int = 10, seed: int = 42) -> ExperimentDataset:
        schema = env.get_schema()
        knob_names = [k.name for k in schema.knobs]
        detector_names = [d.name for d in schema.detectors]
        ds = cls(schema.env_id, knob_names, detector_names)
        rng = np.random.default_rng(seed)
        continuous_knobs = [k for k in schema.knobs if k.knob_type == KnobType.CONTINUOUS]
        discrete_knobs = [k for k in schema.knobs if k.knob_type in (KnobType.DISCRETE, KnobType.INTEGER)]

        if len(continuous_knobs) <= 3:
            grids = [np.linspace(k.range_min, k.range_max, n_samples_per_knob) for k in continuous_knobs]
            mesh = np.meshgrid(*grids, indexing='ij') if grids else []
            flat = [m.ravel() for m in mesh]
            n_grid = len(flat[0]) if flat else 1
            discrete_combos = _discrete_combinations(discrete_knobs)
            for combo in discrete_combos:
                for i in range(n_grid):
                    knobs = {}
                    for j, k in enumerate(continuous_knobs):
                        knobs[k.name] = float(flat[j][i])
                    knobs.update(combo)
                    try:
                        result = env.run(knobs)
                        ds.add(knobs, result)
                    except (ValueError, RuntimeError):
                        pass
        else:
            discrete_combos = _discrete_combinations(discrete_knobs)
            n_total = n_samples_per_knob ** 3
            for _ in range(n_total):
                knobs = {}
                for k in continuous_knobs:
                    knobs[k.name] = float(rng.uniform(k.range_min, k.range_max))
                combo = discrete_combos[rng.integers(len(discrete_combos))]
                knobs.update(combo)
                try:
                    result = env.run(knobs)
                    ds.add(knobs, result)
                except (ValueError, RuntimeError):
                    pass
        return ds


def _discrete_combinations(knobs) -> list[dict[str, float]]:
    if not knobs:
        return [{}]
    combos = [{}]
    for k in knobs:
        if k.knob_type == KnobType.DISCRETE and k.options:
            new_combos = []
            for combo in combos:
                for opt in k.options:
                    new_combos.append({**combo, k.name: opt})
            combos = new_combos
        elif k.knob_type == KnobType.INTEGER:
            vals = [int(k.range_min), int((k.range_min + k.range_max) / 2), int(k.range_max)]
            new_combos = []
            for combo in combos:
                for v in vals:
                    new_combos.append({**combo, k.name: v})
            combos = new_combos
    return combos
