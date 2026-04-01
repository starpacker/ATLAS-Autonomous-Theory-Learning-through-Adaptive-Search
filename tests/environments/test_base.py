# tests/environments/test_base.py
"""Tests for base environment interface."""
import numpy as np
from atlas.environments.base import BaseEnvironment
from atlas.types import EnvSchema, KnobSpec, KnobType, DetectorSpec


class DummyEnv(BaseEnvironment):

    @property
    def env_id(self) -> str:
        return "ENV_DUMMY"

    @property
    def _knob_specs(self) -> list[KnobSpec]:
        return [
            KnobSpec("knob_0", KnobType.CONTINUOUS, 0.0, 1.0),
            KnobSpec("knob_1", KnobType.INTEGER, 1, 100),
        ]

    @property
    def _detector_specs(self) -> list[DetectorSpec]:
        return [DetectorSpec("detector_0", "scalar")]

    def _compute(self, knobs: dict[str, float]) -> dict[str, float | np.ndarray]:
        return {"detector_0": knobs["knob_0"] * 2.0}


def test_schema_structure():
    env = DummyEnv()
    schema = env.get_schema()
    assert isinstance(schema, EnvSchema)
    assert schema.env_id == "ENV_DUMMY"
    assert len(schema.knobs) == 2
    assert len(schema.detectors) == 1


def test_run_returns_detector_values():
    env = DummyEnv()
    result = env.run({"knob_0": 0.5, "knob_1": 10})
    assert "detector_0" in result
    assert result["detector_0"] == 1.0


def test_run_validates_knob_range():
    env = DummyEnv()
    try:
        env.run({"knob_0": 1.5, "knob_1": 10})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_run_validates_missing_knob():
    env = DummyEnv()
    try:
        env.run({"knob_0": 0.5})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_schema_has_no_physics_names():
    env = DummyEnv()
    schema = env.get_schema()
    for knob in schema.knobs:
        assert knob.name.startswith("knob_"), f"Knob name '{knob.name}' leaks semantics"
    for det in schema.detectors:
        assert det.name.startswith("detector_"), f"Detector name '{det.name}' leaks semantics"
