"""Tests for ExperimentDataset."""
import numpy as np
from atlas.data.dataset import ExperimentDataset


def test_create_empty():
    ds = ExperimentDataset(env_id="ENV_01", knob_names=["knob_0", "knob_1"],
                           detector_names=["detector_0"])
    assert ds.env_id == "ENV_01"
    assert len(ds) == 0


def test_add_and_len():
    ds = ExperimentDataset(env_id="ENV_01", knob_names=["knob_0"],
                           detector_names=["detector_0"])
    ds.add(knobs={"knob_0": 0.5}, detectors={"detector_0": 1.0})
    ds.add(knobs={"knob_0": 0.7}, detectors={"detector_0": 2.0})
    assert len(ds) == 2


def test_get_knob_array():
    ds = ExperimentDataset(env_id="ENV_01", knob_names=["knob_0", "knob_1"],
                           detector_names=["detector_0"])
    ds.add(knobs={"knob_0": 0.1, "knob_1": 0.2}, detectors={"detector_0": 1.0})
    ds.add(knobs={"knob_0": 0.3, "knob_1": 0.4}, detectors={"detector_0": 2.0})
    X = ds.knob_array()
    assert X.shape == (2, 2)
    np.testing.assert_array_equal(X[0], [0.1, 0.2])


def test_get_detector_array_scalar():
    ds = ExperimentDataset(env_id="ENV_01", knob_names=["knob_0"],
                           detector_names=["detector_0"])
    ds.add(knobs={"knob_0": 0.1}, detectors={"detector_0": 1.5})
    ds.add(knobs={"knob_0": 0.2}, detectors={"detector_0": 2.5})
    y = ds.detector_array("detector_0")
    np.testing.assert_array_equal(y, [1.5, 2.5])


def test_get_detector_array_vector():
    ds = ExperimentDataset(env_id="ENV_04", knob_names=["knob_0"],
                           detector_names=["detector_0"])
    ds.add(knobs={"knob_0": 0.5}, detectors={"detector_0": np.array([1.0, 2.0, 3.0])})
    y = ds.detector_array("detector_0")
    assert y.shape == (1, 3)


def test_split_train_test():
    ds = ExperimentDataset(env_id="ENV_01", knob_names=["knob_0"],
                           detector_names=["detector_0"])
    for i in range(100):
        ds.add(knobs={"knob_0": i / 100}, detectors={"detector_0": float(i)})
    train, test = ds.split(test_fraction=0.2, seed=42)
    assert len(train) + len(test) == 100
    assert len(test) == 20


def test_from_env():
    from atlas.environments.registry import get_environment
    env = get_environment("ENV_10")
    ds = ExperimentDataset.from_env(env, n_samples_per_knob=5, seed=42)
    assert len(ds) > 0
    assert ds.env_id == "ENV_10"
