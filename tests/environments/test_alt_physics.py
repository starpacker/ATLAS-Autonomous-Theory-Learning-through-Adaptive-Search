"""Tests for alternative physics configuration."""
from atlas.environments.alt_physics import PhysicsConfig, altered_physics
from atlas.environments.registry import get_environment


def test_default_config():
    cfg = PhysicsConfig()
    assert cfg.h_multiplier == 1.0
    assert cfg.c_multiplier == 1.0


def test_alt_config_changes_output():
    """With h->2h, photoelectric cutoff should change."""
    env = get_environment("ENV_01")
    knobs = {"knob_0": 0.7, "knob_1": 0.5, "knob_2": 0, "knob_3": 0.0}

    r_default = env.run(knobs)

    with altered_physics(PhysicsConfig(h_multiplier=2.0)):
        env2 = get_environment("ENV_01")
        r_alt = env2.run(knobs)

    assert r_default["detector_0"] != r_alt["detector_0"]


def test_classical_limit():
    """With h->~0, quantum experiments should behave classically."""
    with altered_physics(PhysicsConfig(h_multiplier=1e-10)):
        env = get_environment("ENV_01")
        result = env.run({"knob_0": 0.5, "knob_1": 0.5, "knob_2": 0, "knob_3": 0.0})
    # At h~0, photon energy negligible -> no emission
    assert result["detector_0"] <= 1e-6


def test_context_manager_restores():
    """After context manager exits, physics should be back to normal."""
    env_before = get_environment("ENV_01")
    r_before = env_before.run({"knob_0": 0.7, "knob_1": 0.5, "knob_2": 0, "knob_3": 0.0})

    with altered_physics(PhysicsConfig(h_multiplier=100.0)):
        pass  # do something with altered physics

    env_after = get_environment("ENV_01")
    r_after = env_after.run({"knob_0": 0.7, "knob_1": 0.5, "knob_2": 0, "knob_3": 0.0})

    assert r_before["detector_0"] == r_after["detector_0"]
