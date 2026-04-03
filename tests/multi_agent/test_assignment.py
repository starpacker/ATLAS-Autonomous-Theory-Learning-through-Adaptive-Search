"""Tests for experiment assignment generation."""
from atlas.multi_agent.assignment import (
    generate_assignment, AssignmentConfig, AgentAssignment, validate_assignment,
)


def test_default_config():
    cfg = AssignmentConfig()
    assert cfg.n_agents == 6
    assert cfg.min_envs_per_agent == 3
    assert cfg.min_coverage == 2


def test_generate_basic():
    config = AssignmentConfig(n_agents=6, min_envs_per_agent=3, min_coverage=2, seed=42)
    assignments = generate_assignment(config)
    assert len(assignments) == 6
    for a in assignments:
        assert isinstance(a, AgentAssignment)
        assert len(a.env_ids) >= 3


def test_all_experiments_covered():
    config = AssignmentConfig(n_agents=6, min_envs_per_agent=4, min_coverage=2, seed=42)
    assignments = generate_assignment(config)
    all_envs = set()
    for a in assignments:
        all_envs.update(a.env_ids)
    expected = {f"ENV_{i:02d}" for i in range(1, 13)}
    assert all_envs == expected


def test_min_coverage():
    config = AssignmentConfig(n_agents=6, min_envs_per_agent=4, min_coverage=2, seed=42)
    assignments = generate_assignment(config)
    # Each experiment should be covered by at least 2 agents
    coverage = {}
    for a in assignments:
        for env_id in a.env_ids:
            coverage[env_id] = coverage.get(env_id, 0) + 1
    for env_id, count in coverage.items():
        assert count >= 2, f"{env_id} only covered by {count} agents"


def test_diverse_assignment():
    """Each agent with enough experiments should have a diverse spread."""
    config = AssignmentConfig(n_agents=6, min_envs_per_agent=4, min_coverage=2, seed=42)
    assignments = generate_assignment(config)
    for a in assignments:
        # With min_envs_per_agent=4 out of 12 total, each agent should
        # have a meaningful subset of experiments
        assert len(a.env_ids) >= 4


def test_deterministic_with_seed():
    c1 = AssignmentConfig(n_agents=4, min_envs_per_agent=3, seed=42)
    c2 = AssignmentConfig(n_agents=4, min_envs_per_agent=3, seed=42)
    a1 = generate_assignment(c1)
    a2 = generate_assignment(c2)
    for x, y in zip(a1, a2):
        assert x.env_ids == y.env_ids


def test_different_seeds_differ():
    c1 = AssignmentConfig(n_agents=4, min_envs_per_agent=3, seed=42)
    c2 = AssignmentConfig(n_agents=4, min_envs_per_agent=3, seed=99)
    a1 = generate_assignment(c1)
    a2 = generate_assignment(c2)
    # At least one agent should have different assignments
    assert any(x.env_ids != y.env_ids for x, y in zip(a1, a2))


def test_validate_assignment():
    config = AssignmentConfig(n_agents=6, min_envs_per_agent=3, min_coverage=2, seed=42)
    assignments = generate_assignment(config)
    errors = validate_assignment(assignments, config)
    assert len(errors) == 0
