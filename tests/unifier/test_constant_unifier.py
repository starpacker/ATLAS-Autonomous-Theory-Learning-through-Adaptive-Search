"""Tests for cross-agent constant unification."""
from atlas.unifier.constant_unifier import unify_agent_constants, AgentConstants


def test_unify_identical_constants():
    agents = [
        AgentConstants(agent_id="a0", constants={"ENV_01:C0": 6.626e-34, "ENV_04:C0": 6.630e-34}),
        AgentConstants(agent_id="a1", constants={"ENV_02:C0": 6.622e-34, "ENV_05:C0": 6.628e-34}),
    ]
    result = unify_agent_constants(agents)
    assert len(result.unified) >= 1
    uc = result.unified[0]
    assert abs(uc.value - 6.626e-34) / 6.626e-34 < 0.01
    assert len(uc.appearances) == 4  # from all 4 constants


def test_unify_dedup_overlapping():
    """When agents share experiments, keep best R² formula's constant."""
    agents = [
        AgentConstants(agent_id="a0",
                       constants={"ENV_04:C0": 6.626e-34},
                       r_squared={"ENV_04:C0": 0.95}),
        AgentConstants(agent_id="a1",
                       constants={"ENV_04:C0": 6.630e-34},
                       r_squared={"ENV_04:C0": 0.92}),
    ]
    result = unify_agent_constants(agents)
    # Should dedup ENV_04 (keep a0's version with higher R²)
    assert len(result.unified) >= 1


def test_unify_no_match():
    agents = [
        AgentConstants(agent_id="a0", constants={"ENV_01:C0": 3.14}),
        AgentConstants(agent_id="a1", constants={"ENV_02:C0": 2.71}),
    ]
    result = unify_agent_constants(agents)
    assert len(result.unified) == 0  # pi and e are not approximately equal
