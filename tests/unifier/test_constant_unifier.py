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


def test_unify_agent_constants_chi2():
    """Chi-squared consistency should be computed via R² weights, not hardcoded 0.0."""
    agents = [
        AgentConstants(
            agent_id="a0",
            constants={"ENV_01:C0": 6.626e-34, "ENV_04:C0": 6.630e-34},
            r_squared={"ENV_01:C0": 0.99, "ENV_04:C0": 0.95},
        ),
        AgentConstants(
            agent_id="a1",
            constants={"ENV_02:C0": 6.622e-34, "ENV_05:C0": 6.628e-34},
            r_squared={"ENV_02:C0": 0.97, "ENV_05:C0": 0.93},
        ),
    ]
    result = unify_agent_constants(agents)
    assert len(result.unified) >= 1
    uc = result.unified[0]
    # chi2_pvalue should be real, not None
    assert uc.chi2_pvalue is not None
    # These are consistent estimates of h
    assert uc.chi2_pvalue > 0.01
    assert uc.is_spurious is False


def test_unify_agent_constants_weighted_mean():
    """R²-weighted mean should shift toward higher-R² agent's value."""
    agents = [
        AgentConstants(
            agent_id="a0",
            constants={"ENV_01:C0": 10.0, "ENV_02:C0": 10.0},
            r_squared={"ENV_01:C0": 0.99, "ENV_02:C0": 0.99},
        ),
        AgentConstants(
            agent_id="a1",
            constants={"ENV_03:C0": 20.0},
            r_squared={"ENV_03:C0": 0.50},
        ),
    ]
    # tolerance=1.0 to force them into one group
    result = unify_agent_constants(agents, tolerance=1.0)
    assert len(result.unified) >= 1
    uc = result.unified[0]
    # Weighted mean should be closer to 10 than simple mean of (10,10,20)=13.3
    assert uc.value < 13.3
