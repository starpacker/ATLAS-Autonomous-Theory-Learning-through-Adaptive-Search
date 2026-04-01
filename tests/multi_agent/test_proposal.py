"""Tests for proposal pool management."""
from atlas.multi_agent.proposal import Proposal, ProposalPool, ProposalStatus


def test_create_proposal():
    p = Proposal(
        proposal_id="PROP-agent_0-0-0",
        source_agent="agent_0",
        source_env="ENV_07",
        trigger="D1=stochastic, D3=K>N",
        extension_type="new_type",
        extension_definition={"name": "State_ENV_07", "dimension": 3},
        evidence={"fit_before": {"ENV_07": 0.3}, "fit_after": {"ENV_07": 0.95}},
    )
    assert p.status == ProposalStatus.PENDING


def test_pool_add_and_get():
    pool = ProposalPool()
    p = Proposal(proposal_id="P1", source_agent="a0", source_env="E01",
                 trigger="test", extension_type="new_operator",
                 extension_definition={"name": "concept_cos2"}, evidence={})
    pool.add(p)
    assert len(pool.pending()) == 1
    assert pool.get("P1") == p


def test_pool_adopt():
    pool = ProposalPool()
    p = Proposal(proposal_id="P1", source_agent="a0", source_env="E01",
                 trigger="test", extension_type="new_operator",
                 extension_definition={}, evidence={})
    pool.add(p)
    pool.set_status("P1", ProposalStatus.ADOPTED, delta_total_mdl=-50.0)
    assert len(pool.pending()) == 0
    assert len(pool.adopted()) == 1
    assert pool.get("P1").delta_total_mdl == -50.0


def test_pool_reject():
    pool = ProposalPool()
    p = Proposal(proposal_id="P1", source_agent="a0", source_env="E01",
                 trigger="test", extension_type="new_operator",
                 extension_definition={}, evidence={})
    pool.add(p)
    pool.set_status("P1", ProposalStatus.REJECTED, delta_total_mdl=10.0)
    assert len(pool.pending()) == 0
    assert len(pool.rejected()) == 1


def test_pool_history():
    pool = ProposalPool()
    for i in range(3):
        pool.add(Proposal(proposal_id=f"P{i}", source_agent="a0", source_env="E01",
                          trigger="test", extension_type="op", extension_definition={},
                          evidence={}))
    pool.set_status("P0", ProposalStatus.ADOPTED, delta_total_mdl=-10.0)
    pool.set_status("P1", ProposalStatus.REJECTED, delta_total_mdl=5.0)
    assert len(pool.pending()) == 1
    assert len(pool.adopted()) == 1
    assert len(pool.rejected()) == 1
