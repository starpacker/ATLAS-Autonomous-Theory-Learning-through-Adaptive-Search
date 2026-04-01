# tests/multi_agent/test_orchestrator.py
"""Tests for multi-agent orchestrator."""
import pytest
from atlas.multi_agent.orchestrator import (
    MultiAgentConfig, MultiAgentOrchestrator, MultiAgentResult, RunMode,
)


def test_config_defaults():
    cfg = MultiAgentConfig()
    assert cfg.mode == RunMode.MODE_A
    assert cfg.n_agents == 6


def test_config_mode_b():
    cfg = MultiAgentConfig(mode=RunMode.MODE_B)
    assert cfg.mode == RunMode.MODE_B


def test_create_orchestrator():
    cfg = MultiAgentConfig(n_agents=3, max_epochs=1,
                           agent_n_samples_per_knob=3,
                           agent_sr_niterations=5,
                           agent_sr_maxsize=8)
    orch = MultiAgentOrchestrator(cfg)
    assert len(orch.assignments) == 3


def test_run_mode_a_returns_result():
    cfg = MultiAgentConfig(
        mode=RunMode.MODE_A, n_agents=2, max_epochs=1,
        agent_n_samples_per_knob=3,
        agent_sr_niterations=5, agent_sr_maxsize=8,
        min_envs_per_agent=3,
    )
    orch = MultiAgentOrchestrator(cfg)
    result = orch.run()
    assert isinstance(result, MultiAgentResult)
    assert result.mode == RunMode.MODE_A
    assert result.n_agents == 2
    assert "theory" in result.output


def test_run_mode_b_returns_result():
    cfg = MultiAgentConfig(
        mode=RunMode.MODE_B, n_agents=2, max_epochs=1,
        agent_n_samples_per_knob=3,
        agent_sr_niterations=5, agent_sr_maxsize=8,
        min_envs_per_agent=3,
    )
    orch = MultiAgentOrchestrator(cfg)
    result = orch.run()
    assert isinstance(result, MultiAgentResult)
    assert result.mode == RunMode.MODE_B


def test_output_has_per_agent_results():
    cfg = MultiAgentConfig(
        mode=RunMode.MODE_A, n_agents=2, max_epochs=1,
        agent_n_samples_per_knob=3,
        agent_sr_niterations=5, agent_sr_maxsize=8,
        min_envs_per_agent=3,
    )
    orch = MultiAgentOrchestrator(cfg)
    result = orch.run()
    assert len(result.agent_outputs) == 2
