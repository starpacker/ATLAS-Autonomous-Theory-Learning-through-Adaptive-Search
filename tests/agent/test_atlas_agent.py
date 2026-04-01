"""Tests for the ATLAS single-agent main loop."""
import pytest
import numpy as np
from atlas.agent.atlas_agent import ATLASAgent, AgentConfig, EpochResult

try:
    import pysr
    HAS_PYSR = True
except ImportError:
    HAS_PYSR = False


def test_agent_creation():
    agent = ATLASAgent(
        env_ids=["ENV_10", "ENV_11"],
        config=AgentConfig(max_epochs=2, n_samples_per_knob=5),
    )
    assert agent.env_ids == ["ENV_10", "ENV_11"]
    assert len(agent.dsl_state.concepts) == 0


def test_agent_config_defaults():
    cfg = AgentConfig()
    assert cfg.max_epochs == 10
    assert cfg.r_squared_threshold == 0.95
    assert cfg.n_samples_per_knob == 10


def test_collect_data():
    agent = ATLASAgent(
        env_ids=["ENV_10"],
        config=AgentConfig(n_samples_per_knob=5),
    )
    agent.collect_data()
    assert "ENV_10" in agent.datasets
    assert len(agent.datasets["ENV_10"]) > 0


def test_agent_output_structure_no_pysr():
    """Test that agent.run() returns proper structure even without PySR."""
    agent = ATLASAgent(
        env_ids=["ENV_10", "ENV_11"],
        config=AgentConfig(max_epochs=1, n_samples_per_knob=5,
                           sr_niterations=5, sr_maxsize=8),
    )
    output = agent.run()
    assert "formulas" in output
    assert "constants" in output
    assert "concepts" in output
    assert "diagnostics" in output
    assert "dsl_state" in output
    assert "fit_metrics" in output
    assert "epochs_run" in output


@pytest.mark.skipif(not HAS_PYSR, reason="PySR not installed")
def test_run_epoch_with_pysr():
    agent = ATLASAgent(
        env_ids=["ENV_10"],
        config=AgentConfig(max_epochs=1, n_samples_per_knob=5,
                           sr_niterations=10, sr_maxsize=10),
    )
    agent.collect_data()
    result = agent.run_epoch()
    assert isinstance(result, EpochResult)
    assert result.epoch == 0
    assert isinstance(result.formulas_found, int)
    assert isinstance(result.diagnostics, dict)
