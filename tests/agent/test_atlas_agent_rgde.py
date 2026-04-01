# tests/agent/test_atlas_agent_rgde.py
import pytest
from atlas.agent.atlas_agent import ATLASAgent, AgentConfig

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def test_agent_config_has_rgde_fields():
    cfg = AgentConfig()
    assert hasattr(cfg, "enable_rgde")
    assert hasattr(cfg, "rgde_k_range")

def test_agent_rgde_disabled_by_default():
    cfg = AgentConfig()
    assert cfg.enable_rgde is False

def test_agent_output_has_extensions():
    agent = ATLASAgent(env_ids=["ENV_10"],
                       config=AgentConfig(max_epochs=1, n_samples_per_knob=5,
                                          sr_niterations=5, sr_maxsize=8, enable_rgde=False))
    output = agent.run()
    assert "extensions" in output

@pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")
def test_agent_with_rgde_enabled_no_crash():
    agent = ATLASAgent(env_ids=["ENV_10"],
                       config=AgentConfig(max_epochs=1, n_samples_per_knob=5,
                                          sr_niterations=5, sr_maxsize=8,
                                          enable_rgde=True, rgde_k_range=[1, 2],
                                          rgde_scinet_epochs=30, rgde_sr_niterations=5))
    output = agent.run()
    assert "extensions" in output
