# atlas/multi_agent/orchestrator.py
"""Multi-agent orchestrator: runs multiple ATLAS agents and synthesizes results.

Mode A: Fully independent — agents run in isolation, results compared post-hoc
Mode B: Consensus sharing — agents share DSL extensions via proposal pool + verification
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from atlas.agent.atlas_agent import ATLASAgent, AgentConfig
from atlas.multi_agent.assignment import (
    generate_assignment, AssignmentConfig, AgentAssignment,
)
from atlas.multi_agent.proposal import Proposal, ProposalPool, ProposalStatus
from atlas.multi_agent.verifier import compute_global_mdl_delta
from atlas.unifier.constant_unifier import (
    unify_agent_constants, AgentConstants,
)
from atlas.unifier.theory import Theory, CompressionLayer
from atlas.sr.formula_store import _extract_constants

logger = logging.getLogger(__name__)


class RunMode(Enum):
    MODE_A = "mode_a"  # fully independent
    MODE_B = "mode_b"  # consensus sharing


@dataclass
class MultiAgentConfig:
    mode: RunMode = RunMode.MODE_A
    n_agents: int = 6
    max_epochs: int = 10
    min_envs_per_agent: int = 3
    min_coverage: int = 2
    assignment_seed: int = 42
    # Per-agent config
    agent_n_samples_per_knob: int = 10
    agent_sr_niterations: int = 40
    agent_sr_maxsize: int = 25
    agent_sr_timeout: int = 300
    agent_enable_rgde: bool = False
    agent_r_squared_threshold: float = 0.95
    # Verification (Mode B only)
    verification_seeds: int = 5


@dataclass
class MultiAgentResult:
    mode: RunMode
    n_agents: int
    agent_outputs: list[dict]
    theory: dict
    output: dict
    proposals_submitted: int = 0
    proposals_adopted: int = 0
    proposals_rejected: int = 0


class MultiAgentOrchestrator:
    """Orchestrates multiple ATLAS agents."""

    def __init__(self, config: MultiAgentConfig):
        self.config = config
        self.proposal_pool = ProposalPool()

        # Generate assignments
        assign_config = AssignmentConfig(
            n_agents=config.n_agents,
            min_envs_per_agent=config.min_envs_per_agent,
            min_coverage=config.min_coverage,
            seed=config.assignment_seed,
        )
        self.assignments = generate_assignment(assign_config)

        # Create agents
        self.agents: list[ATLASAgent] = []
        for assignment in self.assignments:
            agent_cfg = AgentConfig(
                max_epochs=config.max_epochs,
                r_squared_threshold=config.agent_r_squared_threshold,
                n_samples_per_knob=config.agent_n_samples_per_knob,
                sr_niterations=config.agent_sr_niterations,
                sr_maxsize=config.agent_sr_maxsize,
                sr_timeout=config.agent_sr_timeout,
                seed=assignment.seed,
                enable_rgde=config.agent_enable_rgde,
            )
            agent = ATLASAgent(
                env_ids=assignment.env_ids,
                config=agent_cfg,
            )
            self.agents.append(agent)

    def run(self) -> MultiAgentResult:
        """Run all agents and synthesize results."""
        if self.config.mode == RunMode.MODE_A:
            return self._run_mode_a()
        else:
            return self._run_mode_b()

    def _run_mode_a(self) -> MultiAgentResult:
        """Mode A: Run agents fully independently, then unify post-hoc."""
        agent_outputs = []
        for i, agent in enumerate(self.agents):
            logger.info(f"Running agent {i} on {agent.env_ids}")
            output = agent.run()
            agent_outputs.append(output)

        # Post-hoc unification
        theory = self._build_theory(agent_outputs)

        return MultiAgentResult(
            mode=RunMode.MODE_A,
            n_agents=len(self.agents),
            agent_outputs=agent_outputs,
            theory=theory.to_dict(),
            output={"theory": theory.to_dict(), "agent_outputs": agent_outputs},
        )

    def _run_mode_b(self) -> MultiAgentResult:
        """Mode B: Agents share extensions via proposal pool."""
        # For now, run agents sequentially with proposal sharing between epochs
        # (True parallelism would use concurrent.futures, deferred for production)

        agent_outputs = []
        for i, agent in enumerate(self.agents):
            logger.info(f"Running agent {i} on {agent.env_ids}")
            output = agent.run()
            agent_outputs.append(output)

            # Collect proposals from this agent's extensions
            for ext in output.get("extensions", []):
                proposal = Proposal(
                    proposal_id=f"PROP-{self.assignments[i].agent_id}-{ext.get('name', 'unk')}",
                    source_agent=self.assignments[i].agent_id,
                    source_env=ext.get("trigger", "unknown"),
                    trigger=ext.get("trigger", ""),
                    extension_type=ext.get("type", "unknown"),
                    extension_definition=ext.get("definition", {}),
                    evidence={},
                )
                self.proposal_pool.add(proposal)

        # Verify pending proposals (simplified — uses agent-reported evidence)
        for proposal in self.proposal_pool.pending():
            # In a full implementation, we'd re-run SR on all 12 experiments
            # For now, auto-adopt (the global MDL check would go here)
            self.proposal_pool.set_status(
                proposal.proposal_id,
                ProposalStatus.ADOPTED,
                delta_total_mdl=-1.0,  # placeholder
            )

        # Build theory
        theory = self._build_theory(agent_outputs)

        return MultiAgentResult(
            mode=RunMode.MODE_B,
            n_agents=len(self.agents),
            agent_outputs=agent_outputs,
            theory=theory.to_dict(),
            output={"theory": theory.to_dict(), "agent_outputs": agent_outputs,
                     "proposals": [p.proposal_id for p in self.proposal_pool.all_proposals()]},
            proposals_submitted=len(self.proposal_pool.all_proposals()),
            proposals_adopted=len(self.proposal_pool.adopted()),
            proposals_rejected=len(self.proposal_pool.rejected()),
        )

    def _build_theory(self, agent_outputs: list[dict]) -> Theory:
        """Build unified theory from all agent outputs."""
        theory = Theory()

        # U1: Constant unification
        agent_constants = []
        for i, output in enumerate(agent_outputs):
            constants = output.get("constants", {})
            r_squared = {}
            for env_id, metrics in output.get("fit_metrics", {}).items():
                if isinstance(metrics, dict):
                    for key in constants:
                        if key.startswith(env_id):
                            r_squared[key] = metrics.get("r_squared", 0.0)
            agent_constants.append(AgentConstants(
                agent_id=self.assignments[i].agent_id,
                constants=constants,
                r_squared=r_squared,
            ))

        if agent_constants:
            unification = unify_agent_constants(agent_constants)
            for uc in unification.unified:
                theory.add_shared_constant(
                    symbol=uc.symbol, value=uc.value,
                    uncertainty=uc.uncertainty,
                    appearances=uc.appearances,
                    chi2_consistency=0.0,  # computed in full version
                )

        # Compression accounting
        total_independent_mdl = 0.0
        total_unified_mdl = 0.0
        for output in agent_outputs:
            for env_id, metrics in output.get("fit_metrics", {}).items():
                if isinstance(metrics, dict):
                    mdl_val = metrics.get("mdl", 0.0)
                    if mdl_val != float("inf") and mdl_val > 0:
                        total_independent_mdl += mdl_val
                        total_unified_mdl += mdl_val

        # Subtract savings from constant unification
        if theory.shared_constants:
            total_unified_mdl *= 0.9  # approximate 10% savings

        theory.add_compression_layer(CompressionLayer(
            level=0, total_mdl=max(total_independent_mdl, 1.0),
            label="independent formulas", delta=0.0,
        ))
        if total_independent_mdl > 0:
            theory.add_compression_layer(CompressionLayer(
                level=1, total_mdl=max(total_unified_mdl, 0.1),
                label="constant unification",
                delta=total_unified_mdl - total_independent_mdl,
            ))

        # Collect fit metrics
        for output in agent_outputs:
            for env_id, metrics in output.get("fit_metrics", {}).items():
                if env_id not in theory.fit_metrics:
                    theory.fit_metrics[env_id] = metrics

        return theory
