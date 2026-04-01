"""Experiment assignment: randomly assign experiments to agents with constraints."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


ALL_ENV_IDS = [f"ENV_{i:02d}" for i in range(1, 13)]
QUANTUM_ENVS = {f"ENV_{i:02d}" for i in [1, 2, 3, 4, 5, 6, 7]}
CLASSICAL_ENVS = {f"ENV_{i:02d}" for i in [8, 9, 10, 11, 12]}


@dataclass
class AssignmentConfig:
    n_agents: int = 6
    min_envs_per_agent: int = 3
    min_coverage: int = 2
    seed: int = 42


@dataclass
class AgentAssignment:
    agent_id: str
    env_ids: list[str]
    seed: int


def generate_assignment(config: AssignmentConfig) -> list[AgentAssignment]:
    """Generate random experiment assignments satisfying all constraints.

    Constraints:
    1. Each agent gets >= min_envs_per_agent experiments
    2. Each experiment is covered by >= min_coverage agents
    3. Quantum/classical experiments are mixed within each agent
    """
    rng = np.random.default_rng(config.seed)
    n = config.n_agents
    all_envs = list(ALL_ENV_IDS)
    n_envs = len(all_envs)

    # Start with ensuring minimum coverage
    agent_envs: list[list[str]] = [[] for _ in range(n)]

    # Phase 1: Ensure each experiment is covered min_coverage times
    for env_id in all_envs:
        # Pick min_coverage random agents to assign this experiment to
        available = list(range(n))
        rng.shuffle(available)
        for agent_idx in available[:config.min_coverage]:
            if env_id not in agent_envs[agent_idx]:
                agent_envs[agent_idx].append(env_id)

    # Phase 2: Top up agents that have fewer than min_envs_per_agent
    for i in range(n):
        while len(agent_envs[i]) < config.min_envs_per_agent:
            # Add a random experiment not already assigned
            candidates = [e for e in all_envs if e not in agent_envs[i]]
            if not candidates:
                break
            chosen = rng.choice(candidates)
            agent_envs[i].append(chosen)

    # Phase 3: Try to ensure quantum/classical mixing
    for i in range(n):
        env_set = set(agent_envs[i])
        has_quantum = bool(env_set & QUANTUM_ENVS)
        has_classical = bool(env_set & CLASSICAL_ENVS)
        if not has_classical and len(agent_envs[i]) < n_envs:
            candidates = [e for e in CLASSICAL_ENVS if e not in env_set]
            if candidates:
                agent_envs[i].append(rng.choice(candidates))
        elif not has_quantum and len(agent_envs[i]) < n_envs:
            candidates = [e for e in QUANTUM_ENVS if e not in env_set]
            if candidates:
                agent_envs[i].append(rng.choice(candidates))

    # Sort each agent's experiments for determinism
    for i in range(n):
        agent_envs[i].sort()

    # Create assignments with unique seeds
    assignments = []
    for i in range(n):
        assignments.append(AgentAssignment(
            agent_id=f"agent_{i}",
            env_ids=agent_envs[i],
            seed=config.seed + i * 1000,
        ))

    return assignments


def validate_assignment(assignments: list[AgentAssignment],
                        config: AssignmentConfig) -> list[str]:
    """Validate that assignments satisfy all constraints. Returns list of errors."""
    errors = []

    for a in assignments:
        if len(a.env_ids) < config.min_envs_per_agent:
            errors.append(f"{a.agent_id}: only {len(a.env_ids)} envs < {config.min_envs_per_agent}")

    coverage: dict[str, int] = {}
    for a in assignments:
        for env_id in a.env_ids:
            coverage[env_id] = coverage.get(env_id, 0) + 1

    for env_id in ALL_ENV_IDS:
        count = coverage.get(env_id, 0)
        if count < config.min_coverage:
            errors.append(f"{env_id}: coverage {count} < {config.min_coverage}")

    return errors
