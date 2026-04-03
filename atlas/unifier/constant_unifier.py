"""U1: Cross-agent constant unification.

Collects constants from all agents, deduplicates for overlapping experiments,
then runs PSLQ unification.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from atlas.analysis.pslq_unifier import unify_constants, UnifiedConstant


@dataclass
class AgentConstants:
    """Constants discovered by a single agent."""
    agent_id: str
    constants: dict[str, float]       # "ENV_XX:CN" -> value
    r_squared: dict[str, float] = field(default_factory=dict)  # "ENV_XX:CN" -> R²


@dataclass
class CrossAgentUnificationResult:
    unified: list[UnifiedConstant]
    deduplicated_constants: dict[str, float]
    n_before_dedup: int
    n_after_dedup: int


def unify_agent_constants(agents: list[AgentConstants],
                          tolerance: float = 0.01) -> CrossAgentUnificationResult:
    """Unify constants across all agents.

    Steps:
    1. Collect all constants from all agents
    2. Deduplicate: for overlapping experiments, keep the constant from the
       agent with the highest R²
    3. Run PSLQ unification on deduplicated constants
    """
    # Step 1: Collect all
    all_constants: dict[str, list[tuple[float, float, str]]] = {}
    # key = "ENV_XX:CN", value = list of (value, r_squared, agent_id)

    for agent in agents:
        for key, value in agent.constants.items():
            r2 = agent.r_squared.get(key, 0.0)
            if key not in all_constants:
                all_constants[key] = []
            all_constants[key].append((value, r2, agent.agent_id))

    n_before = sum(len(v) for v in all_constants.values())

    # Step 2: Deduplicate — keep best R² for each key
    deduped: dict[str, float] = {}
    deduped_weights: dict[str, float] = {}
    for key, entries in all_constants.items():
        best = max(entries, key=lambda e: e[1])  # highest R²
        deduped[key] = best[0]
        deduped_weights[key] = best[1]  # carry R² as weight

    n_after = len(deduped)

    # Step 3: Unify — keep groups with 2+ appearances (PSLQ cross-agent unification)
    # Also include keys that were deduplicated from multiple agents (same key, different agents)
    all_unified = unify_constants(deduped, tolerance=tolerance,
                                  weights=deduped_weights)
    multi_agent_keys = {key for key, entries in all_constants.items() if len(entries) >= 2}
    unified = [
        uc for uc in all_unified
        if len(uc.appearances) >= 2 or any(a in multi_agent_keys for a in uc.appearances)
    ]

    return CrossAgentUnificationResult(
        unified=unified,
        deduplicated_constants=deduped,
        n_before_dedup=n_before,
        n_after_dedup=n_after,
    )
