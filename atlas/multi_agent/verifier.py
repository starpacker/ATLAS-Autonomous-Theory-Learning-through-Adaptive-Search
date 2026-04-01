"""Experiment-centric verification with global MDL criterion.

Evaluates a proposed DSL extension by checking its MDL impact across all
experiments. Uses statistical significance (pooled noise) to filter spurious gains.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class VerificationResult:
    """Result of global MDL verification for a proposal."""
    delta_total_mdl: float
    pooled_noise: float
    per_env_results: dict[str, dict]
    should_adopt: bool
    reason: str


def compute_global_mdl_delta(per_env_deltas: dict[str, dict]) -> VerificationResult:
    """Compute global MDL delta from per-experiment results.

    Args:
        per_env_deltas: {env_id: {"mu": mean_delta_mdl, "sigma": std_delta_mdl}}

    Returns:
        VerificationResult with adoption decision
    """
    if not per_env_deltas:
        return VerificationResult(
            delta_total_mdl=0.0, pooled_noise=float("inf"),
            per_env_results={}, should_adopt=False,
            reason="No experiment data"
        )

    delta_total = sum(d["mu"] for d in per_env_deltas.values())
    pooled_variance = sum(d["sigma"] ** 2 for d in per_env_deltas.values())
    pooled_noise = math.sqrt(pooled_variance) if pooled_variance > 0 else 0.0

    significant = is_statistically_significant(delta_total, pooled_noise)
    should_adopt = delta_total < 0 and significant

    if should_adopt:
        reason = f"Global MDL decreased by {abs(delta_total):.2f} (noise={pooled_noise:.2f})"
    elif delta_total >= 0:
        reason = f"Global MDL increased by {delta_total:.2f}"
    else:
        reason = f"MDL decrease {abs(delta_total):.2f} not significant (noise={pooled_noise:.2f})"

    return VerificationResult(
        delta_total_mdl=delta_total,
        pooled_noise=pooled_noise,
        per_env_results=dict(per_env_deltas),
        should_adopt=should_adopt,
        reason=reason,
    )


def is_statistically_significant(delta_total: float, pooled_noise: float,
                                  threshold: float = 2.0) -> bool:
    """Check if delta_total exceeds threshold * pooled_noise."""
    if pooled_noise <= 0:
        return delta_total < 0
    return abs(delta_total) > threshold * pooled_noise
