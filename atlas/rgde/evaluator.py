"""RGDE Step 4f: Pareto evaluator — accept/reject DSL extensions via R² + MDL."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    accepted: bool
    delta_r2: float
    delta_mdl: float
    pareto_efficient: bool
    reason: str


def evaluate_extension(r2_before: float, r2_after: float,
                       mdl_before: float, mdl_after: float,
                       type_mdl_cost: float,
                       min_r2_improvement: float = 0.1) -> EvaluationResult:
    delta_r2 = r2_after - r2_before
    total_mdl_after = mdl_after + type_mdl_cost
    delta_mdl = total_mdl_after - mdl_before
    if delta_r2 < min_r2_improvement:
        return EvaluationResult(False, delta_r2, delta_mdl, False,
                                f"R² improvement {delta_r2:.4f} < {min_r2_improvement}")
    efficiency = delta_r2 / max(delta_mdl, 0.1)
    if efficiency > 0.01:
        return EvaluationResult(True, delta_r2, delta_mdl, True,
                                f"Accepted: R² +{delta_r2:.4f}, eff {efficiency:.4f}")
    return EvaluationResult(False, delta_r2, delta_mdl, False,
                            f"Rejected: eff {efficiency:.4f} too low")
