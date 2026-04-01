"""U3: Type isomorphism detection and unification.

Compares DSLType objects from different experiments:
- Dimension must match exactly
- Constraint structures must be isomorphic (alpha-equivalent)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from atlas.rgde.type_builder import DSLType


@dataclass
class TypeUnificationResult:
    unified_types: list[dict]   # each: {name, dimension, constraints, source_envs}
    n_merges: int


def are_types_isomorphic(t1: DSLType, t2: DSLType,
                         coeff_tol: float = 0.05) -> bool:
    """Check if two DSL types are structurally isomorphic.

    Criteria:
    1. Same dimension (exact integer match)
    2. Same number of constraints
    3. Constraints have matching degree and structure (with coefficient tolerance)
    """
    if t1.dimension != t2.dimension:
        return False

    if len(t1.constraints) != len(t2.constraints):
        return False

    if len(t1.constraints) == 0 and len(t2.constraints) == 0:
        # Both unconstrained with same dim -> trivially isomorphic
        return True

    # Match constraints pairwise (greedy, by degree then structure)
    c1_sorted = sorted(t1.constraints, key=lambda c: (c.degree, len(c.terms)))
    c2_sorted = sorted(t2.constraints, key=lambda c: (c.degree, len(c.terms)))

    for a, b in zip(c1_sorted, c2_sorted):
        if a.degree != b.degree:
            return False
        if len(a.terms) != len(b.terms):
            return False
        # Compare constraint types
        if a.constraint_type != b.constraint_type:
            return False
        # Compare term structure (sorted)
        a_terms = sorted(a.terms)
        b_terms = sorted(b.terms)
        if a_terms != b_terms:
            return False
        # Compare constant value
        if abs(a.constant) > 1e-10 and abs(b.constant) > 1e-10:
            rel_diff = abs(a.constant - b.constant) / max(abs(a.constant), abs(b.constant))
            if rel_diff > coeff_tol:
                return False

    return True


def unify_types(types: list[DSLType],
                coeff_tol: float = 0.05) -> TypeUnificationResult:
    """Group isomorphic types and merge them.

    Returns unified types with merged source environment lists.
    """
    if len(types) < 2:
        return TypeUnificationResult(unified_types=[], n_merges=0)

    # Union-find grouping
    groups: list[list[int]] = []
    assigned = [False] * len(types)

    for i in range(len(types)):
        if assigned[i]:
            continue
        group = [i]
        assigned[i] = True
        for j in range(i + 1, len(types)):
            if assigned[j]:
                continue
            if are_types_isomorphic(types[i], types[j], coeff_tol):
                group.append(j)
                assigned[j] = True
        if len(group) >= 2:
            groups.append(group)

    unified = []
    for group in groups:
        representative = types[group[0]]
        source_envs = [types[idx].source_env for idx in group]
        constraint_strs = []
        for c in representative.constraints:
            constraint_strs.append(f"degree={c.degree}, terms={c.terms}, const={c.constant:.4f}")

        unified.append({
            "name": f"Unified_K{representative.dimension}",
            "dimension": representative.dimension,
            "constraints": constraint_strs,
            "source_envs": source_envs,
            "n_merged": len(group),
        })

    return TypeUnificationResult(unified_types=unified, n_merges=len(groups))
