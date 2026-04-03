"""U3: Type isomorphism detection and unification.

Compares DSLType objects from different experiments:
- Dimension must match exactly
- Constraint structures must be isomorphic up to alpha-equivalence
  (permutation of variable indices)
- Constants must agree within tolerance

Merge procedure:
  1. Group isomorphic types via greedy matching
  2. Merge into a single representative (canonical form)
  3. Compute compression savings: n_merged types replaced by 1 definition
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

import numpy as np

from atlas.rgde.type_builder import DSLType
from atlas.rgde.constraint_finder import Constraint


@dataclass
class TypeUnificationResult:
    unified_types: list[dict]   # each: {name, dimension, constraints, source_envs, ...}
    n_merges: int


# ---------------------------------------------------------------------------
# Constraint canonicalization
# ---------------------------------------------------------------------------

def _canonicalize_terms(terms: list[tuple[int, ...]], perm: dict[int, int]
                        ) -> list[tuple[int, ...]]:
    """Apply variable index permutation to terms and sort canonically.

    A "term" is a tuple of variable indices representing a monomial.
    E.g. ``(0, 1)`` = z_0 · z_1,  ``(2, 2)`` = z_2².

    Alpha-equivalence means relabelling variables: e.g. if perm = {0→1, 1→0, 2→2},
    then term (0, 0) becomes (1, 1) and term (0, 1) becomes (0, 1) [after sorting
    each term's indices].
    """
    permuted = []
    for term in terms:
        new_term = tuple(sorted(perm.get(idx, idx) for idx in term))
        permuted.append(new_term)
    return sorted(permuted)


def _canonical_constraint(c: Constraint) -> tuple:
    """Return a hashable canonical form of a constraint (for identity comparison).

    Canonical form: (degree, constraint_type, sorted_terms_tuple, rounded_constant).
    """
    sorted_terms = tuple(sorted(tuple(sorted(t)) for t in c.terms))
    return (c.degree, c.constraint_type, sorted_terms, round(c.constant, 6))


# ---------------------------------------------------------------------------
# Alpha-equivalence matching
# ---------------------------------------------------------------------------

def _try_alpha_match(
    constraints_a: list[Constraint],
    constraints_b: list[Constraint],
    dim: int,
    coeff_tol: float,
) -> bool:
    """Check if constraints_b can be matched to constraints_a under some
    permutation of variable indices 0..dim-1.

    For small dimensions (K ≤ 6), try all permutations.
    For larger dimensions, fall back to canonicalized comparison (no permutation
    search — structural matching only).
    """
    if len(constraints_a) != len(constraints_b):
        return False

    if not constraints_a:
        return True  # both empty

    # Pre-sort constraints by degree and term count for greedy matching
    ca_sorted = sorted(constraints_a, key=lambda c: (c.degree, len(c.terms)))
    cb_sorted = sorted(constraints_b, key=lambda c: (c.degree, len(c.terms)))

    # Quick check: degree profiles must match
    for a, b in zip(ca_sorted, cb_sorted):
        if a.degree != b.degree or len(a.terms) != len(b.terms):
            return False
        if a.constraint_type != b.constraint_type:
            return False

    # For dim ≤ 6, enumerate all permutations (max 720)
    if dim <= 6:
        for perm_tuple in permutations(range(dim)):
            perm = dict(enumerate(perm_tuple))
            if _match_under_permutation(ca_sorted, cb_sorted, perm, coeff_tol):
                return True
        return False

    # For dim > 6, fall back to identity permutation (exact structural match)
    identity = {i: i for i in range(dim)}
    return _match_under_permutation(ca_sorted, cb_sorted, identity, coeff_tol)


def _match_under_permutation(
    ca: list[Constraint],
    cb: list[Constraint],
    perm: dict[int, int],
    coeff_tol: float,
) -> bool:
    """Check if constraint list cb matches ca under the given variable permutation."""
    for a, b in zip(ca, cb):
        # Apply permutation to b's terms and canonicalize
        a_terms = _canonicalize_terms(a.terms, {i: i for i in range(100)})
        b_terms = _canonicalize_terms(b.terms, perm)

        if a_terms != b_terms:
            return False

        # Compare constants
        if abs(a.constant) > 1e-10 and abs(b.constant) > 1e-10:
            rel_diff = abs(a.constant - b.constant) / max(abs(a.constant), abs(b.constant))
            if rel_diff > coeff_tol:
                return False
        elif abs(a.constant - b.constant) > 1e-6:
            return False

    return True


# ---------------------------------------------------------------------------
# Public API: isomorphism detection
# ---------------------------------------------------------------------------

def are_types_isomorphic(t1: DSLType, t2: DSLType,
                         coeff_tol: float = 0.05) -> bool:
    """Check if two DSL types are structurally isomorphic.

    Criteria:
    1. Same dimension (exact integer match)
    2. Constraints are alpha-equivalent: there exists a permutation of
       variable indices {0..K-1} → {0..K-1} such that the constraint
       structures match (with coefficient tolerance for constants).

    This handles cases like:
    - ENV_07: z_0² + z_1² + z_2² ≤ 1  (terms: [(0,0), (1,1), (2,2)])
    - ENV_03: w_2² + w_0² + w_1² ≤ 1  (terms: [(2,2), (0,0), (1,1)])
    → Isomorphic under permutation {0→2, 1→0, 2→1}
    """
    if t1.dimension != t2.dimension:
        return False

    if len(t1.constraints) != len(t2.constraints):
        return False

    if len(t1.constraints) == 0 and len(t2.constraints) == 0:
        # Both unconstrained with same dim -> trivially isomorphic
        return True

    return _try_alpha_match(t1.constraints, t2.constraints, t1.dimension, coeff_tol)


# ---------------------------------------------------------------------------
# Public API: unification (grouping + merging)
# ---------------------------------------------------------------------------

def unify_types(types: list[DSLType],
                coeff_tol: float = 0.05) -> TypeUnificationResult:
    """Group isomorphic types and merge them.

    Merge procedure for each group:
      1. Pick the representative with the simplest encoding (lowest MDL)
      2. Collect all source environments
      3. Compute compression savings:
         savings = (n_merged - 1) * representative.mdl_cost()
         (n_merged types replaced by 1 definition + n_merged references)
      4. Canonicalize constraint representation

    Returns unified types with merged source environment lists.
    """
    if len(types) < 2:
        return TypeUnificationResult(unified_types=[], n_merges=0)

    # Greedy grouping (same as before but with alpha-equivalence)
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
        # Pick the representative with lowest MDL cost (simplest encoding)
        representative_idx = min(group, key=lambda idx: types[idx].mdl_cost())
        representative = types[representative_idx]
        source_envs = [types[idx].source_env for idx in group]

        # Canonicalize constraint strings for the unified type
        constraint_strs = []
        for c in representative.constraints:
            sorted_terms = sorted(tuple(sorted(t)) for t in c.terms)
            terms_str = " + ".join(
                "·".join(f"z_{k}" for k in term) for term in sorted_terms
            )
            op = "=" if c.constraint_type == "equality" else "≤"
            constraint_strs.append(f"{terms_str} {op} {c.constant:.4f}")

        # Compression savings: replacing n types with 1
        n_merged = len(group)
        type_mdl_cost = representative.mdl_cost()
        compression_savings = (n_merged - 1) * type_mdl_cost

        unified.append({
            "name": f"Unified_K{representative.dimension}",
            "dimension": representative.dimension,
            "constraints": constraint_strs,
            "source_envs": source_envs,
            "n_merged": n_merged,
            "compression_savings": compression_savings,
            "representative_mdl": type_mdl_cost,
        })

    return TypeUnificationResult(
        unified_types=unified,
        n_merges=sum(u["n_merged"] for u in unified),
    )
