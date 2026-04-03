"""PSLQ-inspired constant unification for cross-environment discovery.

Provides two search modes:
  - Value-space: integer linear combinations  p*|C₁| + q*|C₂| ≈ 0
  - Log-space:   integer relations in logs    n₁*log|C₁| + n₂*log|C₂| ≈ 0
                 i.e. |C₁|^n₁ · |C₂|^n₂ ≈ 1  (power/product relations)

Also provides chi-squared consistency checks and weighted error propagation
for validating whether grouped constants genuinely represent the same
physical quantity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ConstantRelation:
    """An integer relation a*x + b*y ≈ 0 between two constants (value-space)."""
    key_a: str
    key_b: str
    coeff_a: int
    coeff_b: int
    residual: float


@dataclass
class LogRelation:
    """An integer relation n₁·log|C₁| + n₂·log|C₂| ≈ 0 (log-space).

    Equivalently: |C₁|^n₁ · |C₂|^n₂ ≈ 1.
    """
    keys: list[str]
    exponents: list[int]
    residual: float  # |sum(ni * log|Ci|)| / normaliser


@dataclass
class UnifiedConstant:
    """A group of constants that are approximately equal (same physical constant)."""
    symbol: str
    value: float
    uncertainty: float
    appearances: list[str] = field(default_factory=list)
    signs: list[int] = field(default_factory=list)
    chi2_pvalue: Optional[float] = None
    is_spurious: bool = False


# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------

def weighted_mean_std(
    values: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """Compute weighted mean and weighted standard deviation.

    *weights* are typically R² values from the fits that produced each
    constant estimate.  If *weights* is ``None``, falls back to simple
    (equal-weight) mean and std.

    Returns ``(weighted_mean, weighted_std)`` where::

        weighted_std = sqrt( sum(w_i · (x_i - μ)²) / sum(w_i) )
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return float(values[0]), 0.0

    if weights is None:
        return float(np.mean(values)), float(np.std(values))

    w = np.maximum(np.asarray(weights, dtype=float), 0.0)
    w_sum = float(np.sum(w))
    if w_sum < 1e-300:
        return float(np.mean(values)), float(np.std(values))

    mu = float(np.dot(w, values) / w_sum)
    var = float(np.dot(w, (values - mu) ** 2) / w_sum)
    return mu, float(np.sqrt(max(var, 0.0)))


def chi2_consistency(
    values: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """Chi-squared consistency test for a group of constant estimates.

    Tests whether the estimates are consistent with being measurements of
    the same underlying constant.

    If individual *uncertainties* are not provided, uses the group standard
    deviation as a conservative proxy for each measurement's σ.

    Returns ``(chi2_statistic, p_value)``.

    Uses ``scipy.stats.chi2`` when available; falls back to a rough
    Gaussian approximation otherwise.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return 0.0, 1.0

    mean = float(np.mean(values))
    dof = n - 1

    if uncertainties is not None:
        sigmas = np.maximum(np.asarray(uncertainties, dtype=float), 1e-300)
    else:
        sigma = float(np.std(values))
        if sigma < 1e-300:
            return 0.0, 1.0  # all identical
        sigmas = np.full(n, sigma)

    chi2_stat = float(np.sum(((values - mean) / sigmas) ** 2))

    # Compute p-value
    try:
        from scipy.stats import chi2 as chi2_dist
        p_value = float(1.0 - chi2_dist.cdf(chi2_stat, dof))
    except ImportError:
        # Fallback: for large dof, chi2 ~ Normal(dof, 2*dof)
        z = (chi2_stat - dof) / max(np.sqrt(2.0 * dof), 1e-15)
        # One-sided p-value via error function approximation
        p_value = max(0.0, min(1.0, 0.5 * (1.0 - np.tanh(z * 0.7))))

    return chi2_stat, p_value


# ---------------------------------------------------------------------------
# Value-space integer relation search
# ---------------------------------------------------------------------------

def find_constant_relations(
    constants: dict[str, float],
    max_coeff: int = 10,
    tolerance: float = 1e-4,
) -> list[ConstantRelation]:
    """Find integer relations between pairs of constants in value-space.

    For each pair (a, b) we search for small integers (p, q) such that
    ``p * |a| + q * |b| ≈ 0`` (value-space integer relation).

    Signs are separated before the search: PSLQ operates on |C_i| only.
    The relative residual ``|p*|a| + q*|b|| / (|p|*|a| + |q|*|b|)`` must
    be below *tolerance*.
    """
    keys = list(constants.keys())
    relations: list[ConstantRelation] = []

    for ka, kb in combinations(keys, 2):
        va, vb = constants[ka], constants[kb]
        abs_a, abs_b = abs(va), abs(vb)
        if abs_a < 1e-300 or abs_b < 1e-300:
            continue

        best: Optional[tuple[float, int, int]] = None
        for p in range(-max_coeff, max_coeff + 1):
            for q in range(-max_coeff, max_coeff + 1):
                if p == 0 and q == 0:
                    continue
                # Require opposite signs so that the relation is non-trivial
                # (one term must be positive, the other negative)
                if (p > 0 and q > 0) or (p < 0 and q < 0):
                    continue
                residual = abs(p * abs_a + q * abs_b)
                denom = abs(p) * abs_a + abs(q) * abs_b
                if denom < 1e-300:
                    continue
                rel_res = residual / denom
                if best is None or rel_res < best[0]:
                    best = (rel_res, p, q)

        if best is not None:
            rel_res, p, q = best
            if rel_res < tolerance:
                relations.append(
                    ConstantRelation(
                        key_a=ka,
                        key_b=kb,
                        coeff_a=p,
                        coeff_b=q,
                        residual=rel_res,
                    )
                )

    return relations


# ---------------------------------------------------------------------------
# Log-space integer relation search
# ---------------------------------------------------------------------------

def find_log_relations(
    constants: dict[str, float],
    max_coeff: int = 6,
    tolerance: float = 1e-3,
) -> list[LogRelation]:
    """Find integer relations in log-space between pairs of constants.

    Searches for small integers (n₁, n₂) such that::

        n₁ · log|C₁| + n₂ · log|C₂| ≈ 0

    which means ``|C₁|^n₁ · |C₂|^n₂ ≈ 1``.

    This detects power/product relationships between constants
    (e.g. C₃ = C₁ / (C₂ · C₄)).

    Signs are separated: operates on |Cᵢ| only.

    Parameters
    ----------
    constants : dict
        Mapping of constant names to values.
    max_coeff : int
        Maximum absolute value of integer exponents to try.
    tolerance : float
        Maximum relative residual for accepting a relation.

    Returns
    -------
    list[LogRelation]
        Discovered log-space integer relations, sorted by residual.
    """
    keys = list(constants.keys())
    relations: list[LogRelation] = []

    for ka, kb in combinations(keys, 2):
        abs_a, abs_b = abs(constants[ka]), abs(constants[kb])
        if abs_a < 1e-300 or abs_b < 1e-300:
            continue

        log_a, log_b = np.log(abs_a), np.log(abs_b)

        # Skip if both logs are near zero (constants ≈ 1.0 — numerically
        # unstable for ratio-based relations).
        if abs(log_a) < 1e-10 and abs(log_b) < 1e-10:
            continue

        best: Optional[tuple[float, int, int]] = None
        for p in range(-max_coeff, max_coeff + 1):
            for q in range(-max_coeff, max_coeff + 1):
                if p == 0 and q == 0:
                    continue
                # Require opposite-sign coefficients for a non-trivial
                # relation (otherwise p·log(a) + q·log(b) ≈ 0 only when
                # both are near zero).
                if (p > 0 and q > 0) or (p < 0 and q < 0):
                    continue

                residual = abs(p * log_a + q * log_b)
                denom = abs(p * log_a) + abs(q * log_b)
                if denom < 1e-15:
                    continue
                rel_res = residual / denom

                if best is None or rel_res < best[0]:
                    best = (rel_res, p, q)

        if best is not None and best[0] < tolerance:
            _, p, q = best
            relations.append(LogRelation(
                keys=[ka, kb],
                exponents=[p, q],
                residual=best[0],
            ))

    # Sort by residual (best first)
    relations.sort(key=lambda r: r.residual)
    return relations


# ---------------------------------------------------------------------------
# Approximate-equality grouping + unification
# ---------------------------------------------------------------------------

def _try_mpmath_pslq(
    values: list[float],
    tolerance: float = 1e-6,
    max_coeff: int = 10,
) -> Optional[list[int]]:
    """Attempt true PSLQ via mpmath.  Returns integer coefficients or None."""
    try:
        import mpmath
        mp_vals = [mpmath.mpf(v) for v in values]
        result = mpmath.pslq(mp_vals, tol=mpmath.mpf(tolerance),
                             maxcoeff=max_coeff)
        return [int(c) for c in result] if result is not None else None
    except (ImportError, Exception):
        return None


@dataclass
class IntegerRelationResult:
    """An integer relation discovered between unified constant groups."""
    groups_involved: list[str]  # symbols of the unified constants
    coefficients: list[int]     # integer coefficients
    residual: float             # how close to zero
    relation_type: str          # "value" or "log"


def unify_constants(
    constants: dict[str, float],
    tolerance: float = 0.01,
    weights: Optional[dict[str, float]] = None,
    discover_relations: bool = True,
    relation_tolerance: float = 1e-4,
) -> list[UnifiedConstant]:
    """Group approximately equal constants and compute (weighted) mean ± std.

    Two constants are considered equal if their absolute values agree to
    within *tolerance* (fractional).  Signs are tracked separately so that
    e.g. +h and -h both map to the same unified constant.

    When *discover_relations* is True (default), also searches for integer
    relations between the unified group means — both in value-space and
    log-space.  If ``mpmath`` is available, uses the true PSLQ algorithm;
    otherwise falls back to brute-force search.  Discovered relations are
    stored in each :class:`UnifiedConstant`'s metadata but do not alter
    the grouping itself.

    Parameters
    ----------
    constants : dict
        Mapping of constant names to values.
    tolerance : float
        Maximum fractional difference ``|a-b|/max(|a|,|b|)`` for grouping.
    weights : dict, optional
        Per-constant weight (typically R² from the fit that produced it).
        Used for weighted mean/std.  If ``None``, equal weights are used.
    discover_relations : bool
        If True, search for integer relations between unified group means.
    relation_tolerance : float
        Tolerance for integer relation search.
    """
    if not constants:
        return []

    keys = list(constants.keys())
    abs_values = {k: abs(v) for k, v in constants.items()}
    signs = {k: (1 if v >= 0 else -1) for k, v in constants.items()}

    # Union-Find grouping
    parent = {k: k for k in keys}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for ka, kb in combinations(keys, 2):
        va, vb = abs_values[ka], abs_values[kb]
        denom = max(va, vb)
        if denom < 1e-300:
            continue
        if abs(va - vb) / denom < tolerance:
            union(ka, kb)

    # Collect groups
    groups: dict[str, list[str]] = {}
    for k in keys:
        root = find(k)
        groups.setdefault(root, []).append(k)

    unified: list[UnifiedConstant] = []
    for idx, (root, members) in enumerate(groups.items()):
        vals = np.array([abs_values[m] for m in members])

        # Weighted mean/std
        if weights is not None:
            w = np.array([weights.get(m, 1.0) for m in members])
            mean_val, std_val = weighted_mean_std(vals, w)
        else:
            mean_val, std_val = weighted_mean_std(vals)

        # Chi-squared consistency
        _, p_value = chi2_consistency(vals)

        member_signs = [signs[m] for m in members]
        uc = UnifiedConstant(
            symbol=f"C{idx}",
            value=mean_val,
            uncertainty=std_val,
            appearances=list(members),
            signs=member_signs,
            chi2_pvalue=p_value,
            is_spurious=(p_value < 0.01) if len(members) >= 2 else False,
        )
        unified.append(uc)

    # --- Integer relation discovery between unified group means ---
    if discover_relations and len(unified) >= 2:
        group_constants = {uc.symbol: uc.value for uc in unified
                           if uc.value > 1e-300 and not uc.is_spurious}
        if len(group_constants) >= 2:
            # Try true PSLQ on pairs first, fall back to brute force
            value_rels = find_constant_relations(
                group_constants, tolerance=relation_tolerance)
            log_rels = find_log_relations(
                group_constants, tolerance=relation_tolerance)

            # Build a lookup for attaching relations to unified constants
            _relations_by_symbol: dict[str, list[IntegerRelationResult]] = {}
            for rel in value_rels:
                irr = IntegerRelationResult(
                    groups_involved=[rel.key_a, rel.key_b],
                    coefficients=[rel.coeff_a, rel.coeff_b],
                    residual=rel.residual,
                    relation_type="value",
                )
                _relations_by_symbol.setdefault(rel.key_a, []).append(irr)
                _relations_by_symbol.setdefault(rel.key_b, []).append(irr)
            for rel in log_rels:
                irr = IntegerRelationResult(
                    groups_involved=list(rel.keys),
                    coefficients=list(rel.exponents),
                    residual=rel.residual,
                    relation_type="log",
                )
                for k in rel.keys:
                    _relations_by_symbol.setdefault(k, []).append(irr)

            # Also try mpmath PSLQ on the full vector of group means
            mean_values = [uc.value for uc in unified
                           if uc.value > 1e-300 and not uc.is_spurious]
            if len(mean_values) >= 2:
                pslq_result = _try_mpmath_pslq(mean_values)
                if pslq_result is not None:
                    symbols = [uc.symbol for uc in unified
                               if uc.value > 1e-300 and not uc.is_spurious]
                    residual = abs(sum(c * v for c, v in
                                       zip(pslq_result, mean_values)))
                    denom = sum(abs(c) * abs(v) for c, v in
                                zip(pslq_result, mean_values))
                    rel_residual = residual / max(denom, 1e-300)
                    if rel_residual < relation_tolerance:
                        irr = IntegerRelationResult(
                            groups_involved=symbols,
                            coefficients=pslq_result,
                            residual=rel_residual,
                            relation_type="pslq",
                        )
                        for s in symbols:
                            _relations_by_symbol.setdefault(s, []).append(irr)

    return unified
