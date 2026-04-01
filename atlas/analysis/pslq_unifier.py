"""PSLQ-inspired constant unification for cross-environment discovery."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np


@dataclass
class ConstantRelation:
    """An integer relation a*x + b*y ≈ 0 between two constants."""
    key_a: str
    key_b: str
    coeff_a: int
    coeff_b: int
    residual: float


@dataclass
class UnifiedConstant:
    """A group of constants that are approximately equal (same physical constant)."""
    symbol: str
    value: float
    uncertainty: float
    appearances: list[str] = field(default_factory=list)
    signs: list[int] = field(default_factory=list)


def find_constant_relations(
    constants: dict[str, float],
    max_coeff: int = 10,
    tolerance: float = 1e-4,
) -> list[ConstantRelation]:
    """Find integer relations between pairs of constants.

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


def unify_constants(
    constants: dict[str, float],
    tolerance: float = 0.01,
) -> list[UnifiedConstant]:
    """Group approximately equal constants and compute mean + std.

    Two constants are considered equal if their absolute values agree to
    within *tolerance* (fractional).  Signs are tracked separately so that
    e.g. +h and -h both map to the same unified constant.
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
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals)) if len(vals) > 1 else 0.0
        member_signs = [signs[m] for m in members]
        uc = UnifiedConstant(
            symbol=f"C{idx}",
            value=mean_val,
            uncertainty=std_val,
            appearances=list(members),
            signs=member_signs,
        )
        unified.append(uc)

    return unified
