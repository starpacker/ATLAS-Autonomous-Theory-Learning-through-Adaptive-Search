"""Concept extraction: frequent subexpression mining (DreamCoder-style)."""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
from atlas.dsl.expr import Expr, Const, Var, BinOp, UnaryOp, NAryOp
from atlas.dsl.canonicalize import canonicalize
from atlas.dsl.serialize import to_str


@dataclass
class Concept:
    name: str
    expr: Expr
    count: int
    savings: int


def extract_concepts(formulas: list[Expr], min_occurrences: int = 2,
                     min_size: int = 2) -> list[Concept]:
    subexpr_counts: Counter[str] = Counter()
    subexpr_map: dict[str, Expr] = {}
    for formula in formulas:
        seen_in_formula: set[str] = set()
        for sub in _all_subexprs(formula):
            if sub.size() < min_size or isinstance(sub, (Var, Const)):
                continue
            canon = canonicalize(sub)
            key = to_str(canon)
            if key not in seen_in_formula:
                seen_in_formula.add(key)
                subexpr_counts[key] += 1
                subexpr_map[key] = canon
    concepts = []
    concept_id = 0
    for key, count in subexpr_counts.items():
        if count < min_occurrences:
            continue
        expr = subexpr_map[key]
        size = expr.size()
        savings = count * size - size
        if savings > 0:
            concepts.append(Concept(name=f"concept_{concept_id}", expr=expr,
                                    count=count, savings=savings))
            concept_id += 1
    concepts.sort(key=lambda c: c.savings, reverse=True)
    return concepts


def _all_subexprs(expr: Expr) -> list[Expr]:
    result = [expr]
    if isinstance(expr, UnaryOp):
        result.extend(_all_subexprs(expr.operand))
    elif isinstance(expr, BinOp):
        result.extend(_all_subexprs(expr.left))
        result.extend(_all_subexprs(expr.right))
    elif isinstance(expr, NAryOp):
        for c in expr.children:
            result.extend(_all_subexprs(c))
    return result
