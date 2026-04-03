"""U2: Template extraction via anti-unification.

Anti-unification finds the most specific common generalization of two expressions.
Used to discover shared law templates across experiments.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from atlas.dsl.expr import Expr, Const, Var, BinOp, UnaryOp, NAryOp
from atlas.dsl.canonicalize import canonicalize
from atlas.dsl.serialize import to_str


@dataclass
class AntiUnifyResult:
    """Result of anti-unifying two expressions."""
    template: Expr
    holes: dict[str, tuple[Expr, Expr]]  # hole_name -> (binding_from_e1, binding_from_e2)


@dataclass
class TemplateResult:
    """A discovered law template."""
    template: Expr
    env_ids: list[str]
    bindings: dict[str, dict[str, Expr]]  # env_id -> {hole_name -> binding}
    savings: float


def anti_unify(e1: Expr, e2: Expr) -> AntiUnifyResult:
    """Compute the most specific common generalization of two expressions.

    Where e1 and e2 agree structurally, the template keeps that structure.
    Where they differ, a fresh hole variable is introduced.
    """
    holes: dict[str, tuple[Expr, Expr]] = {}
    counter = [0]  # mutable counter — no global state
    template = _anti_unify_impl(e1, e2, holes, counter)
    return AntiUnifyResult(template=template, holes=holes)


def _anti_unify_impl(e1: Expr, e2: Expr,
                     holes: dict[str, tuple[Expr, Expr]],
                     counter: list[int]) -> Expr:
    # If expressions are equal, return as-is
    if e1 == e2:
        return e1

    # Same node type and operator -> recurse
    if isinstance(e1, UnaryOp) and isinstance(e2, UnaryOp) and e1.op == e2.op:
        sub = _anti_unify_impl(e1.operand, e2.operand, holes, counter)
        return UnaryOp(e1.op, sub)

    if isinstance(e1, BinOp) and isinstance(e2, BinOp) and e1.op == e2.op:
        left = _anti_unify_impl(e1.left, e2.left, holes, counter)
        right = _anti_unify_impl(e1.right, e2.right, holes, counter)
        return BinOp(e1.op, left, right)

    if (isinstance(e1, NAryOp) and isinstance(e2, NAryOp) and
            e1.op == e2.op and len(e1.children) == len(e2.children)):
        children = [_anti_unify_impl(c1, c2, holes, counter)
                    for c1, c2 in zip(e1.children, e2.children)]
        return NAryOp(e1.op, children)

    # Different structure or values -> introduce a hole
    hole_name = f"_HOLE_{counter[0]}"
    counter[0] += 1
    holes[hole_name] = (e1, e2)
    return Var(hole_name)


def extract_templates(formulas: dict[str, Expr],
                      min_savings: float = 1.0) -> list[TemplateResult]:
    """Find shared templates across formulas from different experiments.

    Pairwise anti-unification, then check if templates provide MDL savings.
    """
    env_ids = list(formulas.keys())
    if len(env_ids) < 2:
        return []

    templates: list[TemplateResult] = []
    seen: set[str] = set()

    for i in range(len(env_ids)):
        for j in range(i + 1, len(env_ids)):
            e1 = canonicalize(formulas[env_ids[i]])
            e2 = canonicalize(formulas[env_ids[j]])
            result = anti_unify(e1, e2)

            template_key = to_str(result.template)
            if template_key in seen:
                continue
            seen.add(template_key)

            # Compute MDL savings
            template_size = result.template.size()
            bindings_size = sum(b[0].size() + b[1].size()
                                for b in result.holes.values())
            original_size = e1.size() + e2.size()
            unified_size = template_size + bindings_size
            savings = original_size - unified_size

            if savings >= min_savings and template_size >= 3:
                bindings = {
                    env_ids[i]: {h: b[0] for h, b in result.holes.items()},
                    env_ids[j]: {h: b[1] for h, b in result.holes.items()},
                }
                templates.append(TemplateResult(
                    template=result.template,
                    env_ids=[env_ids[i], env_ids[j]],
                    bindings=bindings,
                    savings=savings,
                ))

    templates.sort(key=lambda t: t.savings, reverse=True)
    return templates
