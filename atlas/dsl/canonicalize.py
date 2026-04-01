"""AST canonicalization: alpha-equivalence, commutativity, associativity,
identity elimination, constant folding."""
from __future__ import annotations

import math
import re
from collections.abc import Mapping

from atlas.dsl.expr import Expr, Const, Var, BinOp, UnaryOp, NAryOp
from atlas.dsl.operators import Op

_CANONICAL_VAR_RE = re.compile(r"^x_\d+$")


def canonicalize(expr: Expr) -> Expr:
    """Full canonicalization pipeline: alpha-rename -> simplify -> normalize."""
    expr = alpha_rename(expr)
    expr = _simplify(expr)
    return expr


def alpha_rename(expr: Expr) -> Expr:
    """Rename variables by order of first appearance (depth-first, left-to-right).

    Variables already in canonical form (x_0, x_1, ...) are left unchanged.
    """
    var_order: list[str] = []
    _collect_vars_in_order(expr, var_order)
    # If all variables are already in canonical x_N form, skip renaming
    if all(_CANONICAL_VAR_RE.match(v) for v in var_order):
        return expr
    mapping = {old: f"x_{i}" for i, old in enumerate(var_order)}
    return _apply_rename(expr, mapping)


def _collect_vars_in_order(expr: Expr, order: list[str]) -> None:
    if isinstance(expr, Var):
        if expr.name not in order:
            order.append(expr.name)
    elif isinstance(expr, Const):
        pass
    elif isinstance(expr, UnaryOp):
        _collect_vars_in_order(expr.operand, order)
    elif isinstance(expr, BinOp):
        _collect_vars_in_order(expr.left, order)
        _collect_vars_in_order(expr.right, order)
    elif isinstance(expr, NAryOp):
        for c in expr.children:
            _collect_vars_in_order(c, order)


def _apply_rename(expr: Expr, mapping: Mapping[str, str]) -> Expr:
    if isinstance(expr, Var):
        return Var(mapping.get(expr.name, expr.name))
    if isinstance(expr, Const):
        return expr
    if isinstance(expr, UnaryOp):
        return UnaryOp(expr.op, _apply_rename(expr.operand, mapping))
    if isinstance(expr, BinOp):
        return BinOp(
            expr.op,
            _apply_rename(expr.left, mapping),
            _apply_rename(expr.right, mapping),
        )
    if isinstance(expr, NAryOp):
        return NAryOp(expr.op, [_apply_rename(c, mapping) for c in expr.children])
    raise TypeError(f"Unknown expr type: {type(expr)}")


def _simplify(expr: Expr) -> Expr:
    """Bottom-up simplification: constant folding, identity elimination,
    associativity flattening, commutativity sorting."""
    if isinstance(expr, (Const, Var)):
        return expr

    if isinstance(expr, UnaryOp):
        operand = _simplify(expr.operand)
        # Constant folding for unary ops
        if isinstance(operand, Const):
            try:
                result = UnaryOp(expr.op, operand).evaluate({})
                if math.isfinite(result):
                    return Const(result)
            except (ValueError, OverflowError):
                pass
        return UnaryOp(expr.op, operand)

    if isinstance(expr, BinOp):
        left = _simplify(expr.left)
        right = _simplify(expr.right)

        # Constant folding
        if isinstance(left, Const) and isinstance(right, Const):
            try:
                result = BinOp(expr.op, left, right).evaluate({})
                if math.isfinite(result):
                    return Const(result)
            except (ValueError, OverflowError, ZeroDivisionError):
                pass

        # Identity elimination
        simplified = _eliminate_identity(expr.op, left, right)
        if simplified is not None:
            return simplified

        # Associativity flattening for commutative ops
        if expr.op.is_commutative:
            children = _flatten(expr.op, left, right)
            children = sorted(children, key=_sort_key)
            if len(children) == 2:
                return BinOp(expr.op, children[0], children[1])
            return NAryOp(expr.op, children)

        return BinOp(expr.op, left, right)

    if isinstance(expr, NAryOp):
        children = [_simplify(c) for c in expr.children]
        flat: list[Expr] = []
        for c in children:
            if isinstance(c, NAryOp) and c.op == expr.op:
                flat.extend(c.children)
            elif isinstance(c, BinOp) and c.op == expr.op and expr.op.is_commutative:
                flat.extend([c.left, c.right])
            else:
                flat.append(c)
        flat = sorted(flat, key=_sort_key)
        if len(flat) == 1:
            return flat[0]
        if len(flat) == 2:
            return BinOp(expr.op, flat[0], flat[1])
        return NAryOp(expr.op, flat)

    raise TypeError(f"Unknown expr type: {type(expr)}")


def _eliminate_identity(op: Op, left: Expr, right: Expr) -> Expr | None:
    """Remove identity elements: x+0=x, x*1=x, x^1=x."""
    if op == Op.ADD:
        if isinstance(right, Const) and right.value == 0.0:
            return left
        if isinstance(left, Const) and left.value == 0.0:
            return right
    elif op == Op.MUL:
        if isinstance(right, Const) and right.value == 1.0:
            return left
        if isinstance(left, Const) and left.value == 1.0:
            return right
    elif op == Op.POW:
        if isinstance(right, Const) and right.value == 1.0:
            return left
    return None


def _flatten(op: Op, left: Expr, right: Expr) -> list[Expr]:
    """Flatten associative binary ops into a list of children."""
    children: list[Expr] = []
    for node in (left, right):
        if isinstance(node, BinOp) and node.op == op:
            children.extend(_flatten(op, node.left, node.right))
        elif isinstance(node, NAryOp) and node.op == op:
            children.extend(node.children)
        else:
            children.append(node)
    return children


def _sort_key(expr: Expr) -> tuple:
    """Canonical ordering: Const < Var (by name) < UnaryOp < BinOp < NAryOp."""
    if isinstance(expr, Const):
        return (0, expr.value)
    if isinstance(expr, Var):
        return (1, expr.name)
    if isinstance(expr, UnaryOp):
        return (2, expr.op.value, _sort_key(expr.operand))
    if isinstance(expr, BinOp):
        return (3, expr.op.value, _sort_key(expr.left), _sort_key(expr.right))
    if isinstance(expr, NAryOp):
        return (4, expr.op.value, tuple(_sort_key(c) for c in expr.children))
    return (99,)
