"""Expression AST nodes."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from atlas.dsl.operators import Op

# Re-export Op for convenience: `from atlas.dsl.expr import Op`
__all__ = ["Expr", "Const", "Var", "BinOp", "UnaryOp", "NAryOp", "Op"]


def _const_encoding_cost(value: float) -> float:
    """Estimate the description length (in bits) of a floating-point constant.

    Small integers and simple fractions are cheap; constants with many
    significant digits are expensive.  The cost is:
        1 (node overhead) + log2(1 + significant_digits)
    where *significant_digits* is the number of decimal digits needed to
    represent the value (ignoring trailing zeros).
    """
    if value == 0.0:
        return 1.0
    # Count significant digits from the string representation
    s = f"{abs(value):.15g}"
    # Remove leading zeros, decimal point, exponent, and sign
    mantissa = s.split("e")[0].replace(".", "").lstrip("0").rstrip("0")
    n_digits = max(len(mantissa), 1)
    return 1.0 + math.log2(1.0 + n_digits)


class Expr:
    """Base class for all expression nodes."""

    def evaluate(self, env: Mapping[str, float]) -> float:
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError

    def mdl_cost(self) -> float:
        """Minimum description length accounting for constant encoding cost."""
        raise NotImplementedError

    def depth(self) -> int:
        raise NotImplementedError

    def variables(self) -> frozenset[str]:
        raise NotImplementedError


@dataclass(frozen=True)
class Const(Expr):
    value: float

    def evaluate(self, env: Mapping[str, float]) -> float:
        return self.value

    def size(self) -> int:
        return 1

    def mdl_cost(self) -> float:
        return _const_encoding_cost(self.value)

    def depth(self) -> int:
        return 1

    def variables(self) -> frozenset[str]:
        return frozenset()


@dataclass(frozen=True)
class Var(Expr):
    name: str

    def evaluate(self, env: Mapping[str, float]) -> float:
        return env[self.name]

    def size(self) -> int:
        return 1

    def mdl_cost(self) -> float:
        return 1.0

    def depth(self) -> int:
        return 1

    def variables(self) -> frozenset[str]:
        return frozenset({self.name})


@dataclass(frozen=True)
class BinOp(Expr):
    op: Op
    left: Expr
    right: Expr

    def evaluate(self, env: Mapping[str, float]) -> float:
        l = self.left.evaluate(env)
        r = self.right.evaluate(env)
        if self.op == Op.ADD:
            return l + r
        if self.op == Op.SUB:
            return l - r
        if self.op == Op.MUL:
            return l * r
        if self.op == Op.DIV:
            return l / r if r != 0 else math.nan
        if self.op == Op.POW:
            try:
                result = l ** r
            except (ValueError, OverflowError):
                return math.nan
            return result if isinstance(result, float) else math.nan
        raise ValueError(f"Unknown binary op: {self.op}")

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()

    def mdl_cost(self) -> float:
        return 1.0 + self.left.mdl_cost() + self.right.mdl_cost()

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def variables(self) -> frozenset[str]:
        return self.left.variables() | self.right.variables()


@dataclass(frozen=True)
class UnaryOp(Expr):
    op: Op
    operand: Expr

    def evaluate(self, env: Mapping[str, float]) -> float:
        v = self.operand.evaluate(env)
        if self.op == Op.SIN:
            return math.sin(v)
        if self.op == Op.COS:
            return math.cos(v)
        if self.op == Op.EXP:
            return math.exp(v) if v < 709 else math.inf
        if self.op == Op.LOG:
            return math.log(v) if v > 0 else math.nan
        if self.op == Op.NEG:
            return -v
        raise ValueError(f"Unknown unary op: {self.op}")

    def size(self) -> int:
        return 1 + self.operand.size()

    def mdl_cost(self) -> float:
        return 1.0 + self.operand.mdl_cost()

    def depth(self) -> int:
        return 1 + self.operand.depth()

    def variables(self) -> frozenset[str]:
        return self.operand.variables()


@dataclass(frozen=True)
class NAryOp(Expr):
    """N-ary operator node (used after associativity flattening)."""
    op: Op
    children: tuple[Expr, ...]

    def __init__(self, op: Op, children: list[Expr] | tuple[Expr, ...]):
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "children", tuple(children))

    def evaluate(self, env: Mapping[str, float]) -> float:
        values = [c.evaluate(env) for c in self.children]
        if self.op == Op.ADD:
            return sum(values)
        if self.op == Op.MUL:
            result = 1.0
            for v in values:
                result *= v
            return result
        raise ValueError(f"NAryOp only supports ADD/MUL, got {self.op}")

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def mdl_cost(self) -> float:
        return 1.0 + sum(c.mdl_cost() for c in self.children)

    def depth(self) -> int:
        return 1 + max(c.depth() for c in self.children)

    def variables(self) -> frozenset[str]:
        result: frozenset[str] = frozenset()
        for c in self.children:
            result = result | c.variables()
        return result
