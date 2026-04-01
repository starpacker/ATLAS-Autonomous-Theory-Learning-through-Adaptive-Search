"""Operator definitions and DSL_0."""
from __future__ import annotations

from enum import Enum


class Op(Enum):
    """All operators available in the DSL."""
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    POW = "pow"
    SIN = "sin"
    COS = "cos"
    EXP = "exp"
    LOG = "log"
    NEG = "neg"

    @property
    def arity(self) -> int:
        if self in _UNARY_OPS:
            return 1
        return 2

    @property
    def is_commutative(self) -> bool:
        return self in (Op.ADD, Op.MUL)


_UNARY_OPS = frozenset({Op.SIN, Op.COS, Op.EXP, Op.LOG, Op.NEG})

DSL_0: frozenset[Op] = frozenset({
    Op.ADD, Op.SUB, Op.MUL, Op.DIV,
    Op.SIN, Op.COS, Op.EXP, Op.LOG,
    Op.POW, Op.NEG,
})
