"""Tests for AST canonicalization."""
from atlas.dsl.expr import Const, Var, BinOp, UnaryOp, NAryOp
from atlas.dsl.operators import Op
from atlas.dsl.canonicalize import canonicalize, alpha_rename


def test_alpha_rename_by_appearance_order():
    expr = BinOp(Op.ADD, Var("knob_2"), Var("knob_0"))
    renamed = alpha_rename(expr)
    assert renamed == BinOp(Op.ADD, Var("x_0"), Var("x_1"))


def test_alpha_rename_preserves_structure():
    expr = BinOp(Op.ADD, Var("knob_0"), Var("knob_0"))
    renamed = alpha_rename(expr)
    assert renamed == BinOp(Op.ADD, Var("x_0"), Var("x_0"))


def test_alpha_rename_nested():
    expr = BinOp(Op.MUL, UnaryOp(Op.SIN, Var("knob_1")), Var("knob_0"))
    renamed = alpha_rename(expr)
    expected = BinOp(Op.MUL, UnaryOp(Op.SIN, Var("x_0")), Var("x_1"))
    assert renamed == expected


def test_commutativity_sort():
    expr = BinOp(Op.ADD, Var("x_1"), Var("x_0"))
    result = canonicalize(expr)
    assert result == BinOp(Op.ADD, Var("x_0"), Var("x_1"))


def test_commutativity_mul():
    expr = BinOp(Op.MUL, Var("x_1"), Var("x_0"))
    result = canonicalize(expr)
    assert result == BinOp(Op.MUL, Var("x_0"), Var("x_1"))


def test_non_commutative_preserved():
    expr = BinOp(Op.SUB, Var("x_1"), Var("x_0"))
    result = canonicalize(expr)
    assert result == BinOp(Op.SUB, Var("x_1"), Var("x_0"))


def test_associativity_flattening():
    expr = BinOp(Op.ADD, BinOp(Op.ADD, Var("x_0"), Var("x_1")), Var("x_2"))
    result = canonicalize(expr)
    assert isinstance(result, NAryOp)
    assert result.op == Op.ADD
    assert len(result.children) == 3


def test_associativity_nested():
    expr = BinOp(Op.ADD, Var("x_0"), BinOp(Op.ADD, Var("x_1"), Var("x_2")))
    result = canonicalize(expr)
    assert isinstance(result, NAryOp)
    assert result.op == Op.ADD
    assert len(result.children) == 3


def test_identity_elimination_mul():
    expr = BinOp(Op.MUL, Var("x_0"), Const(1.0))
    result = canonicalize(expr)
    assert result == Var("x_0")


def test_identity_elimination_add():
    expr = BinOp(Op.ADD, Var("x_0"), Const(0.0))
    result = canonicalize(expr)
    assert result == Var("x_0")


def test_identity_elimination_pow():
    expr = BinOp(Op.POW, Var("x_0"), Const(1.0))
    result = canonicalize(expr)
    assert result == Var("x_0")


def test_constant_folding():
    expr = BinOp(Op.ADD, Const(2.0), Const(3.0))
    result = canonicalize(expr)
    assert result == Const(5.0)


def test_constant_folding_sin():
    expr = UnaryOp(Op.SIN, Const(0.0))
    result = canonicalize(expr)
    assert result == Const(0.0)


def test_full_canonicalization():
    # (knob_2 * 1.0) + (0.0 + knob_0)
    # -> alpha: (x_0 * 1.0) + (0.0 + x_1)
    # -> identity: x_0 + x_1
    # -> commutativity: already sorted
    expr = BinOp(
        Op.ADD,
        BinOp(Op.MUL, Var("knob_2"), Const(1.0)),
        BinOp(Op.ADD, Const(0.0), Var("knob_0")),
    )
    result = canonicalize(expr)
    assert result == BinOp(Op.ADD, Var("x_0"), Var("x_1"))


def test_canonicalize_two_equivalent_exprs():
    e1 = BinOp(Op.ADD, Var("y"), Var("x"))
    e2 = BinOp(Op.ADD, Var("a"), Var("b"))
    assert canonicalize(e1) == canonicalize(e2)
