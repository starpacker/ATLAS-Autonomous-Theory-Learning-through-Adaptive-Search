"""Tests for expression serialization/deserialization."""
from atlas.dsl.expr import Const, Var, BinOp, UnaryOp, NAryOp
from atlas.dsl.operators import Op
from atlas.dsl.serialize import to_str, from_str, to_dict, from_dict


def test_const_roundtrip():
    expr = Const(3.14)
    assert from_str(to_str(expr)) == expr


def test_var_roundtrip():
    expr = Var("x_0")
    assert from_str(to_str(expr)) == expr


def test_binop_roundtrip():
    expr = BinOp(Op.MUL, Var("x_0"), Const(2.0))
    assert from_str(to_str(expr)) == expr


def test_unaryop_roundtrip():
    expr = UnaryOp(Op.SIN, Var("x_0"))
    assert from_str(to_str(expr)) == expr


def test_nary_roundtrip():
    expr = NAryOp(Op.ADD, [Var("x_0"), Var("x_1"), Const(1.0)])
    assert from_str(to_str(expr)) == expr


def test_nested_roundtrip():
    expr = BinOp(
        Op.ADD,
        UnaryOp(Op.SIN, BinOp(Op.MUL, Var("x_0"), Const(3.14))),
        UnaryOp(Op.COS, Var("x_1")),
    )
    assert from_str(to_str(expr)) == expr


def test_to_str_readable():
    expr = BinOp(Op.ADD, Var("x_0"), Const(1.0))
    s = to_str(expr)
    assert "add" in s or "+" in s


def test_dict_roundtrip():
    expr = BinOp(
        Op.ADD,
        UnaryOp(Op.SIN, Var("x_0")),
        Const(1.0),
    )
    d = to_dict(expr)
    assert isinstance(d, dict)
    assert from_dict(d) == expr
