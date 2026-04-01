"""Tests for expression AST nodes."""
import math
from atlas.dsl.expr import Const, Var, BinOp, UnaryOp, NAryOp, Op
from atlas.dsl.operators import DSL_0


def test_const_creation():
    c = Const(3.14)
    assert c.value == 3.14


def test_var_creation():
    v = Var("x_0")
    assert v.name == "x_0"


def test_binop_creation():
    expr = BinOp(Op.ADD, Var("x_0"), Const(1.0))
    assert expr.op == Op.ADD
    assert isinstance(expr.left, Var)
    assert isinstance(expr.right, Const)


def test_unaryop_creation():
    expr = UnaryOp(Op.SIN, Var("x_0"))
    assert expr.op == Op.SIN
    assert isinstance(expr.operand, Var)


def test_nary_creation():
    expr = NAryOp(Op.ADD, [Var("x_0"), Var("x_1"), Const(1.0)])
    assert len(expr.children) == 3


def test_expr_equality():
    a = BinOp(Op.ADD, Var("x_0"), Const(1.0))
    b = BinOp(Op.ADD, Var("x_0"), Const(1.0))
    assert a == b


def test_expr_inequality():
    a = BinOp(Op.ADD, Var("x_0"), Const(1.0))
    b = BinOp(Op.MUL, Var("x_0"), Const(1.0))
    assert a != b


def test_expr_hash():
    a = BinOp(Op.ADD, Var("x_0"), Const(1.0))
    b = BinOp(Op.ADD, Var("x_0"), Const(1.0))
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_eval_simple():
    # x_0 + 1.0
    expr = BinOp(Op.ADD, Var("x_0"), Const(1.0))
    result = expr.evaluate({"x_0": 2.0})
    assert result == 3.0


def test_eval_nested():
    # sin(x_0 * 3.14159)
    expr = UnaryOp(Op.SIN, BinOp(Op.MUL, Var("x_0"), Const(math.pi)))
    result = expr.evaluate({"x_0": 0.5})
    assert abs(result - 1.0) < 1e-10


def test_dsl0_operators():
    ops = DSL_0
    assert Op.ADD in ops
    assert Op.SIN in ops
    assert Op.EXP in ops
    assert Op.LOG in ops
    assert Op.POW in ops
    assert len(ops) == 10  # +, -, *, /, sin, cos, exp, log, ^, neg


def test_expr_size():
    # sin(x_0 + 1.0) -> size 4 (sin, +, x_0, 1.0)
    expr = UnaryOp(Op.SIN, BinOp(Op.ADD, Var("x_0"), Const(1.0)))
    assert expr.size() == 4


def test_expr_depth():
    # sin(x_0 + 1.0) -> depth 3
    expr = UnaryOp(Op.SIN, BinOp(Op.ADD, Var("x_0"), Const(1.0)))
    assert expr.depth() == 3
