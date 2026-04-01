"""Tests for formula store."""
from atlas.sr.formula_store import FormulaStore, StoredFormula, _extract_constants
from atlas.dsl.expr import Const, Var, BinOp, UnaryOp
from atlas.dsl.operators import Op
from atlas.types import FitMetrics


def test_add_and_get():
    store = FormulaStore()
    expr = BinOp(Op.MUL, Var("knob_0"), Const(2.0))
    metrics = FitMetrics(r_squared=0.99, residual_var=0.01, mdl=5.0)
    store.add("ENV_01", expr, metrics)
    results = store.get("ENV_01")
    assert len(results) == 1
    assert results[0].expr == expr


def test_get_best():
    store = FormulaStore()
    e1 = BinOp(Op.ADD, Var("knob_0"), Const(1.0))
    e2 = BinOp(Op.MUL, Var("knob_0"), Const(2.0))
    store.add("ENV_01", e1, FitMetrics(r_squared=0.8, residual_var=0.2, mdl=4.0))
    store.add("ENV_01", e2, FitMetrics(r_squared=0.95, residual_var=0.05, mdl=5.0))
    best = store.get_best("ENV_01")
    assert best.expr == e2


def test_get_best_empty():
    store = FormulaStore()
    assert store.get_best("ENV_99") is None


def test_all_constants():
    store = FormulaStore()
    store.add("ENV_01", BinOp(Op.MUL, Var("k"), Const(6.626e-34)),
              FitMetrics(r_squared=0.99, residual_var=0.01, mdl=5.0))
    store.add("ENV_02", BinOp(Op.ADD, Var("k"), Const(2.998e8)),
              FitMetrics(r_squared=0.99, residual_var=0.01, mdl=4.0))
    constants = store.all_constants()
    assert 6.626e-34 in constants
    assert 2.998e8 in constants


def test_all_env_ids():
    store = FormulaStore()
    e = Const(1.0)
    store.add("ENV_01", e, FitMetrics(r_squared=0.5, residual_var=0.5, mdl=1.0))
    store.add("ENV_02", e, FitMetrics(r_squared=0.5, residual_var=0.5, mdl=1.0))
    assert store.all_env_ids() == {"ENV_01", "ENV_02"}


def test_pareto_front():
    store = FormulaStore()
    e1 = Var("knob_0")
    store.add("ENV_01", e1, FitMetrics(r_squared=0.5, residual_var=0.5, mdl=1.0))
    e2 = BinOp(Op.MUL, UnaryOp(Op.SIN, Var("knob_0")), Const(3.14))
    store.add("ENV_01", e2, FitMetrics(r_squared=0.99, residual_var=0.01, mdl=6.0))
    e3 = BinOp(Op.ADD, BinOp(Op.MUL, Var("knob_0"), Const(1.0)), Const(0.5))
    store.add("ENV_01", e3, FitMetrics(r_squared=0.7, residual_var=0.3, mdl=7.0))
    pareto = store.pareto_front("ENV_01")
    assert len(pareto) == 2
    pareto_exprs = {f.expr for f in pareto}
    assert e1 in pareto_exprs
    assert e2 in pareto_exprs


def test_extract_constants():
    expr = BinOp(Op.ADD, BinOp(Op.MUL, Var("x"), Const(3.14)), Const(2.71))
    constants = _extract_constants(expr)
    assert 3.14 in constants
    assert 2.71 in constants
