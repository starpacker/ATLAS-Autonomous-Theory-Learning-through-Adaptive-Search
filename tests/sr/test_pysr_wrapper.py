"""Tests for PySR wrapper."""
import pytest
import numpy as np
from atlas.sr.pysr_wrapper import SRConfig, pysr_expr_to_atlas, SRResult
from atlas.dsl.expr import Const, Var, BinOp, UnaryOp, Expr
from atlas.dsl.operators import Op, DSL_0


def test_sr_config_defaults():
    cfg = SRConfig()
    assert cfg.niterations == 40
    assert cfg.populations == 15
    assert cfg.maxsize == 25
    assert set(cfg.binary_operators) == {"+", "-", "*", "/", "^"}
    assert set(cfg.unary_operators) == {"sin", "cos", "exp", "log", "neg"}


def test_sr_config_from_dsl():
    cfg = SRConfig.from_dsl(DSL_0)
    assert "+" in cfg.binary_operators
    assert "sin" in cfg.unary_operators


def test_sr_result_structure():
    result = SRResult(formulas=[], best_formula=None, best_r_squared=-1.0,
                      best_mdl=float("inf"), converged=False)
    assert not result.converged


def test_convert_simple_addition():
    expr = pysr_expr_to_atlas("x0 + 1.5", var_names=["knob_0"])
    assert isinstance(expr, Expr)
    assert abs(expr.evaluate({"knob_0": 2.0}) - 3.5) < 1e-10


def test_convert_nested():
    expr = pysr_expr_to_atlas("sin(x0 * 3.14159)", var_names=["knob_0"])
    assert abs(expr.evaluate({"knob_0": 0.5}) - 1.0) < 1e-4


def test_convert_with_two_vars():
    expr = pysr_expr_to_atlas("x0 * 2.5 + x1", var_names=["knob_0", "knob_1"])
    assert abs(expr.evaluate({"knob_0": 1.0, "knob_1": 3.0}) - 5.5) < 1e-10


def test_convert_power():
    expr = pysr_expr_to_atlas("x0 ^ 2.0", var_names=["knob_0"])
    assert abs(expr.evaluate({"knob_0": 3.0}) - 9.0) < 1e-10


def test_convert_cos():
    expr = pysr_expr_to_atlas("cos(x0)", var_names=["knob_0"])
    assert abs(expr.evaluate({"knob_0": 0.0}) - 1.0) < 1e-10


def test_convert_parenthesized():
    expr = pysr_expr_to_atlas("(x0 + 1.0) * 2.0", var_names=["knob_0"])
    assert abs(expr.evaluate({"knob_0": 3.0}) - 8.0) < 1e-10


def test_convert_negative_constant():
    expr = pysr_expr_to_atlas("x0 + -3.5", var_names=["knob_0"])
    assert abs(expr.evaluate({"knob_0": 5.0}) - 1.5) < 1e-10


def test_sr_config_stability_fields():
    """New stability fields: timeout, procs, deterministic."""
    cfg = SRConfig()
    assert cfg.timeout_in_seconds == 300
    assert cfg.procs == -1  # -1 = don't pass to PySR
    assert cfg.deterministic is False


def test_sr_config_custom_stability():
    cfg = SRConfig(timeout_in_seconds=60, procs=1, deterministic=False)
    assert cfg.timeout_in_seconds == 60
    assert cfg.procs == 1
    assert cfg.deterministic is False
