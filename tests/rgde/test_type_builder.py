from atlas.rgde.type_builder import DSLType, build_type
from atlas.rgde.evaluator import evaluate_extension
from atlas.rgde.constraint_finder import Constraint
from atlas.dsl.expr import Var, BinOp, Const
from atlas.dsl.operators import Op
import numpy as np

def test_build_type_basic():
    encoder_formulas = {
        0: BinOp(Op.MUL, Var("knob_0"), Const(2.0)),
        1: BinOp(Op.ADD, Var("knob_1"), Const(0.5)),
    }
    constraints = [Constraint(
        coefficients=np.array([1.0, 1.0]), terms=[(0, 0), (1, 1)],
        degree=2, constant=1.0, residual=0.01, constraint_type="equality",
    )]
    t = build_type("ENV_07", encoder_formulas, constraints)
    assert isinstance(t, DSLType)
    assert t.name == "State_ENV_07"
    assert t.dimension == 2
    assert len(t.constraints) == 1

def test_build_type_no_constraints():
    t = build_type("ENV_01", {0: Var("knob_0")}, [])
    assert t.dimension == 1
    assert len(t.constraints) == 0

def test_evaluate_accepts():
    result = evaluate_extension(r2_before=0.3, r2_after=0.95, mdl_before=5.0, mdl_after=12.0, type_mdl_cost=7.0)
    assert result.accepted

def test_evaluate_rejects():
    result = evaluate_extension(r2_before=0.95, r2_after=0.96, mdl_before=5.0, mdl_after=20.0, type_mdl_cost=15.0)
    assert not result.accepted

def test_dsl_type_mdl_cost():
    t = build_type("ENV_01", {0: BinOp(Op.MUL, Var("k"), Const(2.0))}, [])
    assert t.mdl_cost() > 0
