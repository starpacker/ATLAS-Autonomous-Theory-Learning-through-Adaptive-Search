"""Tests for DSL state management."""
from atlas.agent.dsl_state import DSLState
from atlas.dsl.operators import Op, DSL_0
from atlas.dsl.expr import BinOp, UnaryOp, Var


def test_initial_state():
    state = DSLState()
    assert state.operators == DSL_0
    assert len(state.concepts) == 0
    assert len(state.extensions) == 0


def test_add_concept():
    state = DSLState()
    cos2 = BinOp(Op.MUL, UnaryOp(Op.COS, Var("x_0")), UnaryOp(Op.COS, Var("x_0")))
    state.add_concept("concept_cos2", cos2)
    assert "concept_cos2" in state.concepts
    assert state.concepts["concept_cos2"] == cos2


def test_add_extension():
    state = DSLState()
    state.add_extension(name="prob_mode", ext_type="prob_mode",
                        definition={"desc": "enable P(y|x) search"}, trigger="D1=stochastic")
    assert len(state.extensions) == 1
    assert state.extensions[0]["name"] == "prob_mode"


def test_mdl_cost():
    state = DSLState()
    cost_before = state.mdl_cost()
    cos2 = BinOp(Op.MUL, UnaryOp(Op.COS, Var("x_0")), UnaryOp(Op.COS, Var("x_0")))
    state.add_concept("concept_cos2", cos2)
    cost_after = state.mdl_cost()
    assert cost_after > cost_before


def test_snapshot_and_restore():
    state = DSLState()
    snap = state.snapshot()
    state.add_concept("c", Var("x_0"))
    assert len(state.concepts) == 1
    state.restore(snap)
    assert len(state.concepts) == 0
