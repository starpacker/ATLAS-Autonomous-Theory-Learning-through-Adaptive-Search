"""Tests for type isomorphism detection and unification."""
import numpy as np
from atlas.unifier.type_unifier import (
    are_types_isomorphic, unify_types, TypeUnificationResult,
)
from atlas.rgde.type_builder import DSLType
from atlas.rgde.constraint_finder import Constraint
from atlas.dsl.expr import Var, BinOp, Const
from atlas.dsl.operators import Op


def _make_sphere_type(env_id: str, dim: int = 3) -> DSLType:
    encoding = {i: Var(f"knob_{i}") for i in range(dim)}
    terms = [(i, i) for i in range(dim)]
    constraint = Constraint(
        coefficients=np.ones(dim),
        terms=terms, degree=2, constant=1.0,
        residual=0.01, constraint_type="equality",
    )
    return DSLType(name=f"State_{env_id}", dimension=dim,
                   encoding=encoding, constraints=[constraint],
                   source_env=env_id)


def test_identical_types_isomorphic():
    t1 = _make_sphere_type("ENV_07", dim=3)
    t2 = _make_sphere_type("ENV_03", dim=3)
    assert are_types_isomorphic(t1, t2)


def test_different_dim_not_isomorphic():
    t1 = _make_sphere_type("ENV_07", dim=3)
    t2 = _make_sphere_type("ENV_03", dim=2)
    assert not are_types_isomorphic(t1, t2)


def test_no_constraint_vs_constraint():
    t1 = _make_sphere_type("ENV_07", dim=3)
    t2 = DSLType(name="State_ENV_01", dimension=3,
                 encoding={0: Var("k0"), 1: Var("k1"), 2: Var("k2")},
                 constraints=[], source_env="ENV_01")
    assert not are_types_isomorphic(t1, t2)


def test_unify_types():
    t1 = _make_sphere_type("ENV_07")
    t2 = _make_sphere_type("ENV_03")
    result = unify_types([t1, t2])
    assert isinstance(result, TypeUnificationResult)
    assert len(result.unified_types) == 1
    assert len(result.unified_types[0]["source_envs"]) == 2


def test_unify_no_isomorphic():
    t1 = _make_sphere_type("ENV_07", dim=3)
    t2 = _make_sphere_type("ENV_03", dim=2)
    result = unify_types([t1, t2])
    assert len(result.unified_types) == 0
