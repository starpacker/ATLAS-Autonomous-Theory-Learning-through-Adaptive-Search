"""Tests for type isomorphism detection and unification."""
import numpy as np
from atlas.unifier.type_unifier import (
    are_types_isomorphic, unify_types, TypeUnificationResult,
    _canonicalize_terms, _try_alpha_match,
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


def _make_sphere_type_permuted(env_id: str) -> DSLType:
    """Sphere constraint with permuted variable indices: z_2² + z_0² + z_1² ≤ 1."""
    encoding = {i: Var(f"knob_{i}") for i in range(3)}
    # Same sphere constraint but with indices in different order
    terms = [(2, 2), (0, 0), (1, 1)]
    constraint = Constraint(
        coefficients=np.ones(3),
        terms=terms, degree=2, constant=1.0,
        residual=0.01, constraint_type="equality",
    )
    return DSLType(name=f"State_{env_id}", dimension=3,
                   encoding=encoding, constraints=[constraint],
                   source_env=env_id)


# ── Existing tests ─────────────────────────────────────────────────────────

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


# ── Alpha-equivalence tests ───────────────────────────────────────────────

def test_permuted_sphere_isomorphic():
    """Sphere with permuted indices should be isomorphic to canonical sphere."""
    t1 = _make_sphere_type("ENV_07")
    t2 = _make_sphere_type_permuted("ENV_03")
    assert are_types_isomorphic(t1, t2)


def test_canonicalize_terms_identity():
    """Identity permutation leaves terms unchanged (after sorting)."""
    terms = [(0, 0), (1, 1), (2, 2)]
    perm = {0: 0, 1: 1, 2: 2}
    result = _canonicalize_terms(terms, perm)
    assert result == [(0, 0), (1, 1), (2, 2)]


def test_canonicalize_terms_swap():
    """Swapping indices 0↔2 should permute terms accordingly."""
    terms = [(0, 0), (1, 1), (2, 2)]
    perm = {0: 2, 1: 1, 2: 0}
    result = _canonicalize_terms(terms, perm)
    # (0,0)->perm->(2,2), (1,1)->(1,1), (2,2)->(0,0)
    # After sorting: [(0,0), (1,1), (2,2)]
    assert result == [(0, 0), (1, 1), (2, 2)]


def test_canonicalize_terms_cross_terms():
    """Cross terms (0, 1) should be permuted correctly."""
    terms = [(0, 1)]
    perm = {0: 2, 1: 0}
    result = _canonicalize_terms(terms, perm)
    # (0,1) -> perm -> (2, 0) -> sorted -> (0, 2)
    assert result == [(0, 2)]


def test_alpha_match_swapped_variables():
    """Two constraints identical except variable naming should alpha-match."""
    # Constraint A: z_0² + z_1² = 1 (using terms)
    ca = [Constraint(
        coefficients=np.ones(2),
        terms=[(0, 0), (1, 1)], degree=2, constant=1.0,
        residual=0.01, constraint_type="equality",
    )]
    # Constraint B: z_1² + z_0² = 1 (indices swapped)
    cb = [Constraint(
        coefficients=np.ones(2),
        terms=[(1, 1), (0, 0)], degree=2, constant=1.0,
        residual=0.01, constraint_type="equality",
    )]
    assert _try_alpha_match(ca, cb, dim=2, coeff_tol=0.05)


def test_alpha_no_match_different_constants():
    """Same structure but different constant → not isomorphic."""
    ca = [Constraint(
        coefficients=np.ones(2),
        terms=[(0, 0), (1, 1)], degree=2, constant=1.0,
        residual=0.01, constraint_type="equality",
    )]
    cb = [Constraint(
        coefficients=np.ones(2),
        terms=[(0, 0), (1, 1)], degree=2, constant=2.0,
        residual=0.01, constraint_type="equality",
    )]
    assert not _try_alpha_match(ca, cb, dim=2, coeff_tol=0.05)


def test_unconstrained_same_dim_isomorphic():
    """Two unconstrained types with same dimension are trivially isomorphic."""
    t1 = DSLType(name="A", dimension=3, encoding={}, constraints=[], source_env="E1")
    t2 = DSLType(name="B", dimension=3, encoding={}, constraints=[], source_env="E2")
    assert are_types_isomorphic(t1, t2)


# ── Merge procedure tests ─────────────────────────────────────────────────

def test_unify_picks_simplest_representative():
    """Merge should pick the type with lowest MDL cost as representative."""
    # t1 has a complex encoding, t2 has a simple one
    t1 = DSLType(
        name="State_ENV_07", dimension=3,
        encoding={
            0: BinOp(Op.ADD, Var("a"), Const(1.0)),
            1: BinOp(Op.MUL, Var("b"), Const(2.0)),
            2: Var("c"),
        },
        constraints=[Constraint(np.ones(3), [(0,0),(1,1),(2,2)], 2, 1.0, 0.01, "equality")],
        source_env="ENV_07",
    )
    t2 = DSLType(
        name="State_ENV_03", dimension=3,
        encoding={0: Var("x"), 1: Var("y"), 2: Var("z")},
        constraints=[Constraint(np.ones(3), [(0,0),(1,1),(2,2)], 2, 1.0, 0.01, "equality")],
        source_env="ENV_03",
    )
    result = unify_types([t1, t2])
    assert len(result.unified_types) == 1
    u = result.unified_types[0]
    # t2 has lower MDL cost (simpler encoding)
    assert u["representative_mdl"] <= t1.mdl_cost()


def test_unify_compression_savings():
    """Compression savings should be (n_merged - 1) * type_mdl_cost."""
    t1 = _make_sphere_type("ENV_07")
    t2 = _make_sphere_type("ENV_03")
    t3 = _make_sphere_type("ENV_04")
    result = unify_types([t1, t2, t3])
    assert len(result.unified_types) == 1
    u = result.unified_types[0]
    assert u["n_merged"] == 3
    assert u["compression_savings"] == 2 * u["representative_mdl"]
    assert u["source_envs"] == ["ENV_07", "ENV_03", "ENV_04"]


def test_unify_canonical_constraint_string():
    """Merged type should have canonicalized constraint strings."""
    t1 = _make_sphere_type("ENV_07")
    t2 = _make_sphere_type("ENV_03")
    result = unify_types([t1, t2])
    u = result.unified_types[0]
    # Constraints should be human-readable strings
    assert len(u["constraints"]) >= 1
    c_str = u["constraints"][0]
    assert "z_0" in c_str or "z_" in c_str
    assert "1.0000" in c_str


def test_unify_permuted_types_merged():
    """Types with permuted variable indices should be merged."""
    t1 = _make_sphere_type("ENV_07")
    t2 = _make_sphere_type_permuted("ENV_03")
    result = unify_types([t1, t2])
    assert len(result.unified_types) == 1
    assert result.unified_types[0]["n_merged"] == 2
