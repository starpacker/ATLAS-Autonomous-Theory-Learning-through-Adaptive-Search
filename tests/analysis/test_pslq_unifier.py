"""Tests for PSLQ constant unification."""
import numpy as np
from atlas.analysis.pslq_unifier import (
    find_constant_relations, UnifiedConstant, unify_constants,
)


def test_find_relation_between_multiples():
    constants = {"ENV_01:C0": 6.626, "ENV_02:C0": 3.313}
    relations = find_constant_relations(constants)
    assert len(relations) >= 1


def test_no_relation_for_unrelated():
    constants = {"ENV_01:C0": 3.14159, "ENV_02:C0": 2.71828}
    relations = find_constant_relations(constants, max_coeff=5)
    assert len(relations) == 0


def test_sign_separation():
    constants = {"ENV_01:C0": 6.626, "ENV_02:C0": -6.626}
    relations = find_constant_relations(constants)
    assert len(relations) >= 1


def test_unify_constants_finds_base():
    h = 6.626e-34
    constants = {"ENV_01:C0": h, "ENV_02:C0": h, "ENV_05:C0": h}
    unified = unify_constants(constants)
    assert len(unified) >= 1
    uc = unified[0]
    assert isinstance(uc, UnifiedConstant)
    assert abs(uc.value - h) / h < 0.01
    assert len(uc.appearances) == 3


def test_unify_constants_error_propagation():
    h_estimates = [6.626e-34, 6.630e-34, 6.622e-34]
    constants = {f"ENV_{i:02d}:C0": v for i, v in enumerate(h_estimates)}
    unified = unify_constants(constants)
    if unified:
        uc = unified[0]
        assert uc.uncertainty > 0
