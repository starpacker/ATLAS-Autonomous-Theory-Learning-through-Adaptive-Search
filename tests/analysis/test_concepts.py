"""Tests for concept extraction."""
from atlas.analysis.concepts import extract_concepts, Concept
from atlas.dsl.expr import Const, Var, BinOp, UnaryOp
from atlas.dsl.operators import Op


def test_no_concepts_from_single_formula():
    formulas = [BinOp(Op.ADD, Var("x_0"), Const(1.0))]
    concepts = extract_concepts(formulas, min_occurrences=2)
    assert len(concepts) == 0


def test_find_repeated_subexpression():
    cos2 = BinOp(Op.MUL, UnaryOp(Op.COS, Var("x_0")), UnaryOp(Op.COS, Var("x_0")))
    f1 = BinOp(Op.MUL, Const(2.0), cos2)
    f2 = BinOp(Op.ADD, cos2, Const(1.0))
    concepts = extract_concepts([f1, f2], min_occurrences=2)
    assert len(concepts) >= 1


def test_concept_has_savings():
    cos_x = UnaryOp(Op.COS, Var("x_0"))
    cos2 = BinOp(Op.MUL, cos_x, cos_x)
    f1 = BinOp(Op.MUL, Const(2.0), cos2)
    f2 = BinOp(Op.ADD, cos2, Const(1.0))
    f3 = BinOp(Op.SUB, cos2, Var("x_1"))
    concepts = extract_concepts([f1, f2, f3], min_occurrences=2)
    for c in concepts:
        assert c.savings > 0


def test_concept_structure():
    cos_x = UnaryOp(Op.COS, Var("x_0"))
    f1 = BinOp(Op.MUL, Const(2.0), cos_x)
    f2 = BinOp(Op.ADD, cos_x, Const(1.0))
    f3 = BinOp(Op.SUB, cos_x, Var("x_1"))
    concepts = extract_concepts([f1, f2, f3], min_occurrences=2)
    assert len(concepts) >= 1
    c = concepts[0]
    assert isinstance(c, Concept)
    assert c.expr is not None
    assert c.count >= 2
    assert isinstance(c.name, str)


def test_trivial_subexpressions_filtered():
    f1 = BinOp(Op.ADD, Var("x_0"), Const(1.0))
    f2 = BinOp(Op.MUL, Var("x_0"), Const(2.0))
    concepts = extract_concepts([f1, f2], min_occurrences=2, min_size=2)
    assert len(concepts) == 0
