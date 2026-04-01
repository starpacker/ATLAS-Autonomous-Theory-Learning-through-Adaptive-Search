"""Tests for template extraction via anti-unification."""
from atlas.unifier.template_extractor import (
    anti_unify, extract_templates, TemplateResult,
)
from atlas.dsl.expr import Const, Var, BinOp, UnaryOp
from atlas.dsl.operators import Op


def test_anti_unify_identical():
    """Two identical expressions should yield themselves as template."""
    e = BinOp(Op.MUL, Var("x_0"), Const(2.0))
    result = anti_unify(e, e)
    assert result.template == e
    assert len(result.holes) == 0


def test_anti_unify_different_constants():
    """Same structure, different constants -> template with hole."""
    e1 = BinOp(Op.MUL, Var("x_0"), Const(2.0))
    e2 = BinOp(Op.MUL, Var("x_0"), Const(3.0))
    result = anti_unify(e1, e2)
    # Template should be x_0 * HOLE_0
    assert isinstance(result.template, BinOp)
    assert len(result.holes) == 1


def test_anti_unify_different_structure():
    """Completely different structures -> single hole (too general)."""
    e1 = BinOp(Op.ADD, Var("x_0"), Const(1.0))
    e2 = UnaryOp(Op.SIN, Var("x_0"))
    result = anti_unify(e1, e2)
    assert isinstance(result.template, Var)  # just a hole variable
    assert len(result.holes) == 1


def test_anti_unify_nested():
    """cos(x_0 * C1) vs cos(x_0 * C2) -> cos(x_0 * HOLE)."""
    e1 = UnaryOp(Op.COS, BinOp(Op.MUL, Var("x_0"), Const(3.14)))
    e2 = UnaryOp(Op.COS, BinOp(Op.MUL, Var("x_0"), Const(6.28)))
    result = anti_unify(e1, e2)
    assert isinstance(result.template, UnaryOp)
    assert result.template.op == Op.COS


def test_extract_templates_finds_shared():
    """Formulas with shared structure should yield templates."""
    cos_inner1 = BinOp(Op.MUL, Var("x_0"), Const(3.14))
    cos_inner2 = BinOp(Op.MUL, Var("x_0"), Const(6.28))
    f1 = BinOp(Op.MUL, Const(2.0), UnaryOp(Op.COS, cos_inner1))
    f2 = BinOp(Op.MUL, Const(5.0), UnaryOp(Op.COS, cos_inner2))

    formulas = {"ENV_01": f1, "ENV_02": f2}
    templates = extract_templates(formulas, min_savings=1)
    assert len(templates) >= 1


def test_extract_templates_mdl_filter():
    """Templates that don't save MDL should be filtered out."""
    f1 = Var("x_0")
    f2 = Var("x_1")
    formulas = {"ENV_01": f1, "ENV_02": f2}
    templates = extract_templates(formulas, min_savings=5)
    assert len(templates) == 0  # too simple to benefit from templating
