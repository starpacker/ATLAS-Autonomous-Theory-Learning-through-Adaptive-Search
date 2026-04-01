"""Tests for Theory output structure."""
from atlas.unifier.theory import Theory, LawTemplate, CompressionLayer


def test_empty_theory():
    t = Theory()
    assert t.compression_ratio() == 1.0
    assert len(t.law_templates) == 0


def test_add_law_template():
    t = Theory()
    law = LawTemplate(
        template_id="LAW-1",
        template_str="UC_0 * x_0 - x_1",
        shared_constants=["UC_0"],
        applies_to=["ENV_01", "ENV_05"],
        compression_savings=156.0,
    )
    t.add_law_template(law)
    assert len(t.law_templates) == 1


def test_compression_chain():
    t = Theory()
    t.add_compression_layer(CompressionLayer(
        level=0, total_mdl=1247.0, label="independent formulas", delta=0.0))
    t.add_compression_layer(CompressionLayer(
        level=1, total_mdl=891.0, label="constant unification", delta=-356.0))
    t.add_compression_layer(CompressionLayer(
        level=2, total_mdl=724.0, label="template extraction", delta=-167.0))
    assert t.compression_ratio() == 1247.0 / 724.0
    assert len(t.compression_chain) == 3


def test_theory_to_dict():
    t = Theory()
    t.add_compression_layer(CompressionLayer(
        level=0, total_mdl=100.0, label="base", delta=0.0))
    d = t.to_dict()
    assert "law_templates" in d
    assert "shared_constants" in d
    assert "shared_types" in d
    assert "compression_chain" in d
    assert "compression_ratio" in d


def test_theory_add_shared_constant():
    t = Theory()
    t.add_shared_constant(
        symbol="UC_0", value=6.626e-34, uncertainty=0.003e-34,
        appearances=["ENV_01:C0", "ENV_02:C0", "ENV_05:C0"],
        chi2_consistency=0.87,
    )
    assert len(t.shared_constants) == 1
    assert t.shared_constants[0]["symbol"] == "UC_0"
