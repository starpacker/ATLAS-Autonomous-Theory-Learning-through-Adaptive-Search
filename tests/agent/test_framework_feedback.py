"""Tests for the framework discovery feedback loop.

Verifies the critical changes that close the loop between RGDE type
discovery and SR search:
  1. Multi-variable concepts are properly augmented into SR features
  2. RGDE encoder formulas are promoted to DSL concepts
  3. The combined mechanism enables cross-epoch and cross-experiment
     type propagation
"""
import numpy as np
import pytest

from atlas.agent.atlas_agent import _augment_with_concepts, ATLASAgent, AgentConfig
from atlas.agent.dsl_state import DSLState
from atlas.dsl.expr import Var, BinOp, UnaryOp, Const
from atlas.dsl.operators import Op
from atlas.dsl.serialize import from_str, to_str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_encoder_formula_2var():
    """sin(knob_0) * cos(knob_1) — a typical 2-variable encoder formula."""
    return BinOp(
        Op.MUL,
        UnaryOp(Op.SIN, Var("knob_0")),
        UnaryOp(Op.COS, Var("knob_1")),
    )


def _make_single_var_concept():
    """cos(v)**2 — a typical single-variable concept."""
    return BinOp(
        Op.MUL,
        UnaryOp(Op.COS, Var("v")),
        UnaryOp(Op.COS, Var("v")),
    )


# ---------------------------------------------------------------------------
# Tests: _augment_with_concepts — multi-variable support
# ---------------------------------------------------------------------------

class TestAugmentMultiVariable:
    """Verify that _augment_with_concepts handles multi-variable concepts."""

    def test_single_var_still_works(self):
        """Single-variable concepts should still expand per-variable."""
        X = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
        var_names = ["knob_0", "knob_1"]
        concepts = {"cos2": _make_single_var_concept()}

        X_aug, names_aug = _augment_with_concepts(X, var_names, concepts)

        # Should add cos2__knob_0 and cos2__knob_1
        assert len(names_aug) > len(var_names)
        assert any("cos2__knob_0" in n for n in names_aug)
        assert any("cos2__knob_1" in n for n in names_aug)

    def test_multi_var_adds_single_column(self):
        """Multi-variable concept should produce one column, not one per var."""
        X = np.array([[0.5, 1.0], [1.0, 0.5], [1.5, 2.0]])
        var_names = ["knob_0", "knob_1"]
        encoder = _make_encoder_formula_2var()  # sin(knob_0) * cos(knob_1)
        concepts = {"State_ENV07_z0": encoder}

        X_aug, names_aug = _augment_with_concepts(X, var_names, concepts)

        assert "State_ENV07_z0" in names_aug
        # Original 2 columns + 1 new column
        assert X_aug.shape[1] == 3
        # Verify the values are correct
        expected = np.sin(X[:, 0]) * np.cos(X[:, 1])
        np.testing.assert_allclose(X_aug[:, 2], expected, atol=1e-10)

    def test_multi_var_skipped_if_vars_missing(self):
        """Multi-variable concept should be skipped if var names don't match."""
        X = np.array([[0.5, 1.0], [1.0, 0.5]])
        var_names = ["knob_0", "knob_2"]  # knob_1 missing!
        encoder = _make_encoder_formula_2var()  # needs knob_0 and knob_1
        concepts = {"State_ENV07_z0": encoder}

        X_aug, names_aug = _augment_with_concepts(X, var_names, concepts)

        # Should return original — multi-var concept can't be evaluated
        assert X_aug.shape == X.shape
        assert names_aug == var_names

    def test_mixed_single_and_multi(self):
        """Both single-var and multi-var concepts work together."""
        X = np.array([[0.5, 1.0], [1.0, 0.5], [1.5, 2.0]])
        var_names = ["knob_0", "knob_1"]
        concepts = {
            "cos2": _make_single_var_concept(),
            "State_ENV07_z0": _make_encoder_formula_2var(),
        }

        X_aug, names_aug = _augment_with_concepts(X, var_names, concepts)

        # cos2 expands to 2 columns (one per knob), encoder adds 1
        assert X_aug.shape[1] >= 5  # 2 original + 2 cos2 + 1 encoder
        assert "State_ENV07_z0" in names_aug

    def test_empty_concepts(self):
        """Empty concepts dict returns original data unchanged."""
        X = np.array([[1.0, 2.0]])
        var_names = ["knob_0", "knob_1"]

        X_aug, names_aug = _augment_with_concepts(X, var_names, {})

        assert X_aug is X
        assert names_aug is var_names

    def test_multi_var_constant_output_skipped(self):
        """Multi-variable concept with constant output (zero std) is skipped."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        var_names = ["knob_0", "knob_1"]
        # knob_0 * 0 → always 0
        zero_expr = BinOp(Op.MUL, Var("knob_0"), Const(0.0))
        # This is single-var, but let's test a multi-var constant:
        # (knob_0 - knob_0) → always 0, but this is also single-var...
        # Let's just use knob_0 * knob_1 with all-same rows
        X_const = np.ones((5, 2))  # all 1s → product always 1
        multi_const = BinOp(Op.MUL, Var("knob_0"), Var("knob_1"))
        concepts = {"const_multi": multi_const}

        X_aug, names_aug = _augment_with_concepts(X_const, var_names, concepts)

        # std is 0 → column should be skipped
        assert X_aug.shape == X_const.shape


# ---------------------------------------------------------------------------
# Tests: _promote_extensions_to_concepts
# ---------------------------------------------------------------------------

class TestPromoteExtensions:
    """Verify that RGDE encoder formulas are promoted to DSL concepts."""

    def _make_agent_with_extension(self):
        """Create an agent with a mock RGDE extension in dsl_state."""
        agent = ATLASAgent(
            env_ids=["ENV_10"],
            config=AgentConfig(max_epochs=1, n_samples_per_knob=5),
        )
        # Simulate RGDE having discovered a type
        encoder_z0 = BinOp(Op.MUL, UnaryOp(Op.SIN, Var("knob_0")),
                           UnaryOp(Op.COS, Var("knob_1")))
        encoder_z1 = UnaryOp(Op.EXP, Var("knob_0"))

        agent.dsl_state.add_extension(
            name="State_ENV-07",
            ext_type="new_type",
            definition={
                "name": "State_ENV-07",
                "dimension": 2,
                "encoding": {
                    "0": to_str(encoder_z0),
                    "1": to_str(encoder_z1),
                },
                "constraints": [],
                "source_env": "ENV-07",
            },
            trigger="RGDE on ENV-07, K=2",
            source_env="ENV-07",
        )
        return agent

    def test_promotion_adds_concepts(self):
        agent = self._make_agent_with_extension()
        assert len(agent.dsl_state.concepts) == 0

        n = agent._promote_extensions_to_concepts()

        assert n == 2
        assert "State_ENV-07_z0" in agent.dsl_state.concepts
        assert "State_ENV-07_z1" in agent.dsl_state.concepts

    def test_promotion_idempotent(self):
        """Calling promote twice should not duplicate concepts."""
        agent = self._make_agent_with_extension()

        n1 = agent._promote_extensions_to_concepts()
        n2 = agent._promote_extensions_to_concepts()

        assert n1 == 2
        assert n2 == 0  # already exists
        assert len(agent.dsl_state.concepts) == 2

    def test_promoted_concepts_are_valid_exprs(self):
        """Promoted concepts should be parseable Expr objects."""
        agent = self._make_agent_with_extension()
        agent._promote_extensions_to_concepts()

        for name, expr in agent.dsl_state.concepts.items():
            # Should be able to round-trip serialize
            s = to_str(expr)
            recovered = from_str(s)
            assert to_str(recovered) == s

    def test_promoted_concepts_work_in_augmentation(self):
        """Promoted encoder formulas should produce valid augmentation columns."""
        agent = self._make_agent_with_extension()
        agent._promote_extensions_to_concepts()

        X = np.array([[0.5, 1.0], [1.0, 0.5], [1.5, 2.0]])
        var_names = ["knob_0", "knob_1"]

        X_aug, names_aug = _augment_with_concepts(
            X, var_names, agent.dsl_state.concepts)

        # Multi-var concepts should have been evaluated
        assert X_aug.shape[1] > X.shape[1]
        # z0 = sin(knob_0) * cos(knob_1) — should be present
        assert "State_ENV-07_z0" in names_aug
        # z1 = exp(knob_0) — this is single-var, gets expanded per variable
        assert any("State_ENV-07_z1" in n for n in names_aug)

    def test_non_type_extensions_ignored(self):
        """Extensions that aren't 'new_type' should not be promoted."""
        agent = ATLASAgent(
            env_ids=["ENV_10"],
            config=AgentConfig(max_epochs=1),
        )
        agent.dsl_state.add_extension(
            name="prob_mode",
            ext_type="prob_mode",
            definition={"desc": "enable P(y|x)"},
            trigger="D1=stochastic",
        )

        n = agent._promote_extensions_to_concepts()

        assert n == 0
        assert len(agent.dsl_state.concepts) == 0

    def test_malformed_encoding_handled_gracefully(self):
        """Unparseable encoder strings should be skipped, not crash."""
        agent = ATLASAgent(
            env_ids=["ENV_10"],
            config=AgentConfig(max_epochs=1),
        )
        agent.dsl_state.add_extension(
            name="State_BAD",
            ext_type="new_type",
            definition={
                "name": "State_BAD",
                "dimension": 1,
                "encoding": {"0": "this is not valid s-expr"},
                "constraints": [],
                "source_env": "ENV_BAD",
            },
            trigger="test",
        )

        n = agent._promote_extensions_to_concepts()

        assert n == 0  # skipped, no crash


# ---------------------------------------------------------------------------
# Tests: Cross-experiment propagation via concepts
# ---------------------------------------------------------------------------

class TestCrossExperimentPropagation:
    """Verify that encoder concepts from one experiment can augment another."""

    def test_encoder_applies_to_matching_experiment(self):
        """Encoder from ENV-07 (uses knob_0, knob_1) should augment data
        from another experiment that also has knob_0 and knob_1."""
        encoder = _make_encoder_formula_2var()  # sin(knob_0) * cos(knob_1)
        concepts = {"State_ENV07_z0": encoder}

        # Data from a DIFFERENT experiment with matching knobs
        X_other = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        var_names_other = ["knob_0", "knob_1"]

        X_aug, names_aug = _augment_with_concepts(
            X_other, var_names_other, concepts)

        assert "State_ENV07_z0" in names_aug
        expected = np.sin(X_other[:, 0]) * np.cos(X_other[:, 1])
        idx = names_aug.index("State_ENV07_z0")
        np.testing.assert_allclose(X_aug[:, idx], expected, atol=1e-10)

    def test_encoder_not_applied_to_fewer_knobs(self):
        """Encoder needing knob_0 and knob_1 should not apply to
        an experiment with only knob_0."""
        encoder = _make_encoder_formula_2var()
        concepts = {"State_ENV07_z0": encoder}

        X_other = np.array([[0.1], [0.3], [0.5]])
        var_names_other = ["knob_0"]  # missing knob_1

        X_aug, names_aug = _augment_with_concepts(
            X_other, var_names_other, concepts)

        # Should be unchanged
        assert X_aug.shape == X_other.shape

    def test_encoder_applies_to_superset_knobs(self):
        """Encoder needing knob_0 and knob_1 should work on an experiment
        with knob_0, knob_1, knob_2 (extra knobs are fine)."""
        encoder = _make_encoder_formula_2var()
        concepts = {"State_ENV07_z0": encoder}

        X_other = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6],
                            [0.7, 0.8, 0.9]])
        var_names_other = ["knob_0", "knob_1", "knob_2"]

        X_aug, names_aug = _augment_with_concepts(
            X_other, var_names_other, concepts)

        assert "State_ENV07_z0" in names_aug
        # Should use knob_0 and knob_1 columns (indices 0 and 1)
        expected = np.sin(X_other[:, 0]) * np.cos(X_other[:, 1])
        idx = names_aug.index("State_ENV07_z0")
        np.testing.assert_allclose(X_aug[:, idx], expected, atol=1e-10)
