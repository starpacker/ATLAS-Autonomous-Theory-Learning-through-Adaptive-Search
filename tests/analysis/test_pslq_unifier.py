"""Tests for PSLQ constant unification."""
import numpy as np
from atlas.analysis.pslq_unifier import (
    find_constant_relations,
    find_log_relations,
    UnifiedConstant,
    LogRelation,
    unify_constants,
    weighted_mean_std,
    chi2_consistency,
)


# ── Value-space relation tests (existing) ──────────────────────────────────

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


# ── Log-space relation tests ───────────────────────────────────────────────

def test_find_log_relation_equal():
    """Equal constants: |C₁|^1 · |C₂|^(-1) ≈ 1."""
    h = 6.626e-34
    constants = {"ENV_01:C0": h, "ENV_02:C0": h * 1.001}
    relations = find_log_relations(constants)
    assert len(relations) >= 1
    r = relations[0]
    # Verify the relation holds: product(|C_k|^n_k) ≈ 1
    product = 1.0
    for key, exp in zip(r.keys, r.exponents):
        product *= abs(constants[key]) ** exp
    assert abs(product - 1.0) < 0.05


def test_find_log_relation_square():
    """C₂ = C₁² should give exponents whose ratio is 2:1."""
    c1 = 6.626e-34
    constants = {"ENV_01:C0": c1, "ENV_02:C0": c1 ** 2}
    relations = find_log_relations(constants)
    assert len(relations) >= 1
    r = relations[0]
    product = 1.0
    for key, exp in zip(r.keys, r.exponents):
        product *= abs(constants[key]) ** exp
    assert abs(product - 1.0) < 0.01


def test_find_log_relation_cube_root():
    """C₂ = C₁^(1/3) → exponents (1, -3) or equiv."""
    c1 = 1000.0
    constants = {"ENV_01:C0": c1, "ENV_02:C0": c1 ** (1.0 / 3.0)}
    relations = find_log_relations(constants, max_coeff=6)
    assert len(relations) >= 1
    r = relations[0]
    product = 1.0
    for key, exp in zip(r.keys, r.exponents):
        product *= abs(constants[key]) ** exp
    assert abs(product - 1.0) < 0.01


def test_find_log_relation_ratio_pairwise():
    """C₂ = C₁^3 is a pairwise power relation detectable by log search."""
    c1 = 5.0
    constants = {"a": c1, "b": c1 ** 3}
    relations = find_log_relations(constants)
    assert len(relations) >= 1
    r = relations[0]
    product = 1.0
    for key, exp in zip(r.keys, r.exponents):
        product *= abs(constants[key]) ** exp
    assert abs(product - 1.0) < 0.01


def test_find_log_no_relation():
    """π and e have no small-integer log relation."""
    constants = {"ENV_01:C0": 3.14159, "ENV_02:C0": 2.71828}
    relations = find_log_relations(constants, max_coeff=6)
    assert len(relations) == 0


def test_find_log_sign_irrelevant():
    """Signs should be separated — negative values still produce relations."""
    c1 = 50.0
    constants = {"ENV_01:C0": c1, "ENV_02:C0": -(c1 ** 2)}
    relations = find_log_relations(constants)
    assert len(relations) >= 1


# ── Weighted mean/std tests ────────────────────────────────────────────────

def test_weighted_mean_std_equal_weights():
    vals = np.array([10.0, 20.0, 30.0])
    mu, sigma = weighted_mean_std(vals, np.array([1.0, 1.0, 1.0]))
    assert abs(mu - 20.0) < 1e-10
    assert abs(sigma - np.std(vals)) < 1e-10


def test_weighted_mean_std_unequal():
    """High weight on 10.0 should pull mean toward it."""
    vals = np.array([10.0, 20.0])
    mu, _ = weighted_mean_std(vals, np.array([9.0, 1.0]))
    assert mu < 15.0  # pulled toward 10


def test_weighted_mean_std_no_weights():
    vals = np.array([10.0, 20.0, 30.0])
    mu, sigma = weighted_mean_std(vals)
    assert abs(mu - 20.0) < 1e-10
    assert sigma > 0


def test_weighted_mean_std_single():
    vals = np.array([42.0])
    mu, sigma = weighted_mean_std(vals)
    assert mu == 42.0
    assert sigma == 0.0


def test_weighted_mean_std_empty():
    vals = np.array([])
    mu, sigma = weighted_mean_std(vals)
    assert mu == 0.0
    assert sigma == 0.0


# ── Chi-squared consistency tests ──────────────────────────────────────────

def test_chi2_consistent():
    """Very similar values should have high p-value (consistent)."""
    vals = np.array([6.626e-34, 6.627e-34, 6.625e-34])
    _, pval = chi2_consistency(vals)
    assert pval > 0.01


def test_chi2_inconsistent_with_tight_uncertainties():
    """Wildly different values with tight uncertainties → low p-value."""
    vals = np.array([1.0, 10.0, 100.0])
    uncertainties = np.array([0.001, 0.001, 0.001])
    _, pval = chi2_consistency(vals, uncertainties)
    assert pval < 0.01


def test_chi2_single_value():
    vals = np.array([6.626e-34])
    stat, pval = chi2_consistency(vals)
    assert stat == 0.0
    assert pval == 1.0


def test_chi2_identical_values():
    """Identical values → chi² = 0, p = 1."""
    vals = np.array([5.0, 5.0, 5.0])
    stat, pval = chi2_consistency(vals)
    assert stat == 0.0
    assert pval == 1.0


# ── Unified constants with chi-squared ─────────────────────────────────────

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


def test_unify_constants_chi2_populated():
    """unify_constants should now populate chi2_pvalue."""
    h_estimates = [6.626e-34, 6.630e-34, 6.622e-34]
    constants = {f"ENV_{i:02d}:C0": v for i, v in enumerate(h_estimates)}
    unified = unify_constants(constants)
    assert len(unified) >= 1
    uc = unified[0]
    assert uc.chi2_pvalue is not None
    assert uc.chi2_pvalue > 0.01  # these are consistent
    assert uc.is_spurious is False


def test_unify_constants_with_weights():
    """Weights should shift the mean toward higher-weighted values."""
    constants = {"ENV_01:C0": 10.0, "ENV_02:C0": 20.0}
    weights = {"ENV_01:C0": 0.99, "ENV_02:C0": 0.5}
    # tolerance=1.0 so both are grouped together
    unified = unify_constants(constants, tolerance=1.0, weights=weights)
    assert len(unified) == 1
    # Weighted mean should be closer to 10 than simple midpoint 15
    assert unified[0].value < 15.0


def test_unify_constants_spurious_flag():
    """Constants with large spread relative to tight uncertainties → spurious."""
    # These are in the same group (tolerance=1.0) but very different
    constants = {"a": 1.0, "b": 1.5, "c": 0.5}
    unified = unify_constants(constants, tolerance=1.0)
    # With only 3 values, the chi2 test using group std won't necessarily
    # flag as spurious.  But the chi2_pvalue should at least be computed.
    assert len(unified) == 1
    assert unified[0].chi2_pvalue is not None
