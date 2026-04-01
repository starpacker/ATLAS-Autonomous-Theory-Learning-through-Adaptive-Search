"""RGDE Step 4c: Discover algebraic constraints on bottleneck vectors."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations_with_replacement
import numpy as np

@dataclass
class Constraint:
    coefficients: np.ndarray
    terms: list[tuple[int, ...]]
    degree: int
    constant: float
    residual: float
    constraint_type: str


def find_constraints(Z: np.ndarray, max_degree: int = 2,
                     max_residual: float = 0.05, min_samples: int = 50) -> list[Constraint]:
    N, K = Z.shape
    if N < min_samples:
        return []

    terms, features = _polynomial_features(Z, K, max_degree)
    constraints = []

    # Method 1: Check individual polynomial terms for near-constancy (equality)
    for i, (term, feat) in enumerate(zip(terms, features.T)):
        if len(term) < 2:
            continue
        mean_val = np.mean(feat)
        if abs(mean_val) < 1e-10:
            continue
        rel_std = np.std(feat) / abs(mean_val)
        if rel_std < max_residual:
            constraints.append(Constraint(
                coefficients=np.array([1.0]), terms=[term], degree=len(term),
                constant=float(mean_val), residual=float(rel_std),
                constraint_type="equality"))

    # Method 2: SVD to find near-constant linear combinations of polynomial terms
    if features.shape[1] >= 2:
        centered = features - np.mean(features, axis=0, keepdims=True)
        try:
            U, s, Vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return constraints

        if len(s) > 0 and s[0] > 1e-10:
            for idx in range(len(s)):
                rel_sv = s[idx] / s[0]
                if rel_sv < max_residual:
                    coeffs = Vt[idx]
                    combo = features @ coeffs
                    mean_val = np.mean(combo)
                    if abs(mean_val) < 1e-10:
                        mean_val = 1.0
                    rel_std = np.std(combo) / abs(mean_val)
                    sig_mask = np.abs(coeffs) > 0.01 * np.max(np.abs(coeffs))
                    sig_terms = [terms[j] for j in range(len(terms)) if sig_mask[j]]
                    sig_coeffs = coeffs[sig_mask]
                    if len(sig_terms) >= 1:
                        constraints.append(Constraint(
                            coefficients=sig_coeffs, terms=sig_terms,
                            degree=max(len(t) for t in sig_terms),
                            constant=float(np.mean(features @ coeffs)),
                            residual=float(rel_std), constraint_type="equality"))

    # Method 3: Detect range/inequality constraints via low coefficient of variation
    # of polynomial features — catches cases where a polynomial is bounded (e.g. disk)
    for i, (term, feat) in enumerate(zip(terms, features.T)):
        if len(term) < 2:
            continue
        mean_val = np.mean(feat)
        if abs(mean_val) < 1e-10:
            continue
        # Coefficient of variation: how spread relative to mean
        rel_std = np.std(feat) / abs(mean_val)
        # Range relative to mean: catches bounded but not constant features
        feat_range = (np.max(feat) - np.min(feat)) / abs(mean_val)
        # Detect when all values have the same sign (bounded away from zero)
        all_positive = np.all(feat > 0)
        all_negative = np.all(feat < 0)
        if (all_positive or all_negative) and feat_range < 2.0 and rel_std < 0.5:
            # Check not already found as equality
            already_found = any(
                c.terms == [term] and c.constraint_type == "equality"
                for c in constraints
            )
            if not already_found:
                constraints.append(Constraint(
                    coefficients=np.array([1.0]), terms=[term], degree=len(term),
                    constant=float(mean_val), residual=float(rel_std),
                    constraint_type="inequality"))

    # Method 4: SVD-based range constraint — linear combos with low variation
    if features.shape[1] >= 2 and not any(c.constraint_type == "inequality" for c in constraints):
        for idx in range(min(features.shape[1], 10)):
            if len(s) > 0:
                rel_sv = s[idx] / s[0] if s[0] > 1e-10 else 1.0
                if rel_sv < 0.4:  # relatively small singular value
                    coeffs = Vt[idx]
                    combo = features @ coeffs
                    mean_val = np.mean(combo)
                    if abs(mean_val) < 1e-10:
                        continue
                    rel_std = np.std(combo) / abs(mean_val)
                    all_same_sign = np.all(combo > 0) or np.all(combo < 0)
                    if all_same_sign and rel_std < 0.5:
                        sig_mask = np.abs(coeffs) > 0.01 * np.max(np.abs(coeffs))
                        sig_terms = [terms[j] for j in range(len(terms)) if sig_mask[j]]
                        sig_coeffs = coeffs[sig_mask]
                        if len(sig_terms) >= 1:
                            constraints.append(Constraint(
                                coefficients=sig_coeffs, terms=sig_terms,
                                degree=max(len(t) for t in sig_terms),
                                constant=float(mean_val), residual=float(rel_std),
                                constraint_type="inequality"))

    constraints.sort(key=lambda c: c.residual)
    # Deduplicate
    filtered = []
    seen: list[set] = []
    for c in constraints:
        ts = frozenset(c.terms)
        if not any(ts == s for s in seen):
            filtered.append(c)
            seen.append(ts)
    return filtered


def _polynomial_features(Z, K, max_degree):
    terms = []
    columns = []
    for degree in range(1, max_degree + 1):
        for combo in combinations_with_replacement(range(K), degree):
            term = tuple(combo)
            col = np.ones(Z.shape[0])
            for idx in term:
                col = col * Z[:, idx]
            terms.append(term)
            columns.append(col)
    features = np.column_stack(columns) if columns else np.empty((Z.shape[0], 0))
    return terms, features
