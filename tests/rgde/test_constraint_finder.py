# tests/rgde/test_constraint_finder.py
import numpy as np
from atlas.rgde.constraint_finder import find_constraints, Constraint

def test_find_sphere_constraint():
    rng = np.random.default_rng(42)
    n = 200
    raw = rng.normal(0, 1, (n, 3))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    Z = raw / norms  # all on unit sphere
    constraints = find_constraints(Z, max_degree=2)
    assert len(constraints) >= 1
    c = constraints[0]
    assert isinstance(c, Constraint)
    assert c.residual < 0.05

def test_find_disk_constraint():
    rng = np.random.default_rng(42)
    n = 200
    angles = rng.uniform(0, 2 * np.pi, n)
    radii = rng.uniform(0.5, 1.0, n)
    Z = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    constraints = find_constraints(Z, max_degree=2)
    assert len(constraints) >= 1

def test_no_constraint_for_random():
    rng = np.random.default_rng(42)
    Z = rng.uniform(-1, 1, (200, 3))
    constraints = find_constraints(Z, max_degree=2, max_residual=0.01)
    assert len(constraints) == 0

def test_constraint_structure():
    rng = np.random.default_rng(42)
    raw = rng.normal(0, 1, (200, 2))
    Z = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    constraints = find_constraints(Z, max_degree=2)
    if constraints:
        c = constraints[0]
        assert hasattr(c, "coefficients")
        assert hasattr(c, "degree")
        assert hasattr(c, "constant")
        assert hasattr(c, "residual")
