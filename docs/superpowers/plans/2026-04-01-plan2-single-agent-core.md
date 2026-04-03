# Plan 2: Single-Agent ATLAS Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single ATLAS agent that can collect experiment data, run symbolic regression, extract concepts, diagnose failures, unify constants, and orchestrate the full discovery loop (Steps 1-3, 5 — without RGDE Step 4).

**Architecture:** The agent is a pure algorithmic pipeline (no LLM). It sweeps experiment knobs to collect datasets, feeds them to PySR for symbolic regression, mines frequent subexpressions as concepts (DreamCoder-style), runs 5 statistical diagnostics on failures, and uses PSLQ to find cross-experiment constant relationships. The loop iterates until convergence or budget exhaustion.

**Tech Stack:** Python 3.11+, PySR (Julia backend), NumPy, SciPy, mpmath (PSLQ), scikit-learn (DBSCAN for D2)

**Existing code (Plan 1):**
- `atlas/types.py` — EnvSchema, KnobSpec, FitMetrics, FormulaRecord
- `atlas/dsl/` — Expr AST, Op enum, DSL_0, canonicalize, serialize
- `atlas/environments/` — 12 environments with registry, base, normalizer

---

## File Structure

```
atlas/
  data/
    __init__.py
    collector.py          # Systematic knob sweeping + data collection
    dataset.py            # ExperimentDataset: (knobs, detectors) pairs
  sr/
    __init__.py
    pysr_wrapper.py       # PySR integration: run SR, convert results to Expr
    formula_store.py      # Store/query/compare formulas with fit metrics
  analysis/
    __init__.py
    concepts.py           # Concept extraction: frequent subexpression mining
    diagnostics.py        # D1-D5 diagnostic tests
    pslq_unifier.py       # PSLQ constant unification
  agent/
    __init__.py
    dsl_state.py          # DSL state: current operators + extensions
    atlas_agent.py        # Single-agent main loop orchestrator
tests/
  data/
    __init__.py
    test_collector.py
    test_dataset.py
  sr/
    __init__.py
    test_pysr_wrapper.py
    test_formula_store.py
  analysis/
    __init__.py
    test_concepts.py
    test_diagnostics.py
    test_pslq_unifier.py
  agent/
    __init__.py
    test_dsl_state.py
    test_atlas_agent.py
```

---

## Task 1: ExperimentDataset

**Files:**
- Create: `atlas/data/__init__.py`
- Create: `atlas/data/dataset.py`
- Create: `tests/data/__init__.py`
- Create: `tests/data/test_dataset.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/test_dataset.py
"""Tests for ExperimentDataset."""
import numpy as np
from atlas.data.dataset import ExperimentDataset


def test_create_empty():
    ds = ExperimentDataset(env_id="ENV_01", knob_names=["knob_0", "knob_1"],
                           detector_names=["detector_0"])
    assert ds.env_id == "ENV_01"
    assert len(ds) == 0


def test_add_and_len():
    ds = ExperimentDataset(env_id="ENV_01", knob_names=["knob_0"],
                           detector_names=["detector_0"])
    ds.add(knobs={"knob_0": 0.5}, detectors={"detector_0": 1.0})
    ds.add(knobs={"knob_0": 0.7}, detectors={"detector_0": 2.0})
    assert len(ds) == 2


def test_get_knob_array():
    ds = ExperimentDataset(env_id="ENV_01", knob_names=["knob_0", "knob_1"],
                           detector_names=["detector_0"])
    ds.add(knobs={"knob_0": 0.1, "knob_1": 0.2}, detectors={"detector_0": 1.0})
    ds.add(knobs={"knob_0": 0.3, "knob_1": 0.4}, detectors={"detector_0": 2.0})
    X = ds.knob_array()
    assert X.shape == (2, 2)
    np.testing.assert_array_equal(X[0], [0.1, 0.2])


def test_get_detector_array_scalar():
    ds = ExperimentDataset(env_id="ENV_01", knob_names=["knob_0"],
                           detector_names=["detector_0"])
    ds.add(knobs={"knob_0": 0.1}, detectors={"detector_0": 1.5})
    ds.add(knobs={"knob_0": 0.2}, detectors={"detector_0": 2.5})
    y = ds.detector_array("detector_0")
    np.testing.assert_array_equal(y, [1.5, 2.5])


def test_get_detector_array_vector():
    ds = ExperimentDataset(env_id="ENV_04", knob_names=["knob_0"],
                           detector_names=["detector_0"])
    ds.add(knobs={"knob_0": 0.5}, detectors={"detector_0": np.array([1.0, 2.0, 3.0])})
    y = ds.detector_array("detector_0")
    assert y.shape == (1, 3)


def test_split_train_test():
    ds = ExperimentDataset(env_id="ENV_01", knob_names=["knob_0"],
                           detector_names=["detector_0"])
    for i in range(100):
        ds.add(knobs={"knob_0": i / 100}, detectors={"detector_0": float(i)})
    train, test = ds.split(test_fraction=0.2, seed=42)
    assert len(train) + len(test) == 100
    assert len(test) == 20


def test_from_env():
    """Create dataset by running an environment with a grid of knob values."""
    from atlas.environments.registry import get_environment
    env = get_environment("ENV_10")  # spring, deterministic scalar
    ds = ExperimentDataset.from_env(env, n_samples_per_knob=5, seed=42)
    assert len(ds) > 0
    assert ds.env_id == "ENV_10"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/data/test_dataset.py -v`

- [ ] **Step 3: Implement dataset.py**

```python
# atlas/data/__init__.py
"""Data collection and dataset management."""
```

```python
# atlas/data/dataset.py
"""ExperimentDataset: structured storage for experiment observations."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from atlas.types import KnobType


class ExperimentDataset:
    """Stores (knob_settings, detector_readings) pairs from an experiment."""

    def __init__(self, env_id: str, knob_names: list[str],
                 detector_names: list[str]):
        self.env_id = env_id
        self.knob_names = list(knob_names)
        self.detector_names = list(detector_names)
        self._knobs: list[dict[str, float]] = []
        self._detectors: list[dict[str, float | np.ndarray]] = []

    def __len__(self) -> int:
        return len(self._knobs)

    def add(self, knobs: dict[str, float],
            detectors: dict[str, float | np.ndarray]) -> None:
        self._knobs.append(dict(knobs))
        self._detectors.append(dict(detectors))

    def knob_array(self) -> np.ndarray:
        """Return (N, n_knobs) array of knob values in name order."""
        return np.array([[k[name] for name in self.knob_names]
                         for k in self._knobs])

    def detector_array(self, detector_name: str) -> np.ndarray:
        """Return detector values. Shape: (N,) for scalars, (N, L) for arrays."""
        values = [d[detector_name] for d in self._detectors]
        if isinstance(values[0], np.ndarray):
            return np.stack(values)
        return np.array(values)

    def split(self, test_fraction: float = 0.2,
              seed: int = 42) -> tuple[ExperimentDataset, ExperimentDataset]:
        """Split into train/test sets."""
        rng = np.random.default_rng(seed)
        n = len(self)
        n_test = int(n * test_fraction)
        indices = rng.permutation(n)
        test_idx = set(indices[:n_test].tolist())

        train = ExperimentDataset(self.env_id, self.knob_names, self.detector_names)
        test = ExperimentDataset(self.env_id, self.knob_names, self.detector_names)
        for i in range(n):
            target = test if i in test_idx else train
            target.add(self._knobs[i], self._detectors[i])
        return train, test

    @classmethod
    def from_env(cls, env, n_samples_per_knob: int = 10,
                 seed: int = 42) -> ExperimentDataset:
        """Collect data by sweeping knobs in a grid (or random for high-dim)."""
        schema = env.get_schema()
        knob_names = [k.name for k in schema.knobs]
        detector_names = [d.name for d in schema.detectors]
        ds = cls(schema.env_id, knob_names, detector_names)

        rng = np.random.default_rng(seed)

        # For <= 3 continuous knobs, use grid. Otherwise random.
        continuous_knobs = [k for k in schema.knobs if k.knob_type == KnobType.CONTINUOUS]
        discrete_knobs = [k for k in schema.knobs
                          if k.knob_type in (KnobType.DISCRETE, KnobType.INTEGER)]

        if len(continuous_knobs) <= 3:
            # Grid sweep for continuous knobs
            grids = []
            for k in continuous_knobs:
                grids.append(np.linspace(k.range_min, k.range_max, n_samples_per_knob))
            mesh = np.meshgrid(*grids, indexing='ij')
            flat = [m.ravel() for m in mesh]
            n_grid = len(flat[0]) if flat else 1

            # For each grid point, try each discrete knob combination
            discrete_combos = _discrete_combinations(discrete_knobs)

            for combo in discrete_combos:
                for i in range(n_grid):
                    knobs = {}
                    for j, k in enumerate(continuous_knobs):
                        knobs[k.name] = float(flat[j][i])
                    knobs.update(combo)
                    try:
                        result = env.run(knobs)
                        ds.add(knobs, result)
                    except (ValueError, RuntimeError):
                        pass
        else:
            # Random sampling for high-dimensional spaces
            discrete_combos = _discrete_combinations(discrete_knobs)
            n_total = n_samples_per_knob ** 3  # cap at cubic
            for _ in range(n_total):
                knobs = {}
                for k in continuous_knobs:
                    knobs[k.name] = float(rng.uniform(k.range_min, k.range_max))
                combo = discrete_combos[rng.integers(len(discrete_combos))]
                knobs.update(combo)
                try:
                    result = env.run(knobs)
                    ds.add(knobs, result)
                except (ValueError, RuntimeError):
                    pass

        return ds


def _discrete_combinations(knobs) -> list[dict[str, float]]:
    """Generate all combinations of discrete/integer knob values."""
    if not knobs:
        return [{}]
    combos = [{}]
    for k in knobs:
        if k.knob_type == KnobType.DISCRETE and k.options:
            new_combos = []
            for combo in combos:
                for opt in k.options:
                    new_combos.append({**combo, k.name: opt})
            combos = new_combos
        elif k.knob_type == KnobType.INTEGER:
            # Sample a few integer values
            vals = [int(k.range_min), int((k.range_min + k.range_max) / 2), int(k.range_max)]
            new_combos = []
            for combo in combos:
                for v in vals:
                    new_combos.append({**combo, k.name: v})
            combos = new_combos
    return combos
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/data/test_dataset.py -v`

- [ ] **Step 5: Commit**

```bash
git add atlas/data/ tests/data/
git commit -m "feat: ExperimentDataset with grid/random collection and train/test split"
```

---

## Task 2: PySR Wrapper

**Files:**
- Create: `atlas/sr/__init__.py`
- Create: `atlas/sr/pysr_wrapper.py`
- Create: `tests/sr/__init__.py`
- Create: `tests/sr/test_pysr_wrapper.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/sr/test_pysr_wrapper.py
"""Tests for PySR wrapper.

NOTE: These tests require PySR and Julia to be installed.
Tests are marked with @pytest.mark.slow for CI skip.
For fast unit testing, we test the result conversion logic
with mock PySR outputs.
"""
import pytest
import numpy as np
from atlas.sr.pysr_wrapper import (
    SRConfig, run_sr, pysr_expr_to_atlas, SRResult,
)
from atlas.dsl.expr import Const, Var, BinOp, UnaryOp, Expr
from atlas.dsl.operators import Op, DSL_0


def test_sr_config_defaults():
    cfg = SRConfig()
    assert cfg.niterations == 40
    assert cfg.populations == 15
    assert cfg.maxsize == 25
    assert cfg.parsimony == 0.0032
    assert set(cfg.binary_operators) == {"+", "-", "*", "/", "^"}
    assert set(cfg.unary_operators) == {"sin", "cos", "exp", "log", "neg"}


def test_sr_config_from_dsl():
    cfg = SRConfig.from_dsl(DSL_0)
    assert "+" in cfg.binary_operators
    assert "sin" in cfg.unary_operators


def test_sr_result_structure():
    result = SRResult(
        formulas=[],
        best_formula=None,
        best_r_squared=-1.0,
        best_mdl=float("inf"),
        converged=False,
    )
    assert not result.converged


# --- Mock-based tests for expression conversion ---

def test_convert_simple_addition():
    """Test converting a sympy-like string to ATLAS Expr."""
    # PySR returns string expressions; we parse them
    expr = pysr_expr_to_atlas("x0 + 1.5", var_names=["knob_0"])
    assert isinstance(expr, Expr)
    result = expr.evaluate({"knob_0": 2.0})
    assert abs(result - 3.5) < 1e-10


def test_convert_nested():
    """sin(x0 * 3.14)"""
    expr = pysr_expr_to_atlas("sin(x0 * 3.14159)", var_names=["knob_0"])
    result = expr.evaluate({"knob_0": 0.5})
    assert abs(result - 1.0) < 1e-4


def test_convert_with_constants():
    """x0 * 2.5 + x1"""
    expr = pysr_expr_to_atlas("x0 * 2.5 + x1", var_names=["knob_0", "knob_1"])
    result = expr.evaluate({"knob_0": 1.0, "knob_1": 3.0})
    assert abs(result - 5.5) < 1e-10


def test_convert_power():
    """x0 ^ 2.0"""
    expr = pysr_expr_to_atlas("x0 ^ 2.0", var_names=["knob_0"])
    result = expr.evaluate({"knob_0": 3.0})
    assert abs(result - 9.0) < 1e-10


@pytest.mark.slow
def test_run_sr_on_simple_data():
    """Integration test: PySR should discover y = 2*x + 1."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (100, 1))
    y = 2 * X[:, 0] + 1 + rng.normal(0, 0.01, 100)

    result = run_sr(X, y, var_names=["knob_0"],
                    config=SRConfig(niterations=20, populations=10, maxsize=10))
    assert result.best_r_squared > 0.99
    assert result.best_formula is not None
```

- [ ] **Step 2: Run tests (non-slow) to verify they fail**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/sr/test_pysr_wrapper.py -v -m "not slow"`

- [ ] **Step 3: Implement PySR wrapper**

```python
# atlas/sr/__init__.py
"""Symbolic regression integration."""
```

```python
# atlas/sr/pysr_wrapper.py
"""PySR wrapper: run symbolic regression and convert results to ATLAS Expr."""
from __future__ import annotations

import re
import math
from dataclasses import dataclass, field

import numpy as np

from atlas.dsl.expr import Expr, Const, Var, BinOp, UnaryOp
from atlas.dsl.operators import Op, DSL_0
from atlas.types import FitMetrics


# Mapping from Op to PySR operator strings
_OP_TO_PYSR_BINARY = {
    Op.ADD: "+", Op.SUB: "-", Op.MUL: "*", Op.DIV: "/", Op.POW: "^",
}
_OP_TO_PYSR_UNARY = {
    Op.SIN: "sin", Op.COS: "cos", Op.EXP: "exp", Op.LOG: "log", Op.NEG: "neg",
}
_PYSR_TO_OP_BINARY = {v: k for k, v in _OP_TO_PYSR_BINARY.items()}
_PYSR_TO_OP_UNARY = {v: k for k, v in _OP_TO_PYSR_UNARY.items()}


@dataclass
class SRConfig:
    """Configuration for a PySR run."""
    niterations: int = 40
    populations: int = 15
    maxsize: int = 25
    parsimony: float = 0.0032
    binary_operators: list[str] = field(default_factory=lambda: ["+", "-", "*", "/", "^"])
    unary_operators: list[str] = field(default_factory=lambda: ["sin", "cos", "exp", "log", "neg"])
    timeout_seconds: int = 300
    deterministic: bool = False
    random_state: int | None = None

    @classmethod
    def from_dsl(cls, dsl: frozenset[Op], **kwargs) -> SRConfig:
        binary = [_OP_TO_PYSR_BINARY[op] for op in dsl if op in _OP_TO_PYSR_BINARY]
        unary = [_OP_TO_PYSR_UNARY[op] for op in dsl if op in _OP_TO_PYSR_UNARY]
        return cls(binary_operators=binary, unary_operators=unary, **kwargs)


@dataclass
class SRResult:
    """Result of a symbolic regression run."""
    formulas: list[tuple[Expr, FitMetrics]]  # Pareto front: (expr, metrics)
    best_formula: Expr | None
    best_r_squared: float
    best_mdl: float
    converged: bool


def run_sr(X: np.ndarray, y: np.ndarray, var_names: list[str],
           config: SRConfig | None = None) -> SRResult:
    """Run PySR symbolic regression.

    Args:
        X: (N, n_features) input data
        y: (N,) target data
        var_names: names for each input column (e.g., ["knob_0", "knob_1"])
        config: SR configuration

    Returns:
        SRResult with Pareto front of formulas
    """
    try:
        from pysr import PySRRegressor
    except ImportError:
        raise ImportError(
            "PySR is required. Install with: pip install pysr && python -c 'import pysr; pysr.install()'"
        )

    if config is None:
        config = SRConfig()

    model = PySRRegressor(
        niterations=config.niterations,
        populations=config.populations,
        maxsize=config.maxsize,
        parsimony=config.parsimony,
        binary_operators=config.binary_operators,
        unary_operators=config.unary_operators,
        timeout_in_seconds=config.timeout_seconds,
        deterministic=config.deterministic,
        random_state=config.random_state,
        temp_equation_file=True,
        verbosity=0,
    )

    model.fit(X, y)

    # Extract Pareto front
    pareto = []
    equations = model.equations_
    if equations is not None and len(equations) > 0:
        for _, row in equations.iterrows():
            try:
                expr = pysr_expr_to_atlas(str(row["equation"]), var_names)
                r2 = float(row.get("score", 0.0))
                complexity = int(row.get("complexity", 1))
                loss = float(row.get("loss", 1.0))
                metrics = FitMetrics(
                    r_squared=1.0 - loss / max(np.var(y), 1e-30),
                    residual_var=loss,
                    mdl=float(complexity),
                )
                pareto.append((expr, metrics))
            except (ValueError, KeyError):
                continue

    # Find best by R²
    if pareto:
        best_idx = max(range(len(pareto)), key=lambda i: pareto[i][1].r_squared)
        best_expr, best_metrics = pareto[best_idx]
        return SRResult(
            formulas=pareto,
            best_formula=best_expr,
            best_r_squared=best_metrics.r_squared,
            best_mdl=best_metrics.mdl,
            converged=best_metrics.r_squared > 0.95,
        )

    return SRResult(formulas=[], best_formula=None, best_r_squared=-1.0,
                    best_mdl=float("inf"), converged=False)


def pysr_expr_to_atlas(expr_str: str, var_names: list[str]) -> Expr:
    """Convert a PySR equation string to an ATLAS Expr.

    PySR outputs expressions like: "sin(x0 * 3.14) + x1"
    Variable names are x0, x1, ... which we map to var_names.
    """
    var_map = {f"x{i}": name for i, name in enumerate(var_names)}
    tokens = _tokenize_expr(expr_str)
    expr, pos = _parse_expr(tokens, 0, var_map)
    return expr


def _tokenize_expr(s: str) -> list[str]:
    """Tokenize a PySR expression string."""
    tokens: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c in " \t\n":
            i += 1
        elif c in "(),":
            tokens.append(c)
            i += 1
        elif c in "+-*/^":
            tokens.append(c)
            i += 1
        elif c.isdigit() or (c == '.' and i + 1 < len(s) and s[i + 1].isdigit()):
            j = i
            while j < len(s) and (s[j].isdigit() or s[j] in '.eE+-'):
                if s[j] in 'eE' and j + 1 < len(s) and s[j + 1] in '+-':
                    j += 2
                elif s[j] in '+-' and j > i and s[j - 1] not in 'eE':
                    break
                else:
                    j += 1
            tokens.append(s[i:j])
            i = j
        elif c.isalpha() or c == '_':
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == '_'):
                j += 1
            tokens.append(s[i:j])
            i = j
        else:
            i += 1
    return tokens


def _parse_expr(tokens: list[str], pos: int,
                var_map: dict[str, str]) -> tuple[Expr, int]:
    """Recursive descent parser for PySR expressions."""
    return _parse_additive(tokens, pos, var_map)


def _parse_additive(tokens: list[str], pos: int,
                    var_map: dict[str, str]) -> tuple[Expr, int]:
    left, pos = _parse_multiplicative(tokens, pos, var_map)
    while pos < len(tokens) and tokens[pos] in ("+", "-"):
        op_str = tokens[pos]
        pos += 1
        right, pos = _parse_multiplicative(tokens, pos, var_map)
        op = Op.ADD if op_str == "+" else Op.SUB
        left = BinOp(op, left, right)
    return left, pos


def _parse_multiplicative(tokens: list[str], pos: int,
                          var_map: dict[str, str]) -> tuple[Expr, int]:
    left, pos = _parse_power(tokens, pos, var_map)
    while pos < len(tokens) and tokens[pos] in ("*", "/"):
        op_str = tokens[pos]
        pos += 1
        right, pos = _parse_power(tokens, pos, var_map)
        op = Op.MUL if op_str == "*" else Op.DIV
        left = BinOp(op, left, right)
    return left, pos


def _parse_power(tokens: list[str], pos: int,
                 var_map: dict[str, str]) -> tuple[Expr, int]:
    left, pos = _parse_unary(tokens, pos, var_map)
    if pos < len(tokens) and tokens[pos] == "^":
        pos += 1
        right, pos = _parse_unary(tokens, pos, var_map)
        left = BinOp(Op.POW, left, right)
    return left, pos


def _parse_unary(tokens: list[str], pos: int,
                 var_map: dict[str, str]) -> tuple[Expr, int]:
    if pos < len(tokens) and tokens[pos] == "-":
        pos += 1
        operand, pos = _parse_primary(tokens, pos, var_map)
        return UnaryOp(Op.NEG, operand), pos
    return _parse_primary(tokens, pos, var_map)


def _parse_primary(tokens: list[str], pos: int,
                   var_map: dict[str, str]) -> tuple[Expr, int]:
    if pos >= len(tokens):
        raise ValueError("Unexpected end of expression")

    token = tokens[pos]

    # Function call: sin(expr), cos(expr), etc.
    if token in _PYSR_TO_OP_UNARY and pos + 1 < len(tokens) and tokens[pos + 1] == "(":
        op = _PYSR_TO_OP_UNARY[token]
        pos += 2  # skip func name and '('
        arg, pos = _parse_expr(tokens, pos, var_map)
        if pos < len(tokens) and tokens[pos] == ")":
            pos += 1
        return UnaryOp(op, arg), pos

    # Parenthesized expression
    if token == "(":
        pos += 1
        expr, pos = _parse_expr(tokens, pos, var_map)
        if pos < len(tokens) and tokens[pos] == ")":
            pos += 1
        return expr, pos

    # Variable
    if token in var_map:
        return Var(var_map[token]), pos + 1

    # Number
    try:
        return Const(float(token)), pos + 1
    except ValueError:
        pass

    raise ValueError(f"Unexpected token: '{token}' at position {pos}")
```

- [ ] **Step 4: Run non-slow tests**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/sr/test_pysr_wrapper.py -v -m "not slow"`

- [ ] **Step 5: Commit**

```bash
git add atlas/sr/ tests/sr/
git commit -m "feat: PySR wrapper with expression parser and SR config"
```

---

## Task 3: Formula Store

**Files:**
- Create: `atlas/sr/formula_store.py`
- Create: `tests/sr/test_formula_store.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/sr/test_formula_store.py
"""Tests for formula store."""
from atlas.sr.formula_store import FormulaStore
from atlas.dsl.expr import Const, Var, BinOp, UnaryOp
from atlas.dsl.operators import Op
from atlas.types import FitMetrics


def test_add_and_get():
    store = FormulaStore()
    expr = BinOp(Op.MUL, Var("knob_0"), Const(2.0))
    metrics = FitMetrics(r_squared=0.99, residual_var=0.01, mdl=5.0)
    store.add("ENV_01", expr, metrics)
    results = store.get("ENV_01")
    assert len(results) == 1
    assert results[0].expr == expr
    assert results[0].fit.r_squared == 0.99


def test_get_best():
    store = FormulaStore()
    e1 = BinOp(Op.ADD, Var("knob_0"), Const(1.0))
    e2 = BinOp(Op.MUL, Var("knob_0"), Const(2.0))
    store.add("ENV_01", e1, FitMetrics(r_squared=0.8, residual_var=0.2, mdl=4.0))
    store.add("ENV_01", e2, FitMetrics(r_squared=0.95, residual_var=0.05, mdl=5.0))
    best = store.get_best("ENV_01")
    assert best is not None
    assert best.expr == e2


def test_get_best_empty():
    store = FormulaStore()
    assert store.get_best("ENV_99") is None


def test_all_constants():
    store = FormulaStore()
    e1 = BinOp(Op.MUL, Var("knob_0"), Const(6.626e-34))
    e2 = BinOp(Op.ADD, Var("knob_0"), Const(2.998e8))
    store.add("ENV_01", e1, FitMetrics(r_squared=0.99, residual_var=0.01, mdl=5.0))
    store.add("ENV_02", e2, FitMetrics(r_squared=0.99, residual_var=0.01, mdl=4.0))
    constants = store.all_constants()
    assert 6.626e-34 in constants
    assert 2.998e8 in constants


def test_all_env_ids():
    store = FormulaStore()
    e = Const(1.0)
    store.add("ENV_01", e, FitMetrics(r_squared=0.5, residual_var=0.5, mdl=1.0))
    store.add("ENV_02", e, FitMetrics(r_squared=0.5, residual_var=0.5, mdl=1.0))
    assert store.all_env_ids() == {"ENV_01", "ENV_02"}


def test_pareto_front():
    store = FormulaStore()
    # Simple formula, low accuracy
    e1 = Var("knob_0")
    store.add("ENV_01", e1, FitMetrics(r_squared=0.5, residual_var=0.5, mdl=1.0))
    # Complex formula, high accuracy
    e2 = BinOp(Op.MUL, UnaryOp(Op.SIN, Var("knob_0")), Const(3.14))
    store.add("ENV_01", e2, FitMetrics(r_squared=0.99, residual_var=0.01, mdl=6.0))
    # Dominated formula (worse accuracy AND more complex than e2)
    e3 = BinOp(Op.ADD, BinOp(Op.MUL, Var("knob_0"), Const(1.0)), Const(0.5))
    store.add("ENV_01", e3, FitMetrics(r_squared=0.7, residual_var=0.3, mdl=7.0))

    pareto = store.pareto_front("ENV_01")
    # e3 is dominated by e2 (worse R² AND higher MDL), so should be filtered
    assert len(pareto) == 2
    pareto_exprs = {f.expr for f in pareto}
    assert e1 in pareto_exprs  # simple + low acc = non-dominated
    assert e2 in pareto_exprs  # complex + high acc = non-dominated
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement formula_store.py**

```python
# atlas/sr/formula_store.py
"""FormulaStore: store, query, and compare discovered formulas."""
from __future__ import annotations

from dataclasses import dataclass
from atlas.dsl.expr import Expr, Const, Var, BinOp, UnaryOp, NAryOp
from atlas.types import FitMetrics


@dataclass
class StoredFormula:
    """A formula stored with its environment and fit metrics."""
    env_id: str
    expr: Expr
    fit: FitMetrics


class FormulaStore:
    """Stores all discovered formulas, indexed by environment."""

    def __init__(self):
        self._formulas: dict[str, list[StoredFormula]] = {}

    def add(self, env_id: str, expr: Expr, fit: FitMetrics) -> None:
        if env_id not in self._formulas:
            self._formulas[env_id] = []
        self._formulas[env_id].append(StoredFormula(env_id, expr, fit))

    def get(self, env_id: str) -> list[StoredFormula]:
        return list(self._formulas.get(env_id, []))

    def get_best(self, env_id: str) -> StoredFormula | None:
        formulas = self.get(env_id)
        if not formulas:
            return None
        return max(formulas, key=lambda f: f.fit.r_squared)

    def all_env_ids(self) -> set[str]:
        return set(self._formulas.keys())

    def all_constants(self) -> list[float]:
        """Extract all numerical constants from all stored formulas."""
        constants: list[float] = []
        for formulas in self._formulas.values():
            for sf in formulas:
                constants.extend(_extract_constants(sf.expr))
        return constants

    def pareto_front(self, env_id: str) -> list[StoredFormula]:
        """Return non-dominated formulas (R² vs MDL tradeoff)."""
        formulas = self.get(env_id)
        if not formulas:
            return []
        # A formula is dominated if another has both higher R² AND lower MDL
        pareto = []
        for f in formulas:
            dominated = False
            for other in formulas:
                if (other.fit.r_squared > f.fit.r_squared and
                        other.fit.mdl < f.fit.mdl):
                    dominated = True
                    break
            if not dominated:
                pareto.append(f)
        return pareto


def _extract_constants(expr: Expr) -> list[float]:
    """Recursively extract all Const values from an expression."""
    if isinstance(expr, Const):
        return [expr.value]
    if isinstance(expr, Var):
        return []
    if isinstance(expr, UnaryOp):
        return _extract_constants(expr.operand)
    if isinstance(expr, BinOp):
        return _extract_constants(expr.left) + _extract_constants(expr.right)
    if isinstance(expr, NAryOp):
        result: list[float] = []
        for c in expr.children:
            result.extend(_extract_constants(c))
        return result
    return []
```

- [ ] **Step 4: Run tests**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/sr/test_formula_store.py -v`

- [ ] **Step 5: Commit**

```bash
git add atlas/sr/formula_store.py tests/sr/test_formula_store.py
git commit -m "feat: formula store with Pareto front and constant extraction"
```

---

## Task 4: Concept Extraction

**Files:**
- Create: `atlas/analysis/__init__.py`
- Create: `atlas/analysis/concepts.py`
- Create: `tests/analysis/__init__.py`
- Create: `tests/analysis/test_concepts.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/analysis/test_concepts.py
"""Tests for concept extraction (frequent subexpression mining)."""
from atlas.analysis.concepts import extract_concepts, Concept
from atlas.dsl.expr import Const, Var, BinOp, UnaryOp
from atlas.dsl.operators import Op


def test_no_concepts_from_single_formula():
    formulas = [BinOp(Op.ADD, Var("x_0"), Const(1.0))]
    concepts = extract_concepts(formulas, min_occurrences=2)
    assert len(concepts) == 0


def test_find_repeated_subexpression():
    # cos^2(x) appears in two formulas
    cos2 = BinOp(Op.MUL, UnaryOp(Op.COS, Var("x_0")), UnaryOp(Op.COS, Var("x_0")))
    f1 = BinOp(Op.MUL, Const(2.0), cos2)
    f2 = BinOp(Op.ADD, cos2, Const(1.0))
    concepts = extract_concepts([f1, f2], min_occurrences=2)
    # cos(x_0) appears in both formulas (inside cos^2)
    assert len(concepts) >= 1


def test_concept_has_savings():
    cos_x = UnaryOp(Op.COS, Var("x_0"))
    cos2 = BinOp(Op.MUL, cos_x, cos_x)
    f1 = BinOp(Op.MUL, Const(2.0), cos2)
    f2 = BinOp(Op.ADD, cos2, Const(1.0))
    f3 = BinOp(Op.SUB, cos2, Var("x_1"))
    concepts = extract_concepts([f1, f2, f3], min_occurrences=2)
    # cos^2(x_0) appears 3 times, size=5 (mul, cos, x_0, cos, x_0)
    # savings = 3 * 5 = 15, cost = 5 (definition)
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
    """Single variables and small constants should not be concepts."""
    f1 = BinOp(Op.ADD, Var("x_0"), Const(1.0))
    f2 = BinOp(Op.MUL, Var("x_0"), Const(2.0))
    concepts = extract_concepts([f1, f2], min_occurrences=2, min_size=2)
    # x_0 appears twice but size=1, should be filtered
    assert len(concepts) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement concepts.py**

```python
# atlas/analysis/__init__.py
"""Analysis modules: concept extraction, diagnostics, constant unification."""
```

```python
# atlas/analysis/concepts.py
"""Concept extraction: mine frequent subexpressions from formulas (DreamCoder-style).

A concept is a subexpression that appears frequently across formulas and whose
reuse saves more description length than the cost of defining it.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter

from atlas.dsl.expr import Expr, Const, Var, BinOp, UnaryOp, NAryOp
from atlas.dsl.canonicalize import canonicalize
from atlas.dsl.serialize import to_str


@dataclass
class Concept:
    """A discovered reusable subexpression."""
    name: str
    expr: Expr           # the canonical subexpression
    count: int           # how many formulas it appears in
    savings: int         # MDL savings = count * size - definition_cost


def extract_concepts(formulas: list[Expr], min_occurrences: int = 2,
                     min_size: int = 2) -> list[Concept]:
    """Extract frequent subexpressions that provide MDL savings.

    Args:
        formulas: list of expressions to analyze
        min_occurrences: minimum number of formulas a subexpression must appear in
        min_size: minimum AST size for a subexpression to be considered

    Returns:
        List of Concept objects, sorted by savings (descending)
    """
    # Count subexpressions across formulas (count each subexpr once per formula)
    subexpr_counts: Counter[str] = Counter()
    subexpr_map: dict[str, Expr] = {}

    for formula in formulas:
        # Get all unique subexpressions in this formula
        seen_in_formula: set[str] = set()
        for sub in _all_subexprs(formula):
            if sub.size() < min_size:
                continue
            # Skip trivial subexprs (just a variable or small constant)
            if isinstance(sub, (Var, Const)):
                continue
            canon = canonicalize(sub)
            key = to_str(canon)
            if key not in seen_in_formula:
                seen_in_formula.add(key)
                subexpr_counts[key] += 1
                subexpr_map[key] = canon

    # Filter by min_occurrences and compute savings
    concepts = []
    concept_id = 0
    for key, count in subexpr_counts.items():
        if count < min_occurrences:
            continue
        expr = subexpr_map[key]
        size = expr.size()
        savings = count * size - size  # savings = (count-1) * size
        if savings > 0:
            concepts.append(Concept(
                name=f"concept_{concept_id}",
                expr=expr,
                count=count,
                savings=savings,
            ))
            concept_id += 1

    concepts.sort(key=lambda c: c.savings, reverse=True)
    return concepts


def _all_subexprs(expr: Expr) -> list[Expr]:
    """Return all subexpressions of an expression (including itself)."""
    result = [expr]
    if isinstance(expr, UnaryOp):
        result.extend(_all_subexprs(expr.operand))
    elif isinstance(expr, BinOp):
        result.extend(_all_subexprs(expr.left))
        result.extend(_all_subexprs(expr.right))
    elif isinstance(expr, NAryOp):
        for c in expr.children:
            result.extend(_all_subexprs(c))
    return result
```

- [ ] **Step 4: Run tests**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/analysis/test_concepts.py -v`

- [ ] **Step 5: Commit**

```bash
git add atlas/analysis/ tests/analysis/
git commit -m "feat: concept extraction with MDL-based savings"
```

---

## Task 5: Diagnostics (D1-D5)

**Files:**
- Create: `atlas/analysis/diagnostics.py`
- Create: `tests/analysis/test_diagnostics.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/analysis/test_diagnostics.py
"""Tests for D1-D5 diagnostic tests."""
import numpy as np
from atlas.analysis.diagnostics import (
    DiagnosticResult, diagnose_stochasticity, diagnose_discreteness,
    diagnose_residual_structure, run_all_diagnostics,
)
from atlas.data.dataset import ExperimentDataset


def test_d1_stochastic_positive():
    """Repeated measurements with high variance -> stochastic."""
    result = diagnose_stochasticity(
        repeated_outputs=[
            np.array([1.0, 0.0, 0.0, 1.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
        ]
    )
    assert result.triggered
    assert result.diagnostic_id == "D1"


def test_d1_stochastic_negative():
    """Repeated measurements with zero variance -> deterministic."""
    same = np.array([1.0, 2.0, 3.0])
    result = diagnose_stochasticity(
        repeated_outputs=[same, same, same]
    )
    assert not result.triggered


def test_d2_discrete_positive():
    """Outputs clustered into few values -> discrete."""
    outputs = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    result = diagnose_discreteness(outputs, max_clusters=10)
    assert result.triggered
    assert result.diagnostic_id == "D2"
    assert result.details["n_clusters"] == 2


def test_d2_discrete_negative():
    """Outputs spanning a continuous range -> not discrete."""
    rng = np.random.default_rng(42)
    outputs = rng.uniform(0, 1, 1000)
    result = diagnose_discreteness(outputs, max_clusters=10)
    assert not result.triggered


def test_d4_residual_structure_positive():
    """Residuals with periodic structure -> structured."""
    x = np.linspace(0, 10, 200)
    residuals = np.sin(5 * x)  # periodic residuals
    result = diagnose_residual_structure(residuals)
    assert result.triggered
    assert result.diagnostic_id == "D4"


def test_d4_residual_structure_negative():
    """White noise residuals -> no structure."""
    rng = np.random.default_rng(42)
    residuals = rng.normal(0, 1, 200)
    result = diagnose_residual_structure(residuals)
    assert not result.triggered


def test_run_all_returns_list():
    ds = ExperimentDataset("ENV_TEST", ["knob_0"], ["detector_0"])
    for i in range(50):
        ds.add({"knob_0": i / 50}, {"detector_0": float(i)})
    results = run_all_diagnostics(ds, best_r_squared=0.99, residuals=np.zeros(50))
    assert isinstance(results, list)
    for r in results:
        assert isinstance(r, DiagnosticResult)
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement diagnostics.py**

```python
# atlas/analysis/diagnostics.py
"""D1-D5 diagnostic tests for detecting DSL insufficiency.

All diagnostics are pure statistical tests — no physics prior knowledge.

D1: Stochasticity — repeated identical experiments have high variance
D2: Discreteness — outputs cluster into few distinct values
D3: Dimension insufficiency — (deferred to Plan 3, requires SciNet)
D4: Residual structure — best-fit residuals have non-white-noise patterns
D5: Cross-experiment inconsistency — (implemented in Unifier, Plan 4)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from atlas.data.dataset import ExperimentDataset


@dataclass
class DiagnosticResult:
    """Result of a single diagnostic test."""
    diagnostic_id: str       # "D1", "D2", "D3", "D4", "D5"
    triggered: bool          # True if diagnostic detects an issue
    confidence: float        # 0.0 to 1.0
    details: dict = field(default_factory=dict)


def diagnose_stochasticity(repeated_outputs: list[np.ndarray],
                           threshold: float = 0.05) -> DiagnosticResult:
    """D1: Check if repeated measurements under identical conditions have high variance.

    Args:
        repeated_outputs: list of output arrays from repeated runs (same knob settings)
        threshold: coefficient of variation threshold

    Returns:
        DiagnosticResult with D1 findings
    """
    if len(repeated_outputs) < 2:
        return DiagnosticResult("D1", False, 0.0, {"reason": "need >= 2 repeats"})

    stacked = np.array(repeated_outputs, dtype=float)
    # Compute variance across repeats for each output element
    var_across_repeats = np.var(stacked, axis=0)
    mean_across_repeats = np.mean(np.abs(stacked), axis=0)

    # Coefficient of variation (avoid division by zero)
    safe_mean = np.where(mean_across_repeats > 1e-10, mean_across_repeats, 1.0)
    cv = np.mean(var_across_repeats / (safe_mean ** 2))

    triggered = cv > threshold
    return DiagnosticResult(
        "D1", triggered, min(cv / threshold, 1.0),
        {"coefficient_of_variation": float(cv), "threshold": threshold}
    )


def diagnose_discreteness(outputs: np.ndarray,
                          max_clusters: int = 10) -> DiagnosticResult:
    """D2: Check if outputs cluster into a small number of distinct values.

    Uses simple histogram-based gap detection instead of DBSCAN for robustness.

    Args:
        outputs: 1D array of scalar outputs
        max_clusters: maximum number of clusters to consider "discrete"
    """
    if len(outputs) < 10:
        return DiagnosticResult("D2", False, 0.0, {"reason": "need >= 10 samples"})

    outputs = outputs.ravel()
    unique_vals = np.unique(np.round(outputs, decimals=6))

    if len(unique_vals) <= max_clusters:
        return DiagnosticResult(
            "D2", True, 1.0,
            {"n_clusters": len(unique_vals), "values": unique_vals.tolist()}
        )

    # Also check for gaps: sort and look for large jumps
    sorted_vals = np.sort(outputs)
    diffs = np.diff(sorted_vals)
    if len(diffs) == 0:
        return DiagnosticResult("D2", True, 1.0, {"n_clusters": 1})

    median_diff = np.median(diffs[diffs > 0]) if np.any(diffs > 0) else 1.0
    large_gaps = np.sum(diffs > 10 * median_diff)
    n_clusters_est = large_gaps + 1

    triggered = n_clusters_est <= max_clusters and n_clusters_est < len(outputs) / 5
    return DiagnosticResult(
        "D2", triggered, min(float(max_clusters / max(n_clusters_est, 1)), 1.0),
        {"n_clusters": int(n_clusters_est)}
    )


def diagnose_residual_structure(residuals: np.ndarray,
                                significance: float = 0.05) -> DiagnosticResult:
    """D4: Check if residuals have non-white-noise structure (periodicity, trends).

    Uses FFT to detect dominant frequencies and autocorrelation for trends.
    """
    residuals = residuals.ravel()
    n = len(residuals)
    if n < 20:
        return DiagnosticResult("D4", False, 0.0, {"reason": "need >= 20 residuals"})

    # FFT analysis: check if any frequency has disproportionate power
    fft = np.abs(np.fft.rfft(residuals - np.mean(residuals)))
    fft[0] = 0  # remove DC component
    if np.sum(fft ** 2) < 1e-20:
        return DiagnosticResult("D4", False, 0.0, {"reason": "residuals near zero"})

    total_power = np.sum(fft ** 2)
    max_power = np.max(fft ** 2)
    concentration = max_power / total_power

    # Autocorrelation at lag 1
    if np.std(residuals) > 1e-10:
        autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    else:
        autocorr = 0.0

    # Trigger if strong spectral concentration OR high autocorrelation
    triggered = concentration > 0.2 or abs(autocorr) > 0.5

    return DiagnosticResult(
        "D4", triggered,
        max(concentration, abs(autocorr)),
        {
            "spectral_concentration": float(concentration),
            "autocorrelation_lag1": float(autocorr),
            "dominant_freq_idx": int(np.argmax(fft)),
        }
    )


def run_all_diagnostics(dataset: ExperimentDataset,
                        best_r_squared: float,
                        residuals: np.ndarray | None = None,
                        repeated_outputs: list[np.ndarray] | None = None,
                        ) -> list[DiagnosticResult]:
    """Run all applicable diagnostics on a dataset.

    Args:
        dataset: the experiment dataset
        best_r_squared: R² of the best formula found for this experiment
        residuals: residuals from best fit (if available)
        repeated_outputs: outputs from repeated runs at same knob settings (if available)
    """
    results = []

    # D1: Stochasticity
    if repeated_outputs is not None and len(repeated_outputs) >= 2:
        results.append(diagnose_stochasticity(repeated_outputs))
    else:
        results.append(DiagnosticResult("D1", False, 0.0, {"reason": "no repeated data"}))

    # D2: Discreteness
    try:
        y = dataset.detector_array(dataset.detector_names[0])
        if y.ndim == 1:
            results.append(diagnose_discreteness(y))
        else:
            results.append(DiagnosticResult("D2", False, 0.0, {"reason": "array output"}))
    except (IndexError, KeyError):
        results.append(DiagnosticResult("D2", False, 0.0, {"reason": "no detector data"}))

    # D3: Dimension insufficiency (requires SciNet — deferred to Plan 3)
    results.append(DiagnosticResult("D3", False, 0.0, {"reason": "requires SciNet (Plan 3)"}))

    # D4: Residual structure
    if residuals is not None and len(residuals) >= 20:
        results.append(diagnose_residual_structure(residuals))
    else:
        results.append(DiagnosticResult("D4", False, 0.0, {"reason": "no residuals"}))

    # D5: Cross-experiment (handled by Unifier — deferred to Plan 4)
    results.append(DiagnosticResult("D5", False, 0.0, {"reason": "requires Unifier (Plan 4)"}))

    return results
```

- [ ] **Step 4: Run tests**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/analysis/test_diagnostics.py -v`

- [ ] **Step 5: Commit**

```bash
git add atlas/analysis/diagnostics.py tests/analysis/test_diagnostics.py
git commit -m "feat: D1/D2/D4 diagnostics (stochasticity, discreteness, residual structure)"
```

---

## Task 6: PSLQ Constant Unification

**Files:**
- Create: `atlas/analysis/pslq_unifier.py`
- Create: `tests/analysis/test_pslq_unifier.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/analysis/test_pslq_unifier.py
"""Tests for PSLQ constant unification."""
import numpy as np
from atlas.analysis.pslq_unifier import (
    find_constant_relations, UnifiedConstant, unify_constants,
)


def test_find_relation_between_multiples():
    """C1 = 2*C2 should be found."""
    constants = {"ENV_01:C0": 6.626, "ENV_02:C0": 3.313}
    relations = find_constant_relations(constants)
    # Should find that C1 ≈ 2 * C2
    assert len(relations) >= 1


def test_no_relation_for_unrelated():
    """Random constants should have no simple relations."""
    constants = {"ENV_01:C0": 3.14159, "ENV_02:C0": 2.71828}
    relations = find_constant_relations(constants, max_coeff=5)
    # pi and e have no simple integer relation
    assert len(relations) == 0


def test_sign_separation():
    """PSLQ should handle negative constants correctly."""
    constants = {"ENV_01:C0": 6.626, "ENV_02:C0": -6.626}
    relations = find_constant_relations(constants)
    # Should find |C1| = |C2| with opposite signs
    assert len(relations) >= 1


def test_unify_constants_finds_base():
    """Multiple constants that are powers of a base should be unified."""
    # C1 = h, C2 = h^2, C3 = 2*h
    h = 6.626e-34
    constants = {
        "ENV_01:C0": h,
        "ENV_02:C0": h,
        "ENV_05:C0": h,
    }
    unified = unify_constants(constants)
    assert len(unified) >= 1
    # All three should map to the same universal constant
    uc = unified[0]
    assert isinstance(uc, UnifiedConstant)
    assert abs(uc.value - h) / h < 0.01
    assert len(uc.appearances) == 3


def test_unify_constants_error_propagation():
    """Unified constant should have uncertainty from spread of estimates."""
    h_estimates = [6.626e-34, 6.630e-34, 6.622e-34]
    constants = {f"ENV_{i:02d}:C0": v for i, v in enumerate(h_estimates)}
    unified = unify_constants(constants)
    if unified:
        uc = unified[0]
        assert uc.uncertainty > 0
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement pslq_unifier.py**

```python
# atlas/analysis/pslq_unifier.py
"""PSLQ-based constant unification.

Discovers integer relations between constants found in different experiments.
Uses log-space PSLQ with proper sign separation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ConstantRelation:
    """An integer relation between constants: prod(|C_i|^n_i) ≈ 1."""
    constants: dict[str, float]    # name -> value
    exponents: dict[str, int]      # name -> integer exponent
    residual: float                # how close the relation is to exact


@dataclass
class UnifiedConstant:
    """A universal constant extracted from multiple experiments."""
    symbol: str                    # e.g., "UC_0"
    value: float                   # best estimate
    uncertainty: float             # standard deviation across estimates
    appearances: list[str]         # list of "ENV_XX:CN" where it appears
    signs: dict[str, int]          # sign of each appearance (+1 or -1)


def find_constant_relations(constants: dict[str, float],
                            max_coeff: int = 10,
                            tolerance: float = 1e-4) -> list[ConstantRelation]:
    """Find integer relations between constants using PSLQ in log-space.

    Sign separation: signs are recorded separately; PSLQ operates on |C_i|.
    """
    if len(constants) < 2:
        return []

    names = list(constants.keys())
    values = [constants[n] for n in names]

    # Separate signs
    signs = [1 if v >= 0 else -1 for v in values]
    abs_values = [abs(v) for v in values]

    # Filter out zeros
    valid = [(n, av, s) for n, av, s in zip(names, abs_values, signs) if av > 1e-100]
    if len(valid) < 2:
        return []

    valid_names, valid_abs, valid_signs = zip(*valid)
    log_values = [math.log(v) for v in valid_abs]

    relations = []

    # Try all pairs for simple integer relations
    for i in range(len(valid_names)):
        for j in range(i + 1, len(valid_names)):
            ratio = log_values[i] / log_values[j] if abs(log_values[j]) > 1e-30 else float('inf')
            # Check if ratio is close to a simple fraction p/q
            for p in range(-max_coeff, max_coeff + 1):
                for q in range(1, max_coeff + 1):
                    if p == 0 and q == 0:
                        continue
                    target = p / q
                    if abs(ratio - target) < tolerance:
                        # Found: log(|Ci|) ≈ (p/q) * log(|Cj|)
                        # i.e., |Ci|^q ≈ |Cj|^p
                        residual = abs(valid_abs[i] ** q - valid_abs[j] ** p)
                        norm = max(valid_abs[i] ** q, valid_abs[j] ** p, 1e-100)
                        rel_residual = residual / norm
                        if rel_residual < tolerance:
                            relations.append(ConstantRelation(
                                constants={valid_names[i]: values[i],
                                           valid_names[j]: values[j]},
                                exponents={valid_names[i]: q, valid_names[j]: -p},
                                residual=rel_residual,
                            ))

    # Deduplicate: keep only the simplest relation for each pair
    seen_pairs: set[tuple[str, str]] = set()
    unique: list[ConstantRelation] = []
    for r in sorted(relations, key=lambda r: sum(abs(v) for v in r.exponents.values())):
        pair = tuple(sorted(r.constants.keys()))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            unique.append(r)

    return unique


def unify_constants(constants: dict[str, float],
                    tolerance: float = 0.01) -> list[UnifiedConstant]:
    """Group constants that are approximately equal and compute unified values.

    Uses relative tolerance for comparison. Sign-aware.

    Args:
        constants: mapping from "ENV_XX:CN" to constant value
        tolerance: relative tolerance for grouping

    Returns:
        List of UnifiedConstant objects
    """
    if not constants:
        return []

    names = list(constants.keys())
    values = [constants[n] for n in names]
    abs_values = [abs(v) for v in values]
    signs_list = [1 if v >= 0 else -1 for v in values]

    # Group by approximate absolute value
    groups: list[list[int]] = []
    assigned = [False] * len(names)

    for i in range(len(names)):
        if assigned[i]:
            continue
        group = [i]
        assigned[i] = True
        for j in range(i + 1, len(names)):
            if assigned[j]:
                continue
            if abs_values[i] > 1e-100 and abs_values[j] > 1e-100:
                rel_diff = abs(abs_values[i] - abs_values[j]) / max(abs_values[i], abs_values[j])
                if rel_diff < tolerance:
                    group.append(j)
                    assigned[j] = True
        if len(group) >= 2:  # only unify if appears in multiple experiments
            groups.append(group)

    # Build UnifiedConstants
    unified = []
    for idx, group in enumerate(groups):
        group_abs = [abs_values[i] for i in group]
        group_names = [names[i] for i in group]
        group_signs = {names[i]: signs_list[i] for i in group}

        value = float(np.mean(group_abs))
        uncertainty = float(np.std(group_abs)) if len(group_abs) > 1 else 0.0

        unified.append(UnifiedConstant(
            symbol=f"UC_{idx}",
            value=value,
            uncertainty=uncertainty,
            appearances=group_names,
            signs=group_signs,
        ))

    return unified
```

- [ ] **Step 4: Run tests**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/analysis/test_pslq_unifier.py -v`

- [ ] **Step 5: Commit**

```bash
git add atlas/analysis/pslq_unifier.py tests/analysis/test_pslq_unifier.py
git commit -m "feat: PSLQ constant unification with sign separation"
```

---

## Task 7: DSL State

**Files:**
- Create: `atlas/agent/__init__.py`
- Create: `atlas/agent/dsl_state.py`
- Create: `tests/agent/__init__.py`
- Create: `tests/agent/test_dsl_state.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/agent/test_dsl_state.py
"""Tests for DSL state management."""
from atlas.agent.dsl_state import DSLState
from atlas.dsl.operators import Op, DSL_0
from atlas.dsl.expr import BinOp, UnaryOp, Var


def test_initial_state():
    state = DSLState()
    assert state.operators == DSL_0
    assert len(state.concepts) == 0
    assert len(state.extensions) == 0


def test_add_concept():
    state = DSLState()
    cos2 = BinOp(Op.MUL, UnaryOp(Op.COS, Var("x_0")), UnaryOp(Op.COS, Var("x_0")))
    state.add_concept("concept_cos2", cos2)
    assert "concept_cos2" in state.concepts
    assert state.concepts["concept_cos2"] == cos2


def test_add_extension():
    state = DSLState()
    state.add_extension(
        name="prob_mode",
        ext_type="prob_mode",
        definition={"desc": "enable P(y|x) search"},
        trigger="D1=stochastic",
    )
    assert len(state.extensions) == 1
    assert state.extensions[0]["name"] == "prob_mode"


def test_mdl_cost():
    state = DSLState()
    cost_before = state.mdl_cost()
    cos2 = BinOp(Op.MUL, UnaryOp(Op.COS, Var("x_0")), UnaryOp(Op.COS, Var("x_0")))
    state.add_concept("concept_cos2", cos2)
    cost_after = state.mdl_cost()
    assert cost_after > cost_before  # adding a concept increases DSL complexity


def test_snapshot_and_restore():
    state = DSLState()
    snap = state.snapshot()
    state.add_concept("c", Var("x_0"))
    assert len(state.concepts) == 1
    state.restore(snap)
    assert len(state.concepts) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement dsl_state.py**

```python
# atlas/agent/__init__.py
"""ATLAS agent: orchestration of the discovery loop."""
```

```python
# atlas/agent/dsl_state.py
"""DSL state: tracks current operators, concepts, and extensions."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from atlas.dsl.operators import Op, DSL_0
from atlas.dsl.expr import Expr


class DSLState:
    """Mutable state of the DSL as it grows through discovery."""

    def __init__(self):
        self.operators: frozenset[Op] = DSL_0
        self.concepts: dict[str, Expr] = {}
        self.extensions: list[dict] = []
        self._history: list[dict] = []

    def add_concept(self, name: str, expr: Expr) -> None:
        """Add a discovered concept (reusable subexpression) to the DSL."""
        self.concepts[name] = expr
        self._history.append({
            "action": "add_concept",
            "name": name,
        })

    def add_extension(self, name: str, ext_type: str,
                      definition: dict, trigger: str) -> None:
        """Add a DSL extension (new type, new mode, etc.)."""
        ext = {
            "name": name,
            "type": ext_type,
            "definition": definition,
            "trigger": trigger,
        }
        self.extensions.append(ext)
        self._history.append({
            "action": "add_extension",
            "name": name,
            "type": ext_type,
        })

    def mdl_cost(self) -> float:
        """Compute the description length cost of the current DSL.

        Cost = base operators + concept definitions + extension definitions.
        """
        cost = float(len(self.operators))
        for expr in self.concepts.values():
            cost += expr.size()
        for ext in self.extensions:
            cost += 5.0  # fixed cost per extension type
        return cost

    def snapshot(self) -> dict:
        """Save current state for later restoration."""
        return {
            "operators": self.operators,
            "concepts": dict(self.concepts),
            "extensions": list(self.extensions),
        }

    def restore(self, snap: dict) -> None:
        """Restore state from a snapshot."""
        self.operators = snap["operators"]
        self.concepts = dict(snap["concepts"])
        self.extensions = list(snap["extensions"])

    @property
    def history(self) -> list[dict]:
        return list(self._history)
```

- [ ] **Step 4: Run tests**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/agent/test_dsl_state.py -v`

- [ ] **Step 5: Commit**

```bash
git add atlas/agent/ tests/agent/
git commit -m "feat: DSL state management with snapshot/restore"
```

---

## Task 8: ATLAS Agent Loop

**Files:**
- Create: `atlas/agent/atlas_agent.py`
- Create: `tests/agent/test_atlas_agent.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/agent/test_atlas_agent.py
"""Tests for the ATLAS single-agent main loop."""
import numpy as np
from atlas.agent.atlas_agent import ATLASAgent, AgentConfig, EpochResult


def test_agent_creation():
    agent = ATLASAgent(
        env_ids=["ENV_10", "ENV_11"],
        config=AgentConfig(max_epochs=2, n_samples_per_knob=5),
    )
    assert agent.env_ids == ["ENV_10", "ENV_11"]
    assert len(agent.dsl_state.concepts) == 0


def test_agent_config_defaults():
    cfg = AgentConfig()
    assert cfg.max_epochs == 10
    assert cfg.r_squared_threshold == 0.95
    assert cfg.n_samples_per_knob == 10


def test_collect_data():
    agent = ATLASAgent(
        env_ids=["ENV_10"],
        config=AgentConfig(n_samples_per_knob=5),
    )
    agent.collect_data()
    assert "ENV_10" in agent.datasets
    assert len(agent.datasets["ENV_10"]) > 0


def test_run_epoch_returns_result():
    agent = ATLASAgent(
        env_ids=["ENV_10"],
        config=AgentConfig(
            max_epochs=1,
            n_samples_per_knob=5,
            sr_niterations=5,
            sr_maxsize=8,
        ),
    )
    agent.collect_data()
    result = agent.run_epoch()
    assert isinstance(result, EpochResult)
    assert result.epoch == 0
    assert isinstance(result.formulas_found, int)
    assert isinstance(result.diagnostics, dict)


def test_agent_output_structure():
    agent = ATLASAgent(
        env_ids=["ENV_10", "ENV_11"],
        config=AgentConfig(max_epochs=1, n_samples_per_knob=5,
                           sr_niterations=5, sr_maxsize=8),
    )
    output = agent.run()
    assert "formulas" in output
    assert "constants" in output
    assert "concepts" in output
    assert "diagnostics" in output
    assert "dsl_state" in output
    assert "fit_metrics" in output
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement atlas_agent.py**

```python
# atlas/agent/atlas_agent.py
"""ATLAS single-agent main loop.

Orchestrates the 5-step discovery cycle:
  Step 1 (Solve):     Run SR on assigned experiments
  Step 2 (Extract):   Mine frequent subexpressions as concepts
  Step 3 (Diagnose):  Run D1-D5 diagnostics on failed experiments
  Step 4 (Extend):    RGDE — deferred to Plan 3
  Step 5 (Unify):     PSLQ constant unification (local, within agent's experiments)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from atlas.agent.dsl_state import DSLState
from atlas.data.dataset import ExperimentDataset
from atlas.environments.registry import get_environment
from atlas.sr.pysr_wrapper import SRConfig, run_sr, SRResult
from atlas.sr.formula_store import FormulaStore
from atlas.analysis.concepts import extract_concepts
from atlas.analysis.diagnostics import run_all_diagnostics, DiagnosticResult
from atlas.analysis.pslq_unifier import unify_constants
from atlas.dsl.expr import Expr
from atlas.types import FitMetrics

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for an ATLAS agent."""
    max_epochs: int = 10
    r_squared_threshold: float = 0.95
    n_samples_per_knob: int = 10
    test_fraction: float = 0.2
    sr_niterations: int = 40
    sr_populations: int = 15
    sr_maxsize: int = 25
    sr_timeout: int = 300
    seed: int = 42
    min_concept_occurrences: int = 2


@dataclass
class EpochResult:
    """Summary of one epoch's results."""
    epoch: int
    formulas_found: int
    concepts_found: int
    diagnostics: dict[str, list[DiagnosticResult]]
    constants_unified: int
    converged_envs: list[str]
    failed_envs: list[str]


class ATLASAgent:
    """Single ATLAS agent: runs the discovery loop on assigned experiments."""

    def __init__(self, env_ids: list[str], config: AgentConfig | None = None):
        self.env_ids = list(env_ids)
        self.config = config or AgentConfig()
        self.dsl_state = DSLState()
        self.formula_store = FormulaStore()
        self.datasets: dict[str, ExperimentDataset] = {}
        self._epoch = 0

    def collect_data(self) -> None:
        """Step 0: Collect experiment data for all assigned environments."""
        for env_id in self.env_ids:
            env = get_environment(env_id)
            ds = ExperimentDataset.from_env(
                env,
                n_samples_per_knob=self.config.n_samples_per_knob,
                seed=self.config.seed,
            )
            self.datasets[env_id] = ds
            logger.info(f"Collected {len(ds)} samples for {env_id}")

    def run_epoch(self) -> EpochResult:
        """Run one epoch of the ATLAS loop (Steps 1-3, 5)."""
        epoch = self._epoch
        self._epoch += 1

        # Step 1: Solve — SR on each experiment
        formulas_found = 0
        converged_envs = []
        failed_envs = []

        for env_id in self.env_ids:
            ds = self.datasets.get(env_id)
            if ds is None or len(ds) == 0:
                failed_envs.append(env_id)
                continue

            train, test = ds.split(self.config.test_fraction, self.config.seed)
            if len(train) < 5:
                failed_envs.append(env_id)
                continue

            X_train = train.knob_array()
            det_name = train.detector_names[0]
            y_train = train.detector_array(det_name)

            # Only handle scalar outputs for now
            if y_train.ndim > 1:
                # For array outputs, use mean as a scalar proxy
                # (proper handling requires per-position SR, future work)
                y_train = np.mean(y_train, axis=1)

            sr_config = SRConfig.from_dsl(
                self.dsl_state.operators,
                niterations=self.config.sr_niterations,
                maxsize=self.config.sr_maxsize,
                timeout_seconds=self.config.sr_timeout,
                random_state=self.config.seed + epoch,
            )

            try:
                result = run_sr(X_train, y_train, train.knob_names, sr_config)
            except ImportError:
                logger.warning("PySR not installed, skipping SR for %s", env_id)
                failed_envs.append(env_id)
                continue
            except Exception as e:
                logger.warning("SR failed for %s: %s", env_id, e)
                failed_envs.append(env_id)
                continue

            if result.best_formula is not None:
                # Evaluate on test set
                X_test = test.knob_array()
                y_test = test.detector_array(det_name)
                if y_test.ndim > 1:
                    y_test = np.mean(y_test, axis=1)

                test_metrics = _evaluate_formula(result.best_formula, X_test, y_test,
                                                 train.knob_names)
                self.formula_store.add(env_id, result.best_formula, test_metrics)
                formulas_found += 1

                if test_metrics.r_squared > self.config.r_squared_threshold:
                    converged_envs.append(env_id)
                else:
                    failed_envs.append(env_id)
            else:
                failed_envs.append(env_id)

        # Step 2: Extract — concept mining
        all_exprs = []
        for env_id in self.env_ids:
            best = self.formula_store.get_best(env_id)
            if best is not None:
                all_exprs.append(best.expr)

        concepts = extract_concepts(
            all_exprs,
            min_occurrences=self.config.min_concept_occurrences,
        )
        concepts_found = 0
        for concept in concepts:
            if concept.name not in self.dsl_state.concepts:
                self.dsl_state.add_concept(concept.name, concept.expr)
                concepts_found += 1

        # Step 3: Diagnose — run diagnostics on failed experiments
        all_diagnostics: dict[str, list[DiagnosticResult]] = {}
        for env_id in failed_envs:
            ds = self.datasets.get(env_id)
            if ds is None:
                continue

            best = self.formula_store.get_best(env_id)
            best_r2 = best.fit.r_squared if best else -1.0

            # Compute residuals if we have a formula
            residuals = None
            if best is not None:
                X = ds.knob_array()
                y = ds.detector_array(ds.detector_names[0])
                if y.ndim > 1:
                    y = np.mean(y, axis=1)
                try:
                    y_pred = np.array([
                        best.expr.evaluate(dict(zip(ds.knob_names, row)))
                        for row in X
                    ])
                    residuals = y - y_pred
                except Exception:
                    pass

            diag = run_all_diagnostics(ds, best_r2, residuals)
            all_diagnostics[env_id] = diag

        # Step 5: Unify — local constant unification
        all_constants = {}
        for env_id in self.env_ids:
            best = self.formula_store.get_best(env_id)
            if best is not None:
                from atlas.sr.formula_store import _extract_constants
                consts = _extract_constants(best.expr)
                for i, c in enumerate(consts):
                    if abs(c) > 1e-6 and abs(c) != 1.0:  # skip trivial
                        all_constants[f"{env_id}:C{i}"] = c

        constants_unified = 0
        if len(all_constants) >= 2:
            unified = unify_constants(all_constants)
            constants_unified = len(unified)

        return EpochResult(
            epoch=epoch,
            formulas_found=formulas_found,
            concepts_found=concepts_found,
            diagnostics=all_diagnostics,
            constants_unified=constants_unified,
            converged_envs=converged_envs,
            failed_envs=failed_envs,
        )

    def run(self) -> dict:
        """Run the full ATLAS loop until convergence or budget exhaustion."""
        if not self.datasets:
            self.collect_data()

        results = []
        for epoch in range(self.config.max_epochs):
            result = self.run_epoch()
            results.append(result)
            logger.info(
                f"Epoch {result.epoch}: {result.formulas_found} formulas, "
                f"{result.concepts_found} concepts, "
                f"{len(result.converged_envs)} converged, "
                f"{len(result.failed_envs)} failed"
            )

            # Check convergence
            all_converged = set()
            for r in results:
                all_converged.update(r.converged_envs)
            if all_converged >= set(self.env_ids):
                logger.info("All experiments converged!")
                break

        # Build output
        formulas_out = {}
        fit_metrics_out = {}
        for env_id in self.env_ids:
            best = self.formula_store.get_best(env_id)
            if best is not None:
                from atlas.dsl.serialize import to_str
                formulas_out[env_id] = to_str(best.expr)
                fit_metrics_out[env_id] = {
                    "r_squared": best.fit.r_squared,
                    "residual_var": best.fit.residual_var,
                    "mdl": best.fit.mdl,
                }

        all_constants = {}
        for env_id in self.env_ids:
            best = self.formula_store.get_best(env_id)
            if best is not None:
                from atlas.sr.formula_store import _extract_constants
                consts = _extract_constants(best.expr)
                for i, c in enumerate(consts):
                    if abs(c) > 1e-6 and abs(c) != 1.0:
                        all_constants[f"{env_id}:C{i}"] = c

        all_diag = {}
        for r in results:
            all_diag.update(r.diagnostics)

        return {
            "formulas": formulas_out,
            "constants": all_constants,
            "concepts": {k: str(v) for k, v in self.dsl_state.concepts.items()},
            "diagnostics": {
                env_id: [{"id": d.diagnostic_id, "triggered": d.triggered,
                          "confidence": d.confidence}
                         for d in diags]
                for env_id, diags in all_diag.items()
            },
            "dsl_state": {
                "n_operators": len(self.dsl_state.operators),
                "n_concepts": len(self.dsl_state.concepts),
                "n_extensions": len(self.dsl_state.extensions),
                "mdl_cost": self.dsl_state.mdl_cost(),
            },
            "fit_metrics": fit_metrics_out,
            "epochs_run": len(results),
        }


def _evaluate_formula(expr: Expr, X: np.ndarray, y: np.ndarray,
                      var_names: list[str]) -> FitMetrics:
    """Evaluate a formula on data and compute fit metrics."""
    try:
        y_pred = np.array([
            expr.evaluate(dict(zip(var_names, row)))
            for row in X
        ])
    except Exception:
        return FitMetrics(r_squared=-1.0, residual_var=float("inf"), mdl=float("inf"))

    # Handle NaN/inf
    valid = np.isfinite(y_pred) & np.isfinite(y)
    if np.sum(valid) < 5:
        return FitMetrics(r_squared=-1.0, residual_var=float("inf"), mdl=float("inf"))

    y_valid = y[valid]
    y_pred_valid = y_pred[valid]

    ss_res = np.sum((y_valid - y_pred_valid) ** 2)
    ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
    r_squared = 1 - ss_res / max(ss_tot, 1e-30)

    residual_var = float(np.var(y_valid - y_pred_valid))
    mdl = float(expr.size())

    return FitMetrics(r_squared=float(r_squared), residual_var=residual_var, mdl=mdl)
```

- [ ] **Step 4: Run tests**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/agent/test_atlas_agent.py -v -m "not slow"`

Note: The `test_run_epoch_returns_result` and `test_agent_output_structure` tests require PySR. If PySR is not installed, these tests should gracefully handle the ImportError. Mark them with `@pytest.mark.slow` if needed, and verify the non-SR tests pass.

- [ ] **Step 5: Update tests to handle missing PySR**

If PySR is not installed, update the test file to mark SR-dependent tests:

```python
# Add at top of test file:
import pytest

try:
    import pysr
    HAS_PYSR = True
except ImportError:
    HAS_PYSR = False

# Then mark SR-dependent tests:
@pytest.mark.skipif(not HAS_PYSR, reason="PySR not installed")
def test_run_epoch_returns_result():
    ...

@pytest.mark.skipif(not HAS_PYSR, reason="PySR not installed")
def test_agent_output_structure():
    ...
```

- [ ] **Step 6: Run all tests**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/ -v --tb=short`

- [ ] **Step 7: Commit**

```bash
git add atlas/agent/atlas_agent.py tests/agent/test_atlas_agent.py
git commit -m "feat: ATLAS single-agent main loop with SR, concepts, diagnostics, unification"
```

---

## Summary

After completing all 8 tasks, the project has:

- **Data collection**: `ExperimentDataset` with grid/random sweeping, train/test split
- **Symbolic regression**: PySR wrapper with expression parser, configurable from DSL
- **Formula management**: Store with Pareto front, constant extraction
- **Concept extraction**: DreamCoder-style frequent subexpression mining with MDL savings
- **Diagnostics**: D1 (stochasticity), D2 (discreteness), D4 (residual structure) — D3/D5 deferred
- **Constant unification**: PSLQ-based constant relation discovery with sign separation
- **Agent loop**: Full orchestration of Steps 1-3, 5 (Step 4 RGDE deferred to Plan 3)

This delivers a working single-agent ATLAS that can:
1. Collect data from any assigned experiments
2. Discover symbolic formulas via PySR
3. Extract reusable concepts
4. Diagnose why some experiments fail
5. Find shared constants across experiments

**Plan 3** will add SciNet + RGDE (Step 4: DSL extension when SR fails).
**Plan 4** will add multi-agent consensus + Unifier.
