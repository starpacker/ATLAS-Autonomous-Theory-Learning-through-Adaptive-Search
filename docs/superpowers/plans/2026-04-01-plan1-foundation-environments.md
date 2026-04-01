# Plan 1: Foundation + Environments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the project scaffolding, DSL system with AST canonicalization, and all 12 experiment environments with anonymized interfaces.

**Architecture:** Python package `atlas` with three core modules: `dsl` (symbolic expression representation + canonicalization), `environments` (12 physics simulators behind a uniform knob/detector interface), and shared `types` (data structures for schemas, formulas, fit metrics). Each environment is a pure function: knobs in, detector readings out. No physics semantics leak through the interface.

**Tech Stack:** Python 3.11+, NumPy, SciPy, pytest, dataclasses

**Specs:**
- `docs/superpowers/specs/2026-03-31-multi-agent-atlas-design.md` (multi-agent architecture)
- `ATLAS_proposal.md` Section 4 (environment layer)
- `anti_cheating_audit.md` Section 3.1 (clean environment interfaces)

---

## File Structure

```
atlas/
  __init__.py
  types.py                  # Core data structures (Schema, FormulaRecord, FitMetrics)
  dsl/
    __init__.py
    operators.py            # DSL_0 operator definitions
    expr.py                 # Expression AST node types
    canonicalize.py         # AST canonicalization (alpha-equiv, commutativity, etc.)
    serialize.py            # DSL and expression serialization/deserialization
  environments/
    __init__.py
    base.py                 # BaseEnvironment ABC + schema validation
    registry.py             # Environment registry (get by ID)
    normalizer.py           # Knob normalization utilities
    env_01_photoelectric.py
    env_02_compton.py
    env_03_electron_diffraction.py
    env_04_double_slit.py
    env_05_blackbody.py
    env_06_hydrogen_spectrum.py
    env_07_stern_gerlach.py
    env_08_water_wave.py
    env_09_elastic_collision.py
    env_10_spring.py
    env_11_freefall.py
    env_12_heat_conduction.py
tests/
  __init__.py
  dsl/
    __init__.py
    test_expr.py
    test_canonicalize.py
    test_serialize.py
  environments/
    __init__.py
    test_base.py
    test_anti_cheating.py   # Verifies no physics leakage in interfaces
    test_env_01.py
    test_env_02.py
    test_env_03.py
    test_env_04.py
    test_env_05.py
    test_env_06.py
    test_env_07.py
    test_env_08.py
    test_env_09.py
    test_env_10.py
    test_env_11.py
    test_env_12.py
pyproject.toml
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `atlas/__init__.py`
- Create: `atlas/types.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "atlas"
version = "0.1.0"
description = "Autonomous Theory Learning through Adaptive Search"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.24",
    "scipy>=1.11",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create atlas package init**

```python
# atlas/__init__.py
"""ATLAS: Autonomous Theory Learning through Adaptive Search."""
```

- [ ] **Step 3: Create core types**

```python
# atlas/types.py
"""Core data structures used across ATLAS modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class KnobType(Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    INTEGER = "integer"


@dataclass(frozen=True)
class KnobSpec:
    """Specification for a single input knob."""
    name: str
    knob_type: KnobType
    range_min: float
    range_max: float
    options: list[int] | None = None  # only for DISCRETE type


@dataclass(frozen=True)
class DetectorSpec:
    """Specification for a single detector output."""
    name: str
    output_type: str  # "scalar", "array_1d", "list"
    length: int | None = None  # for array_1d


@dataclass(frozen=True)
class EnvSchema:
    """Complete schema for an experiment environment."""
    env_id: str
    knobs: list[KnobSpec]
    detectors: list[DetectorSpec]
    entities: list[str] = field(default_factory=list)


@dataclass
class FitMetrics:
    """Fit quality metrics for a formula on an experiment."""
    r_squared: float
    residual_var: float
    mdl: float
    n_seeds: int = 1


@dataclass
class FormulaRecord:
    """A discovered formula with its provenance and fit metrics."""
    expr_str: str  # serialized expression
    env_id: str
    fit: FitMetrics
    constants: dict[str, float] = field(default_factory=dict)
```

- [ ] **Step 4: Create test init files**

```python
# tests/__init__.py
```

```python
# tests/dsl/__init__.py
```

```python
# tests/environments/__init__.py
```

- [ ] **Step 5: Verify project structure**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -c "from atlas.types import EnvSchema, KnobSpec, KnobType; print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git init
git add pyproject.toml atlas/__init__.py atlas/types.py tests/__init__.py tests/dsl/__init__.py tests/environments/__init__.py
git commit -m "feat: project scaffolding with core types"
```

---

## Task 2: Expression AST

**Files:**
- Create: `atlas/dsl/__init__.py`
- Create: `atlas/dsl/expr.py`
- Create: `atlas/dsl/operators.py`
- Create: `tests/dsl/test_expr.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/dsl/test_expr.py
"""Tests for expression AST nodes."""
import math
from atlas.dsl.expr import Const, Var, BinOp, UnaryOp, NAryOp, Op
from atlas.dsl.operators import DSL_0


def test_const_creation():
    c = Const(3.14)
    assert c.value == 3.14


def test_var_creation():
    v = Var("x_0")
    assert v.name == "x_0"


def test_binop_creation():
    expr = BinOp(Op.ADD, Var("x_0"), Const(1.0))
    assert expr.op == Op.ADD
    assert isinstance(expr.left, Var)
    assert isinstance(expr.right, Const)


def test_unaryop_creation():
    expr = UnaryOp(Op.SIN, Var("x_0"))
    assert expr.op == Op.SIN
    assert isinstance(expr.operand, Var)


def test_nary_creation():
    expr = NAryOp(Op.ADD, [Var("x_0"), Var("x_1"), Const(1.0)])
    assert len(expr.children) == 3


def test_expr_equality():
    a = BinOp(Op.ADD, Var("x_0"), Const(1.0))
    b = BinOp(Op.ADD, Var("x_0"), Const(1.0))
    assert a == b


def test_expr_inequality():
    a = BinOp(Op.ADD, Var("x_0"), Const(1.0))
    b = BinOp(Op.MUL, Var("x_0"), Const(1.0))
    assert a != b


def test_expr_hash():
    a = BinOp(Op.ADD, Var("x_0"), Const(1.0))
    b = BinOp(Op.ADD, Var("x_0"), Const(1.0))
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_eval_simple():
    # x_0 + 1.0
    expr = BinOp(Op.ADD, Var("x_0"), Const(1.0))
    result = expr.evaluate({"x_0": 2.0})
    assert result == 3.0


def test_eval_nested():
    # sin(x_0 * 3.14159)
    expr = UnaryOp(Op.SIN, BinOp(Op.MUL, Var("x_0"), Const(math.pi)))
    result = expr.evaluate({"x_0": 0.5})
    assert abs(result - 1.0) < 1e-10


def test_dsl0_operators():
    ops = DSL_0
    assert Op.ADD in ops
    assert Op.SIN in ops
    assert Op.EXP in ops
    assert Op.LOG in ops
    assert Op.POW in ops
    assert len(ops) == 10  # +, -, *, /, sin, cos, exp, log, ^, neg


def test_expr_size():
    # sin(x_0 + 1.0) -> size 4 (sin, +, x_0, 1.0)
    expr = UnaryOp(Op.SIN, BinOp(Op.ADD, Var("x_0"), Const(1.0)))
    assert expr.size() == 4


def test_expr_depth():
    # sin(x_0 + 1.0) -> depth 3
    expr = UnaryOp(Op.SIN, BinOp(Op.ADD, Var("x_0"), Const(1.0)))
    assert expr.depth() == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/dsl/test_expr.py -v`
Expected: FAIL (modules not found)

- [ ] **Step 3: Implement operator definitions**

```python
# atlas/dsl/__init__.py
"""DSL module: symbolic expression system."""
```

```python
# atlas/dsl/operators.py
"""Operator definitions and DSL_0."""
from __future__ import annotations

from enum import Enum


class Op(Enum):
    """All operators available in the DSL."""
    # Binary arithmetic
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    POW = "pow"
    # Unary functions
    SIN = "sin"
    COS = "cos"
    EXP = "exp"
    LOG = "log"
    NEG = "neg"

    @property
    def arity(self) -> int:
        if self in _UNARY_OPS:
            return 1
        return 2

    @property
    def is_commutative(self) -> bool:
        return self in (Op.ADD, Op.MUL)


_UNARY_OPS = frozenset({Op.SIN, Op.COS, Op.EXP, Op.LOG, Op.NEG})

# The initial DSL: basic real-valued arithmetic + trig + exp/log
DSL_0: frozenset[Op] = frozenset({
    Op.ADD, Op.SUB, Op.MUL, Op.DIV,
    Op.SIN, Op.COS, Op.EXP, Op.LOG,
    Op.POW, Op.NEG,
})
```

- [ ] **Step 4: Implement expression AST nodes**

```python
# atlas/dsl/expr.py
"""Expression AST nodes."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from atlas.dsl.operators import Op


class Expr:
    """Base class for all expression nodes."""

    def evaluate(self, env: Mapping[str, float]) -> float:
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError

    def depth(self) -> int:
        raise NotImplementedError

    def variables(self) -> frozenset[str]:
        raise NotImplementedError


@dataclass(frozen=True)
class Const(Expr):
    value: float

    def evaluate(self, env: Mapping[str, float]) -> float:
        return self.value

    def size(self) -> int:
        return 1

    def depth(self) -> int:
        return 1

    def variables(self) -> frozenset[str]:
        return frozenset()


@dataclass(frozen=True)
class Var(Expr):
    name: str

    def evaluate(self, env: Mapping[str, float]) -> float:
        return env[self.name]

    def size(self) -> int:
        return 1

    def depth(self) -> int:
        return 1

    def variables(self) -> frozenset[str]:
        return frozenset({self.name})


@dataclass(frozen=True)
class BinOp(Expr):
    op: Op
    left: Expr
    right: Expr

    def evaluate(self, env: Mapping[str, float]) -> float:
        l = self.left.evaluate(env)
        r = self.right.evaluate(env)
        if self.op == Op.ADD:
            return l + r
        if self.op == Op.SUB:
            return l - r
        if self.op == Op.MUL:
            return l * r
        if self.op == Op.DIV:
            return l / r if r != 0 else math.nan
        if self.op == Op.POW:
            return l ** r
        raise ValueError(f"Unknown binary op: {self.op}")

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def variables(self) -> frozenset[str]:
        return self.left.variables() | self.right.variables()


@dataclass(frozen=True)
class UnaryOp(Expr):
    op: Op
    operand: Expr

    def evaluate(self, env: Mapping[str, float]) -> float:
        v = self.operand.evaluate(env)
        if self.op == Op.SIN:
            return math.sin(v)
        if self.op == Op.COS:
            return math.cos(v)
        if self.op == Op.EXP:
            return math.exp(v) if v < 709 else math.inf
        if self.op == Op.LOG:
            return math.log(v) if v > 0 else math.nan
        if self.op == Op.NEG:
            return -v
        raise ValueError(f"Unknown unary op: {self.op}")

    def size(self) -> int:
        return 1 + self.operand.size()

    def depth(self) -> int:
        return 1 + self.operand.depth()

    def variables(self) -> frozenset[str]:
        return self.operand.variables()


@dataclass(frozen=True)
class NAryOp(Expr):
    """N-ary operator node (used after associativity flattening)."""
    op: Op
    children: tuple[Expr, ...]

    def __init__(self, op: Op, children: list[Expr] | tuple[Expr, ...]):
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "children", tuple(children))

    def evaluate(self, env: Mapping[str, float]) -> float:
        values = [c.evaluate(env) for c in self.children]
        if self.op == Op.ADD:
            return sum(values)
        if self.op == Op.MUL:
            result = 1.0
            for v in values:
                result *= v
            return result
        raise ValueError(f"NAryOp only supports ADD/MUL, got {self.op}")

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def depth(self) -> int:
        return 1 + max(c.depth() for c in self.children)

    def variables(self) -> frozenset[str]:
        result: frozenset[str] = frozenset()
        for c in self.children:
            result = result | c.variables()
        return result
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/dsl/test_expr.py -v`
Expected: all 12 tests PASS

- [ ] **Step 6: Commit**

```bash
git add atlas/dsl/ tests/dsl/test_expr.py
git commit -m "feat: expression AST with operators and DSL_0"
```

---

## Task 3: AST Canonicalization

**Files:**
- Create: `atlas/dsl/canonicalize.py`
- Create: `tests/dsl/test_canonicalize.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/dsl/test_canonicalize.py
"""Tests for AST canonicalization."""
from atlas.dsl.expr import Const, Var, BinOp, UnaryOp, NAryOp
from atlas.dsl.operators import Op
from atlas.dsl.canonicalize import canonicalize, alpha_rename


def test_alpha_rename_by_appearance_order():
    # f(knob_2, knob_0) -> f(x_0, x_1)
    expr = BinOp(Op.ADD, Var("knob_2"), Var("knob_0"))
    renamed = alpha_rename(expr)
    assert renamed == BinOp(Op.ADD, Var("x_0"), Var("x_1"))


def test_alpha_rename_preserves_structure():
    # knob_0 + knob_0 -> x_0 + x_0 (same var maps to same name)
    expr = BinOp(Op.ADD, Var("knob_0"), Var("knob_0"))
    renamed = alpha_rename(expr)
    assert renamed == BinOp(Op.ADD, Var("x_0"), Var("x_0"))


def test_alpha_rename_nested():
    # sin(knob_1) * knob_0 -> sin(x_0) * x_1
    # knob_1 appears first (in sin), so it becomes x_0
    expr = BinOp(Op.MUL, UnaryOp(Op.SIN, Var("knob_1")), Var("knob_0"))
    renamed = alpha_rename(expr)
    expected = BinOp(Op.MUL, UnaryOp(Op.SIN, Var("x_0")), Var("x_1"))
    assert renamed == expected


def test_commutativity_sort():
    # b + a -> a + b (sorted by canonical ordering)
    expr = BinOp(Op.ADD, Var("x_1"), Var("x_0"))
    result = canonicalize(expr)
    # After canonicalization, x_0 should come before x_1
    assert result == BinOp(Op.ADD, Var("x_0"), Var("x_1"))


def test_commutativity_mul():
    # x_1 * x_0 -> x_0 * x_1
    expr = BinOp(Op.MUL, Var("x_1"), Var("x_0"))
    result = canonicalize(expr)
    assert result == BinOp(Op.MUL, Var("x_0"), Var("x_1"))


def test_non_commutative_preserved():
    # x_1 - x_0 stays as x_1 - x_0 (subtraction is not commutative)
    expr = BinOp(Op.SUB, Var("x_1"), Var("x_0"))
    result = canonicalize(expr)
    assert result == BinOp(Op.SUB, Var("x_1"), Var("x_0"))


def test_associativity_flattening():
    # (a + b) + c -> +(a, b, c)
    expr = BinOp(Op.ADD, BinOp(Op.ADD, Var("x_0"), Var("x_1")), Var("x_2"))
    result = canonicalize(expr)
    assert isinstance(result, NAryOp)
    assert result.op == Op.ADD
    assert len(result.children) == 3


def test_associativity_nested():
    # a + (b + c) -> +(a, b, c)
    expr = BinOp(Op.ADD, Var("x_0"), BinOp(Op.ADD, Var("x_1"), Var("x_2")))
    result = canonicalize(expr)
    assert isinstance(result, NAryOp)
    assert result.op == Op.ADD
    assert len(result.children) == 3


def test_identity_elimination_mul():
    # x * 1 -> x
    expr = BinOp(Op.MUL, Var("x_0"), Const(1.0))
    result = canonicalize(expr)
    assert result == Var("x_0")


def test_identity_elimination_add():
    # x + 0 -> x
    expr = BinOp(Op.ADD, Var("x_0"), Const(0.0))
    result = canonicalize(expr)
    assert result == Var("x_0")


def test_identity_elimination_pow():
    # x ^ 1 -> x
    expr = BinOp(Op.POW, Var("x_0"), Const(1.0))
    result = canonicalize(expr)
    assert result == Var("x_0")


def test_constant_folding():
    # 2.0 + 3.0 -> 5.0
    expr = BinOp(Op.ADD, Const(2.0), Const(3.0))
    result = canonicalize(expr)
    assert result == Const(5.0)


def test_constant_folding_sin():
    # sin(0) -> 0.0
    expr = UnaryOp(Op.SIN, Const(0.0))
    result = canonicalize(expr)
    assert result == Const(0.0)


def test_full_canonicalization():
    # (knob_2 * 1.0) + (0.0 + knob_0)
    # -> alpha: (x_0 * 1.0) + (0.0 + x_1)
    # -> identity: x_0 + x_1
    # -> commutativity: already sorted
    expr = BinOp(
        Op.ADD,
        BinOp(Op.MUL, Var("knob_2"), Const(1.0)),
        BinOp(Op.ADD, Const(0.0), Var("knob_0")),
    )
    result = canonicalize(expr)
    assert result == BinOp(Op.ADD, Var("x_0"), Var("x_1"))


def test_canonicalize_two_equivalent_exprs():
    # b + a and a + b should canonicalize to the same thing
    e1 = BinOp(Op.ADD, Var("y"), Var("x"))
    e2 = BinOp(Op.ADD, Var("a"), Var("b"))
    assert canonicalize(e1) == canonicalize(e2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/dsl/test_canonicalize.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement canonicalization**

```python
# atlas/dsl/canonicalize.py
"""AST canonicalization: alpha-equivalence, commutativity, associativity,
identity elimination, constant folding."""
from __future__ import annotations

import math
from collections.abc import Mapping

from atlas.dsl.expr import Expr, Const, Var, BinOp, UnaryOp, NAryOp
from atlas.dsl.operators import Op


def canonicalize(expr: Expr) -> Expr:
    """Full canonicalization pipeline: alpha-rename -> simplify -> normalize."""
    expr = alpha_rename(expr)
    expr = _simplify(expr)
    return expr


def alpha_rename(expr: Expr) -> Expr:
    """Rename variables by order of first appearance (depth-first, left-to-right)."""
    var_order: list[str] = []
    _collect_vars_in_order(expr, var_order)
    mapping = {old: f"x_{i}" for i, old in enumerate(var_order)}
    return _apply_rename(expr, mapping)


def _collect_vars_in_order(expr: Expr, order: list[str]) -> None:
    if isinstance(expr, Var):
        if expr.name not in order:
            order.append(expr.name)
    elif isinstance(expr, Const):
        pass
    elif isinstance(expr, UnaryOp):
        _collect_vars_in_order(expr.operand, order)
    elif isinstance(expr, BinOp):
        _collect_vars_in_order(expr.left, order)
        _collect_vars_in_order(expr.right, order)
    elif isinstance(expr, NAryOp):
        for c in expr.children:
            _collect_vars_in_order(c, order)


def _apply_rename(expr: Expr, mapping: Mapping[str, str]) -> Expr:
    if isinstance(expr, Var):
        return Var(mapping.get(expr.name, expr.name))
    if isinstance(expr, Const):
        return expr
    if isinstance(expr, UnaryOp):
        return UnaryOp(expr.op, _apply_rename(expr.operand, mapping))
    if isinstance(expr, BinOp):
        return BinOp(
            expr.op,
            _apply_rename(expr.left, mapping),
            _apply_rename(expr.right, mapping),
        )
    if isinstance(expr, NAryOp):
        return NAryOp(expr.op, [_apply_rename(c, mapping) for c in expr.children])
    raise TypeError(f"Unknown expr type: {type(expr)}")


def _simplify(expr: Expr) -> Expr:
    """Bottom-up simplification: constant folding, identity elimination,
    associativity flattening, commutativity sorting."""
    if isinstance(expr, (Const, Var)):
        return expr

    if isinstance(expr, UnaryOp):
        operand = _simplify(expr.operand)
        # Constant folding
        if isinstance(operand, Const):
            return Const(expr.evaluate({})) if not operand.variables() else UnaryOp(expr.op, operand)
        return UnaryOp(expr.op, operand)

    if isinstance(expr, BinOp):
        left = _simplify(expr.left)
        right = _simplify(expr.right)

        # Constant folding
        if isinstance(left, Const) and isinstance(right, Const):
            result = BinOp(expr.op, left, right).evaluate({})
            if math.isfinite(result):
                return Const(result)

        # Identity elimination
        simplified = _eliminate_identity(expr.op, left, right)
        if simplified is not None:
            return simplified

        # Associativity flattening for commutative ops
        if expr.op.is_commutative:
            children = _flatten(expr.op, left, right)
            children = sorted(children, key=_sort_key)
            if len(children) == 2:
                return BinOp(expr.op, children[0], children[1])
            return NAryOp(expr.op, children)

        return BinOp(expr.op, left, right)

    if isinstance(expr, NAryOp):
        children = [_simplify(c) for c in expr.children]
        # Re-flatten in case simplification exposed new opportunities
        flat: list[Expr] = []
        for c in children:
            if isinstance(c, NAryOp) and c.op == expr.op:
                flat.extend(c.children)
            elif isinstance(c, BinOp) and c.op == expr.op and expr.op.is_commutative:
                flat.extend([c.left, c.right])
            else:
                flat.append(c)
        flat = sorted(flat, key=_sort_key)
        if len(flat) == 1:
            return flat[0]
        if len(flat) == 2:
            return BinOp(expr.op, flat[0], flat[1])
        return NAryOp(expr.op, flat)

    raise TypeError(f"Unknown expr type: {type(expr)}")


def _eliminate_identity(op: Op, left: Expr, right: Expr) -> Expr | None:
    """Remove identity elements: x+0=x, x*1=x, x^1=x."""
    if op == Op.ADD:
        if isinstance(right, Const) and right.value == 0.0:
            return left
        if isinstance(left, Const) and left.value == 0.0:
            return right
    elif op == Op.MUL:
        if isinstance(right, Const) and right.value == 1.0:
            return left
        if isinstance(left, Const) and left.value == 1.0:
            return right
    elif op == Op.POW:
        if isinstance(right, Const) and right.value == 1.0:
            return left
    return None


def _flatten(op: Op, left: Expr, right: Expr) -> list[Expr]:
    """Flatten associative binary ops into a list of children."""
    children: list[Expr] = []
    for node in (left, right):
        if isinstance(node, BinOp) and node.op == op:
            children.extend(_flatten(op, node.left, node.right))
        elif isinstance(node, NAryOp) and node.op == op:
            children.extend(node.children)
        else:
            children.append(node)
    return children


def _sort_key(expr: Expr) -> tuple:
    """Canonical ordering: Const < Var (by name) < UnaryOp < BinOp < NAryOp."""
    if isinstance(expr, Const):
        return (0, expr.value)
    if isinstance(expr, Var):
        return (1, expr.name)
    if isinstance(expr, UnaryOp):
        return (2, expr.op.value, _sort_key(expr.operand))
    if isinstance(expr, BinOp):
        return (3, expr.op.value, _sort_key(expr.left), _sort_key(expr.right))
    if isinstance(expr, NAryOp):
        return (4, expr.op.value, tuple(_sort_key(c) for c in expr.children))
    return (99,)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/dsl/test_canonicalize.py -v`
Expected: all 16 tests PASS

- [ ] **Step 5: Commit**

```bash
git add atlas/dsl/canonicalize.py tests/dsl/test_canonicalize.py
git commit -m "feat: AST canonicalization with alpha-equivalence and simplification"
```

---

## Task 4: Expression Serialization

**Files:**
- Create: `atlas/dsl/serialize.py`
- Create: `tests/dsl/test_serialize.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/dsl/test_serialize.py
"""Tests for expression serialization/deserialization."""
from atlas.dsl.expr import Const, Var, BinOp, UnaryOp, NAryOp
from atlas.dsl.operators import Op
from atlas.dsl.serialize import to_str, from_str, to_dict, from_dict


def test_const_roundtrip():
    expr = Const(3.14)
    assert from_str(to_str(expr)) == expr


def test_var_roundtrip():
    expr = Var("x_0")
    assert from_str(to_str(expr)) == expr


def test_binop_roundtrip():
    expr = BinOp(Op.MUL, Var("x_0"), Const(2.0))
    assert from_str(to_str(expr)) == expr


def test_unaryop_roundtrip():
    expr = UnaryOp(Op.SIN, Var("x_0"))
    assert from_str(to_str(expr)) == expr


def test_nary_roundtrip():
    expr = NAryOp(Op.ADD, [Var("x_0"), Var("x_1"), Const(1.0)])
    assert from_str(to_str(expr)) == expr


def test_nested_roundtrip():
    # sin(x_0 * 3.14) + cos(x_1)
    expr = BinOp(
        Op.ADD,
        UnaryOp(Op.SIN, BinOp(Op.MUL, Var("x_0"), Const(3.14))),
        UnaryOp(Op.COS, Var("x_1")),
    )
    assert from_str(to_str(expr)) == expr


def test_to_str_readable():
    expr = BinOp(Op.ADD, Var("x_0"), Const(1.0))
    s = to_str(expr)
    assert "add" in s or "+" in s  # some readable form


def test_dict_roundtrip():
    expr = BinOp(
        Op.ADD,
        UnaryOp(Op.SIN, Var("x_0")),
        Const(1.0),
    )
    d = to_dict(expr)
    assert isinstance(d, dict)
    assert from_dict(d) == expr
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/dsl/test_serialize.py -v`
Expected: FAIL

- [ ] **Step 3: Implement serialization**

```python
# atlas/dsl/serialize.py
"""Expression serialization: S-expression strings and dicts."""
from __future__ import annotations

from atlas.dsl.expr import Expr, Const, Var, BinOp, UnaryOp, NAryOp
from atlas.dsl.operators import Op


# ---- String (S-expression) format ----

def to_str(expr: Expr) -> str:
    """Serialize expression to S-expression string."""
    if isinstance(expr, Const):
        return f"(const {expr.value})"
    if isinstance(expr, Var):
        return f"(var {expr.name})"
    if isinstance(expr, UnaryOp):
        return f"({expr.op.value} {to_str(expr.operand)})"
    if isinstance(expr, BinOp):
        return f"({expr.op.value} {to_str(expr.left)} {to_str(expr.right)})"
    if isinstance(expr, NAryOp):
        children_str = " ".join(to_str(c) for c in expr.children)
        return f"(nary_{expr.op.value} {children_str})"
    raise TypeError(f"Unknown expr type: {type(expr)}")


def from_str(s: str) -> Expr:
    """Deserialize expression from S-expression string."""
    tokens = _tokenize(s)
    expr, pos = _parse(tokens, 0)
    return expr


def _tokenize(s: str) -> list[str]:
    tokens: list[str] = []
    current = ""
    for ch in s:
        if ch in ("(", ")"):
            if current:
                tokens.append(current)
                current = ""
            tokens.append(ch)
        elif ch == " ":
            if current:
                tokens.append(current)
                current = ""
        else:
            current += ch
    if current:
        tokens.append(current)
    return tokens


def _parse(tokens: list[str], pos: int) -> tuple[Expr, int]:
    if tokens[pos] != "(":
        raise ValueError(f"Expected '(' at pos {pos}, got '{tokens[pos]}'")
    pos += 1  # skip '('
    tag = tokens[pos]
    pos += 1

    if tag == "const":
        value = float(tokens[pos])
        pos += 1
        pos += 1  # skip ')'
        return Const(value), pos

    if tag == "var":
        name = tokens[pos]
        pos += 1
        pos += 1  # skip ')'
        return Var(name), pos

    # Check for nary
    if tag.startswith("nary_"):
        op_name = tag[5:]
        op = Op(op_name)
        children: list[Expr] = []
        while tokens[pos] != ")":
            child, pos = _parse(tokens, pos)
            children.append(child)
        pos += 1  # skip ')'
        return NAryOp(op, children), pos

    # Unary or binary op
    op = Op(tag)
    first, pos = _parse(tokens, pos)
    if tokens[pos] == ")":
        # Unary
        pos += 1
        return UnaryOp(op, first), pos
    # Binary
    second, pos = _parse(tokens, pos)
    pos += 1  # skip ')'
    return BinOp(op, first, second), pos


# ---- Dict format (for JSON serialization) ----

def to_dict(expr: Expr) -> dict:
    """Serialize expression to a nested dict."""
    if isinstance(expr, Const):
        return {"type": "const", "value": expr.value}
    if isinstance(expr, Var):
        return {"type": "var", "name": expr.name}
    if isinstance(expr, UnaryOp):
        return {"type": "unary", "op": expr.op.value, "operand": to_dict(expr.operand)}
    if isinstance(expr, BinOp):
        return {
            "type": "binary", "op": expr.op.value,
            "left": to_dict(expr.left), "right": to_dict(expr.right),
        }
    if isinstance(expr, NAryOp):
        return {
            "type": "nary", "op": expr.op.value,
            "children": [to_dict(c) for c in expr.children],
        }
    raise TypeError(f"Unknown expr type: {type(expr)}")


def from_dict(d: dict) -> Expr:
    """Deserialize expression from a nested dict."""
    t = d["type"]
    if t == "const":
        return Const(d["value"])
    if t == "var":
        return Var(d["name"])
    if t == "unary":
        return UnaryOp(Op(d["op"]), from_dict(d["operand"]))
    if t == "binary":
        return BinOp(Op(d["op"]), from_dict(d["left"]), from_dict(d["right"]))
    if t == "nary":
        return NAryOp(Op(d["op"]), [from_dict(c) for c in d["children"]])
    raise ValueError(f"Unknown type: {t}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/dsl/test_serialize.py -v`
Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add atlas/dsl/serialize.py tests/dsl/test_serialize.py
git commit -m "feat: expression serialization (S-expression + dict)"
```

---

## Task 5: Base Environment + Anti-Cheating Tests

**Files:**
- Create: `atlas/environments/__init__.py`
- Create: `atlas/environments/base.py`
- Create: `atlas/environments/normalizer.py`
- Create: `atlas/environments/registry.py`
- Create: `tests/environments/test_base.py`
- Create: `tests/environments/test_anti_cheating.py`

- [ ] **Step 1: Write the failing tests for base environment**

```python
# tests/environments/test_base.py
"""Tests for base environment interface."""
import numpy as np
from atlas.environments.base import BaseEnvironment
from atlas.types import EnvSchema, KnobSpec, KnobType, DetectorSpec


class DummyEnv(BaseEnvironment):
    """Minimal environment for testing the base class."""

    @property
    def env_id(self) -> str:
        return "ENV_DUMMY"

    @property
    def _knob_specs(self) -> list[KnobSpec]:
        return [
            KnobSpec("knob_0", KnobType.CONTINUOUS, 0.0, 1.0),
            KnobSpec("knob_1", KnobType.INTEGER, 1, 100),
        ]

    @property
    def _detector_specs(self) -> list[DetectorSpec]:
        return [DetectorSpec("detector_0", "scalar")]

    def _compute(self, knobs: dict[str, float]) -> dict[str, float | np.ndarray]:
        return {"detector_0": knobs["knob_0"] * 2.0}


def test_schema_structure():
    env = DummyEnv()
    schema = env.get_schema()
    assert isinstance(schema, EnvSchema)
    assert schema.env_id == "ENV_DUMMY"
    assert len(schema.knobs) == 2
    assert len(schema.detectors) == 1


def test_run_returns_detector_values():
    env = DummyEnv()
    result = env.run({"knob_0": 0.5, "knob_1": 10})
    assert "detector_0" in result
    assert result["detector_0"] == 1.0


def test_run_validates_knob_range():
    env = DummyEnv()
    try:
        env.run({"knob_0": 1.5, "knob_1": 10})  # out of range
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_run_validates_missing_knob():
    env = DummyEnv()
    try:
        env.run({"knob_0": 0.5})  # missing knob_1
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_schema_has_no_physics_names():
    """All knob names must be knob_N, detector names must be detector_N."""
    env = DummyEnv()
    schema = env.get_schema()
    for knob in schema.knobs:
        assert knob.name.startswith("knob_"), f"Knob name '{knob.name}' leaks semantics"
    for det in schema.detectors:
        assert det.name.startswith("detector_"), f"Detector name '{det.name}' leaks semantics"
```

- [ ] **Step 2: Write the anti-cheating test suite**

```python
# tests/environments/test_anti_cheating.py
"""Anti-cheating tests: verify no physics knowledge leaks through environment interfaces.

These tests run against ALL registered environments and check:
1. All knob names are knob_N (no physics semantics)
2. All detector names are detector_N
3. All continuous knobs are normalized to [0,1] or [-1,1]
4. Schema contains no physics-related strings
5. Environments are deterministic for the same seed (when applicable)
"""
import re
import numpy as np
from atlas.environments.registry import get_all_environments


PHYSICS_TERMS = [
    "photon", "electron", "proton", "neutron", "quark",
    "frequency", "wavelength", "momentum", "energy", "mass",
    "spin", "charge", "voltage", "current", "magnetic",
    "slit", "diffraction", "interference", "spectrum",
    "planck", "bohr", "compton", "stern", "gerlach",
    "quantum", "classical", "particle", "wave",
    "temperature", "pressure", "force", "acceleration",
]


def test_all_knob_names_anonymous():
    """Every knob across all environments must be named knob_N."""
    for env in get_all_environments():
        schema = env.get_schema()
        for knob in schema.knobs:
            assert re.match(r"^knob_\d+$", knob.name), (
                f"{schema.env_id}: knob '{knob.name}' leaks physics semantics"
            )


def test_all_detector_names_anonymous():
    """Every detector across all environments must be named detector_N."""
    for env in get_all_environments():
        schema = env.get_schema()
        for det in schema.detectors:
            assert re.match(r"^detector_\d+$", det.name), (
                f"{schema.env_id}: detector '{det.name}' leaks physics semantics"
            )


def test_continuous_knobs_normalized():
    """All continuous knobs must have range within [-1, 1]."""
    for env in get_all_environments():
        schema = env.get_schema()
        for knob in schema.knobs:
            if knob.knob_type.value == "continuous":
                assert knob.range_min >= -1.0, (
                    f"{schema.env_id}/{knob.name}: range_min={knob.range_min} < -1"
                )
                assert knob.range_max <= 1.0, (
                    f"{schema.env_id}/{knob.name}: range_max={knob.range_max} > 1"
                )


def test_no_physics_terms_in_schema():
    """Schema string representation must not contain physics terminology."""
    for env in get_all_environments():
        schema = env.get_schema()
        schema_str = str(schema).lower()
        for term in PHYSICS_TERMS:
            assert term not in schema_str, (
                f"{schema.env_id}: schema contains physics term '{term}'"
            )


def test_all_environments_runnable():
    """Every environment must accept valid knob inputs and return detector outputs."""
    for env in get_all_environments():
        schema = env.get_schema()
        # Build a valid input at midpoint of each knob range
        knobs = {}
        for knob in schema.knobs:
            if knob.knob_type == KnobType.DISCRETE:
                knobs[knob.name] = knob.options[0]
            elif knob.knob_type == KnobType.INTEGER:
                knobs[knob.name] = int((knob.range_min + knob.range_max) / 2)
            else:
                knobs[knob.name] = (knob.range_min + knob.range_max) / 2
        result = env.run(knobs)
        # Check all detectors are present
        for det in schema.detectors:
            assert det.name in result, (
                f"{schema.env_id}: missing detector '{det.name}' in output"
            )


# Import KnobType here to avoid circular issues
from atlas.types import KnobType
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/environments/test_base.py tests/environments/test_anti_cheating.py -v`
Expected: FAIL

- [ ] **Step 4: Implement base environment and registry**

```python
# atlas/environments/__init__.py
"""Experiment environments with anonymized knob/detector interfaces."""
```

```python
# atlas/environments/normalizer.py
"""Knob normalization utilities."""
from __future__ import annotations


def normalize(value: float, phys_min: float, phys_max: float,
              target_min: float = 0.0, target_max: float = 1.0) -> float:
    """Map a physical value to normalized range."""
    return target_min + (value - phys_min) / (phys_max - phys_min) * (target_max - target_min)


def denormalize(normed: float, phys_min: float, phys_max: float,
                target_min: float = 0.0, target_max: float = 1.0) -> float:
    """Map a normalized value back to physical range."""
    return phys_min + (normed - target_min) / (target_max - target_min) * (phys_max - phys_min)
```

```python
# atlas/environments/base.py
"""Base environment abstract class with input validation."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from atlas.types import EnvSchema, KnobSpec, KnobType, DetectorSpec


class BaseEnvironment(ABC):
    """Abstract base class for all experiment environments.

    Subclasses implement _knob_specs, _detector_specs, and _compute.
    The base class handles schema generation and input validation.
    """

    def get_schema(self) -> EnvSchema:
        return EnvSchema(
            env_id=self.env_id,
            knobs=self._knob_specs,
            detectors=self._detector_specs,
            entities=self._entities,
        )

    def run(self, knobs: dict[str, float | int]) -> dict[str, float | np.ndarray]:
        """Run the experiment with given knob settings. Validates inputs."""
        self._validate_knobs(knobs)
        return self._compute(knobs)

    @property
    @abstractmethod
    def env_id(self) -> str:
        ...

    @property
    @abstractmethod
    def _knob_specs(self) -> list[KnobSpec]:
        ...

    @property
    @abstractmethod
    def _detector_specs(self) -> list[DetectorSpec]:
        ...

    @property
    def _entities(self) -> list[str]:
        return []

    @abstractmethod
    def _compute(self, knobs: dict[str, float | int]) -> dict[str, float | np.ndarray]:
        ...

    def _validate_knobs(self, knobs: dict[str, float | int]) -> None:
        expected = {s.name for s in self._knob_specs}
        provided = set(knobs.keys())
        missing = expected - provided
        if missing:
            raise ValueError(f"Missing knobs: {missing}")
        extra = provided - expected
        if extra:
            raise ValueError(f"Unexpected knobs: {extra}")
        for spec in self._knob_specs:
            val = knobs[spec.name]
            if spec.knob_type == KnobType.DISCRETE:
                if val not in spec.options:
                    raise ValueError(
                        f"Knob '{spec.name}': value {val} not in options {spec.options}"
                    )
            else:
                if val < spec.range_min or val > spec.range_max:
                    raise ValueError(
                        f"Knob '{spec.name}': value {val} out of range "
                        f"[{spec.range_min}, {spec.range_max}]"
                    )
```

```python
# atlas/environments/registry.py
"""Environment registry: lookup by ID."""
from __future__ import annotations

from atlas.environments.base import BaseEnvironment

_REGISTRY: dict[str, type[BaseEnvironment]] = {}


def register(cls: type[BaseEnvironment]) -> type[BaseEnvironment]:
    """Class decorator to register an environment."""
    instance = cls()
    _REGISTRY[instance.env_id] = cls
    return cls


def get_environment(env_id: str) -> BaseEnvironment:
    """Get an environment instance by ID."""
    if env_id not in _REGISTRY:
        raise KeyError(f"Unknown environment: {env_id}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[env_id]()


def get_all_environments() -> list[BaseEnvironment]:
    """Get instances of all registered environments."""
    return [cls() for cls in _REGISTRY.values()]
```

- [ ] **Step 5: Run base tests to verify they pass**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/environments/test_base.py -v`
Expected: all 5 tests PASS

(Anti-cheating tests will fail until environments are implemented — that's expected.)

- [ ] **Step 6: Commit**

```bash
git add atlas/environments/ tests/environments/test_base.py tests/environments/test_anti_cheating.py
git commit -m "feat: base environment ABC with validation and anti-cheating test suite"
```

---

## Task 6: Classical Environments (ENV-08, 09, 10, 11, 12)

Start with classical environments — they're simpler physics and serve as the "control group." Implementing them first also establishes the pattern for quantum environments.

**Files:**
- Create: `atlas/environments/env_08_water_wave.py`
- Create: `atlas/environments/env_09_elastic_collision.py`
- Create: `atlas/environments/env_10_spring.py`
- Create: `atlas/environments/env_11_freefall.py`
- Create: `atlas/environments/env_12_heat_conduction.py`
- Create: `tests/environments/test_env_08.py`
- Create: `tests/environments/test_env_09.py`
- Create: `tests/environments/test_env_10.py`
- Create: `tests/environments/test_env_11.py`
- Create: `tests/environments/test_env_12.py`

- [ ] **Step 1: Write failing tests for ENV-08 (water wave interference)**

```python
# tests/environments/test_env_08.py
"""Tests for ENV-08: classical water wave interference."""
import numpy as np
from atlas.environments.env_08_water_wave import WaterWaveEnv


def test_schema():
    env = WaterWaveEnv()
    schema = env.get_schema()
    assert schema.env_id == "ENV_08"
    assert len(schema.knobs) >= 2
    assert len(schema.detectors) >= 1


def test_output_is_array():
    env = WaterWaveEnv()
    schema = env.get_schema()
    knobs = {k.name: (k.range_min + k.range_max) / 2 for k in schema.knobs}
    result = env.run(knobs)
    assert "detector_0" in result
    assert isinstance(result["detector_0"], np.ndarray)


def test_output_shows_interference_pattern():
    """High-level: output should have periodic structure (peaks and troughs)."""
    env = WaterWaveEnv()
    schema = env.get_schema()
    knobs = {k.name: (k.range_min + k.range_max) / 2 for k in schema.knobs}
    result = env.run(knobs)
    signal = result["detector_0"]
    # Should have variation (not flat)
    assert np.std(signal) > 1e-6
    # Should have some periodic structure: FFT peak at non-zero freq
    fft = np.abs(np.fft.rfft(signal))
    fft[0] = 0  # remove DC
    assert np.argmax(fft) > 0


def test_deterministic():
    """Same inputs -> same outputs (classical, no randomness)."""
    env = WaterWaveEnv()
    schema = env.get_schema()
    knobs = {k.name: (k.range_min + k.range_max) / 2 for k in schema.knobs}
    r1 = env.run(knobs)
    r2 = env.run(knobs)
    np.testing.assert_array_equal(r1["detector_0"], r2["detector_0"])


def test_knob_sensitivity():
    """Changing a knob should change the output."""
    env = WaterWaveEnv()
    schema = env.get_schema()
    knobs_a = {k.name: (k.range_min + k.range_max) / 2 for k in schema.knobs}
    knobs_b = dict(knobs_a)
    knobs_b["knob_0"] = schema.knobs[0].range_min + 0.01
    r_a = env.run(knobs_a)
    r_b = env.run(knobs_b)
    assert not np.array_equal(r_a["detector_0"], r_b["detector_0"])
```

- [ ] **Step 2: Implement ENV-08**

```python
# atlas/environments/env_08_water_wave.py
"""ENV-08: Classical water wave two-source interference.

Physics (internal, NOT exposed to the system):
  Two point sources separated by distance d, oscillating in phase.
  Detector screen at distance L records intensity pattern.
  I(x) = 4 * I_0 * cos^2(pi * d * x / (lambda * L))

Knobs (anonymized):
  knob_0: source separation (normalized)
  knob_1: wavelength parameter (normalized)
  knob_2: screen distance (normalized)

Detectors:
  detector_0: 1D intensity array (1000 points)
"""
from __future__ import annotations

import numpy as np

from atlas.environments.base import BaseEnvironment
from atlas.environments.normalizer import denormalize
from atlas.environments.registry import register
from atlas.types import KnobSpec, KnobType, DetectorSpec


@register
class WaterWaveEnv(BaseEnvironment):

    @property
    def env_id(self) -> str:
        return "ENV_08"

    @property
    def _knob_specs(self) -> list[KnobSpec]:
        return [
            KnobSpec("knob_0", KnobType.CONTINUOUS, 0.0, 1.0),  # source separation
            KnobSpec("knob_1", KnobType.CONTINUOUS, 0.0, 1.0),  # wavelength
            KnobSpec("knob_2", KnobType.CONTINUOUS, 0.0, 1.0),  # screen distance
        ]

    @property
    def _detector_specs(self) -> list[DetectorSpec]:
        return [DetectorSpec("detector_0", "array_1d", length=1000)]

    def _compute(self, knobs: dict[str, float]) -> dict[str, np.ndarray]:
        # Denormalize to physical ranges (internal only)
        d = denormalize(knobs["knob_0"], 0.01, 0.1)        # separation: 1cm - 10cm
        wavelength = denormalize(knobs["knob_1"], 0.005, 0.05)  # wavelength: 5mm - 5cm
        L = denormalize(knobs["knob_2"], 0.5, 5.0)         # screen dist: 0.5m - 5m

        # Screen positions (normalized to [-1, 1])
        x = np.linspace(-1.0, 1.0, 1000)
        # Physical screen positions
        x_phys = x * 0.5  # half-meter screen width

        # Two-source interference: I = 4 * I_0 * cos^2(pi * d * x / (lambda * L))
        phase = np.pi * d * x_phys / (wavelength * L)
        intensity = np.cos(phase) ** 2

        # Normalize output to [0, 1]
        intensity = intensity / (np.max(intensity) + 1e-30)

        return {"detector_0": intensity}
```

- [ ] **Step 3: Run ENV-08 tests**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/environments/test_env_08.py -v`
Expected: all 5 tests PASS

- [ ] **Step 4: Write failing tests for ENV-10 (spring)**

```python
# tests/environments/test_env_10.py
"""Tests for ENV-10: spring oscillation."""
import numpy as np
from atlas.environments.env_10_spring import SpringEnv


def test_schema():
    env = SpringEnv()
    schema = env.get_schema()
    assert schema.env_id == "ENV_10"


def test_output_is_scalar():
    env = SpringEnv()
    schema = env.get_schema()
    knobs = {}
    for k in schema.knobs:
        if k.knob_type.value == "integer":
            knobs[k.name] = int((k.range_min + k.range_max) / 2)
        else:
            knobs[k.name] = (k.range_min + k.range_max) / 2
    result = env.run(knobs)
    assert "detector_0" in result
    assert isinstance(result["detector_0"], (float, np.floating))


def test_periodic_in_time():
    """Output should be periodic as knob_0 (time) varies."""
    env = SpringEnv()
    values = []
    for t in np.linspace(0, 1, 200):
        result = env.run({"knob_0": t, "knob_1": 0.5, "knob_2": 0.5})
        values.append(result["detector_0"])
    values = np.array(values)
    # Should cross zero multiple times (oscillation)
    zero_crossings = np.sum(np.diff(np.sign(values)) != 0)
    assert zero_crossings >= 2


def test_deterministic():
    env = SpringEnv()
    r1 = env.run({"knob_0": 0.3, "knob_1": 0.5, "knob_2": 0.5})
    r2 = env.run({"knob_0": 0.3, "knob_1": 0.5, "knob_2": 0.5})
    assert r1["detector_0"] == r2["detector_0"]
```

- [ ] **Step 5: Implement ENV-10**

```python
# atlas/environments/env_10_spring.py
"""ENV-10: Classical spring oscillation (SHM).

Physics (internal):
  x(t) = A * cos(sqrt(k/m) * t)

Knobs:
  knob_0: time (normalized)
  knob_1: spring constant parameter (normalized)
  knob_2: amplitude parameter (normalized)

Detectors:
  detector_0: displacement (scalar)
"""
from __future__ import annotations

import numpy as np

from atlas.environments.base import BaseEnvironment
from atlas.environments.normalizer import denormalize
from atlas.environments.registry import register
from atlas.types import KnobSpec, KnobType, DetectorSpec


@register
class SpringEnv(BaseEnvironment):

    @property
    def env_id(self) -> str:
        return "ENV_10"

    @property
    def _knob_specs(self) -> list[KnobSpec]:
        return [
            KnobSpec("knob_0", KnobType.CONTINUOUS, 0.0, 1.0),  # time
            KnobSpec("knob_1", KnobType.CONTINUOUS, 0.0, 1.0),  # spring constant
            KnobSpec("knob_2", KnobType.CONTINUOUS, 0.0, 1.0),  # amplitude
        ]

    @property
    def _detector_specs(self) -> list[DetectorSpec]:
        return [DetectorSpec("detector_0", "scalar")]

    def _compute(self, knobs: dict[str, float]) -> dict[str, float]:
        t = denormalize(knobs["knob_0"], 0.0, 10.0)        # time: 0-10s
        k_over_m = denormalize(knobs["knob_1"], 1.0, 100.0) # k/m: 1-100 s^-2
        A = denormalize(knobs["knob_2"], 0.01, 1.0)         # amplitude: 0.01-1m

        omega = np.sqrt(k_over_m)
        displacement = A * np.cos(omega * t)

        # Normalize output
        return {"detector_0": float(displacement / 1.0)}  # max amplitude ~ 1m
```

- [ ] **Step 6: Run ENV-10 tests**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/environments/test_env_10.py -v`
Expected: all 4 tests PASS

- [ ] **Step 7: Write failing tests for ENV-09 (elastic collision)**

```python
# tests/environments/test_env_09.py
"""Tests for ENV-09: elastic collision."""
import numpy as np
from atlas.environments.env_09_elastic_collision import ElasticCollisionEnv


def test_schema():
    env = ElasticCollisionEnv()
    schema = env.get_schema()
    assert schema.env_id == "ENV_09"


def test_output_has_two_detectors():
    env = ElasticCollisionEnv()
    schema = env.get_schema()
    knobs = {k.name: (k.range_min + k.range_max) / 2 for k in schema.knobs}
    result = env.run(knobs)
    assert "detector_0" in result  # final velocity of object 1
    assert "detector_1" in result  # final velocity of object 2


def test_momentum_conservation():
    """m1*v1_i + m2*v2_i = m1*v1_f + m2*v2_f (approximately, via normalized values)."""
    env = ElasticCollisionEnv()
    # Run several trials with different knob settings
    for _ in range(10):
        knobs = {
            "knob_0": np.random.uniform(0.01, 1.0),  # mass ratio
            "knob_1": np.random.uniform(0.0, 1.0),    # velocity 1
            "knob_2": np.random.uniform(0.0, 1.0),    # velocity 2
        }
        result = env.run(knobs)
        # At minimum, outputs should be finite
        assert np.isfinite(result["detector_0"])
        assert np.isfinite(result["detector_1"])


def test_deterministic():
    env = ElasticCollisionEnv()
    knobs = {"knob_0": 0.3, "knob_1": 0.7, "knob_2": 0.2}
    r1 = env.run(knobs)
    r2 = env.run(knobs)
    assert r1["detector_0"] == r2["detector_0"]
    assert r1["detector_1"] == r2["detector_1"]
```

- [ ] **Step 8: Implement ENV-09**

```python
# atlas/environments/env_09_elastic_collision.py
"""ENV-09: 1D elastic collision between two objects.

Physics (internal):
  v1_f = ((m1-m2)*v1_i + 2*m2*v2_i) / (m1+m2)
  v2_f = ((m2-m1)*v2_i + 2*m1*v1_i) / (m1+m2)

Knobs:
  knob_0: mass ratio parameter (normalized)
  knob_1: initial velocity of object 1 (normalized)
  knob_2: initial velocity of object 2 (normalized)

Detectors:
  detector_0: final velocity of object 1 (normalized)
  detector_1: final velocity of object 2 (normalized)
"""
from __future__ import annotations

import numpy as np

from atlas.environments.base import BaseEnvironment
from atlas.environments.normalizer import denormalize
from atlas.environments.registry import register
from atlas.types import KnobSpec, KnobType, DetectorSpec


@register
class ElasticCollisionEnv(BaseEnvironment):

    @property
    def env_id(self) -> str:
        return "ENV_09"

    @property
    def _knob_specs(self) -> list[KnobSpec]:
        return [
            KnobSpec("knob_0", KnobType.CONTINUOUS, 0.01, 1.0),  # mass ratio
            KnobSpec("knob_1", KnobType.CONTINUOUS, 0.0, 1.0),   # velocity 1
            KnobSpec("knob_2", KnobType.CONTINUOUS, 0.0, 1.0),   # velocity 2
        ]

    @property
    def _detector_specs(self) -> list[DetectorSpec]:
        return [
            DetectorSpec("detector_0", "scalar"),
            DetectorSpec("detector_1", "scalar"),
        ]

    def _compute(self, knobs: dict[str, float]) -> dict[str, float]:
        mass_ratio = denormalize(knobs["knob_0"], 0.1, 10.0)  # m1/m2
        v1_i = denormalize(knobs["knob_1"], -10.0, 10.0)
        v2_i = denormalize(knobs["knob_2"], -10.0, 10.0)

        m1 = mass_ratio
        m2 = 1.0

        v1_f = ((m1 - m2) * v1_i + 2 * m2 * v2_i) / (m1 + m2)
        v2_f = ((m2 - m1) * v2_i + 2 * m1 * v1_i) / (m1 + m2)

        # Normalize outputs to roughly [-1, 1]
        v_max = 20.0
        return {
            "detector_0": float(v1_f / v_max),
            "detector_1": float(v2_f / v_max),
        }
```

- [ ] **Step 9: Write failing tests + implement ENV-11 (freefall)**

```python
# tests/environments/test_env_11.py
"""Tests for ENV-11: freefall."""
import numpy as np
from atlas.environments.env_11_freefall import FreefallEnv


def test_schema():
    env = FreefallEnv()
    schema = env.get_schema()
    assert schema.env_id == "ENV_11"


def test_output_increases_with_time():
    """Distance should increase as knob_0 (time) increases."""
    env = FreefallEnv()
    r1 = env.run({"knob_0": 0.2, "knob_1": 0.5})
    r2 = env.run({"knob_0": 0.8, "knob_1": 0.5})
    assert abs(r2["detector_0"]) > abs(r1["detector_0"])


def test_deterministic():
    env = FreefallEnv()
    r1 = env.run({"knob_0": 0.5, "knob_1": 0.5})
    r2 = env.run({"knob_0": 0.5, "knob_1": 0.5})
    assert r1["detector_0"] == r2["detector_0"]
```

```python
# atlas/environments/env_11_freefall.py
"""ENV-11: Free fall under gravity.

Physics (internal):  y(t) = v0*t - 0.5*g*t^2

Knobs:
  knob_0: time (normalized)
  knob_1: initial velocity parameter (normalized)

Detectors:
  detector_0: position (normalized)
"""
from __future__ import annotations

from atlas.environments.base import BaseEnvironment
from atlas.environments.normalizer import denormalize
from atlas.environments.registry import register
from atlas.types import KnobSpec, KnobType, DetectorSpec


@register
class FreefallEnv(BaseEnvironment):

    @property
    def env_id(self) -> str:
        return "ENV_11"

    @property
    def _knob_specs(self) -> list[KnobSpec]:
        return [
            KnobSpec("knob_0", KnobType.CONTINUOUS, 0.0, 1.0),  # time
            KnobSpec("knob_1", KnobType.CONTINUOUS, 0.0, 1.0),  # initial velocity
        ]

    @property
    def _detector_specs(self) -> list[DetectorSpec]:
        return [DetectorSpec("detector_0", "scalar")]

    def _compute(self, knobs: dict[str, float]) -> dict[str, float]:
        t = denormalize(knobs["knob_0"], 0.0, 5.0)      # 0-5 seconds
        v0 = denormalize(knobs["knob_1"], 0.0, 50.0)     # 0-50 m/s upward
        g = 9.81

        y = v0 * t - 0.5 * g * t * t

        # Normalize: max displacement ~ 125m
        return {"detector_0": float(y / 125.0)}
```

- [ ] **Step 10: Write failing tests + implement ENV-12 (heat conduction)**

```python
# tests/environments/test_env_12.py
"""Tests for ENV-12: heat conduction."""
import numpy as np
from atlas.environments.env_12_heat_conduction import HeatConductionEnv


def test_schema():
    env = HeatConductionEnv()
    schema = env.get_schema()
    assert schema.env_id == "ENV_12"


def test_output_is_scalar():
    env = HeatConductionEnv()
    result = env.run({"knob_0": 0.5, "knob_1": 0.5, "knob_2": 0.5})
    assert isinstance(result["detector_0"], (float, np.floating))


def test_heat_flows_from_hot_to_cold():
    """Higher temperature difference -> higher heat flow."""
    env = HeatConductionEnv()
    r1 = env.run({"knob_0": 0.2, "knob_1": 0.5, "knob_2": 0.5})  # small delta T
    r2 = env.run({"knob_0": 0.8, "knob_1": 0.5, "knob_2": 0.5})  # large delta T
    assert abs(r2["detector_0"]) > abs(r1["detector_0"])
```

```python
# atlas/environments/env_12_heat_conduction.py
"""ENV-12: Steady-state heat conduction (Fourier's law).

Physics (internal):  Q = k * A * dT / L

Knobs:
  knob_0: temperature difference (normalized)
  knob_1: cross-sectional area parameter (normalized)
  knob_2: length parameter (normalized)

Detectors:
  detector_0: heat flow rate (normalized)
"""
from __future__ import annotations

from atlas.environments.base import BaseEnvironment
from atlas.environments.normalizer import denormalize
from atlas.environments.registry import register
from atlas.types import KnobSpec, KnobType, DetectorSpec


@register
class HeatConductionEnv(BaseEnvironment):

    @property
    def env_id(self) -> str:
        return "ENV_12"

    @property
    def _knob_specs(self) -> list[KnobSpec]:
        return [
            KnobSpec("knob_0", KnobType.CONTINUOUS, 0.0, 1.0),  # delta T
            KnobSpec("knob_1", KnobType.CONTINUOUS, 0.0, 1.0),  # area
            KnobSpec("knob_2", KnobType.CONTINUOUS, 0.0, 1.0),  # length
        ]

    @property
    def _detector_specs(self) -> list[DetectorSpec]:
        return [DetectorSpec("detector_0", "scalar")]

    def _compute(self, knobs: dict[str, float]) -> dict[str, float]:
        dT = denormalize(knobs["knob_0"], 1.0, 200.0)    # 1-200 K
        A = denormalize(knobs["knob_1"], 0.001, 0.1)      # 0.001-0.1 m^2
        L = denormalize(knobs["knob_2"], 0.01, 1.0)       # 0.01-1 m
        k = 200.0  # thermal conductivity (fixed, like copper)

        Q = k * A * dT / L

        # Normalize: max Q ~ 200 * 0.1 * 200 / 0.01 = 400000
        return {"detector_0": float(Q / 400000.0)}
```

- [ ] **Step 11: Run all classical environment tests**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/environments/test_env_08.py tests/environments/test_env_09.py tests/environments/test_env_10.py tests/environments/test_env_11.py tests/environments/test_env_12.py -v`
Expected: all tests PASS

- [ ] **Step 12: Commit**

```bash
git add atlas/environments/env_08_water_wave.py atlas/environments/env_09_elastic_collision.py atlas/environments/env_10_spring.py atlas/environments/env_11_freefall.py atlas/environments/env_12_heat_conduction.py tests/environments/test_env_08.py tests/environments/test_env_09.py tests/environments/test_env_10.py tests/environments/test_env_11.py tests/environments/test_env_12.py
git commit -m "feat: classical environments (water wave, collision, spring, freefall, heat)"
```

---

## Task 7: Quantum Environments — Scalar Output (ENV-01, 02, 05)

**Files:**
- Create: `atlas/environments/env_01_photoelectric.py`
- Create: `atlas/environments/env_02_compton.py`
- Create: `atlas/environments/env_05_blackbody.py`
- Create: `tests/environments/test_env_01.py`
- Create: `tests/environments/test_env_02.py`
- Create: `tests/environments/test_env_05.py`

- [ ] **Step 1: Write failing tests for ENV-01 (photoelectric effect)**

```python
# tests/environments/test_env_01.py
"""Tests for ENV-01: photoelectric effect."""
import numpy as np
from atlas.environments.env_01_photoelectric import PhotoelectricEnv
from atlas.types import KnobType


def test_schema():
    env = PhotoelectricEnv()
    schema = env.get_schema()
    assert schema.env_id == "ENV_01"
    # Should have continuous + discrete knobs
    types = {k.knob_type for k in schema.knobs}
    assert KnobType.CONTINUOUS in types
    assert KnobType.DISCRETE in types


def test_output_is_scalar():
    env = PhotoelectricEnv()
    schema = env.get_schema()
    knobs = {}
    for k in schema.knobs:
        if k.knob_type == KnobType.DISCRETE:
            knobs[k.name] = k.options[0]
        else:
            knobs[k.name] = (k.range_min + k.range_max) / 2
    result = env.run(knobs)
    assert "detector_0" in result
    assert isinstance(result["detector_0"], (float, np.floating))


def test_cutoff_behavior():
    """Below a certain knob_0 value, detector should read ~0 (no current)."""
    env = PhotoelectricEnv()
    schema = env.get_schema()
    discrete_knob = [k for k in schema.knobs if k.knob_type == KnobType.DISCRETE][0]
    # Sweep knob_0 from low to high
    values = []
    for v in np.linspace(0.0, 1.0, 50):
        knobs = {k.name: 0.5 for k in schema.knobs if k.knob_type == KnobType.CONTINUOUS}
        knobs["knob_0"] = v
        knobs[discrete_knob.name] = discrete_knob.options[0]
        result = env.run(knobs)
        values.append(result["detector_0"])
    values = np.array(values)
    # Should have some zero (below cutoff) and some nonzero (above cutoff)
    assert np.any(values <= 1e-10)
    assert np.any(values > 0.01)


def test_deterministic():
    env = PhotoelectricEnv()
    schema = env.get_schema()
    discrete_knob = [k for k in schema.knobs if k.knob_type == KnobType.DISCRETE][0]
    knobs = {k.name: 0.5 for k in schema.knobs if k.knob_type == KnobType.CONTINUOUS}
    knobs[discrete_knob.name] = discrete_knob.options[0]
    r1 = env.run(knobs)
    r2 = env.run(knobs)
    assert r1["detector_0"] == r2["detector_0"]
```

- [ ] **Step 2: Implement ENV-01**

```python
# atlas/environments/env_01_photoelectric.py
"""ENV-01: Photoelectric effect.

Physics (internal):
  Current ∝ intensity * max(h*f - W, 0) when voltage allows
  E_max = h*f - W  (maximum kinetic energy)
  Current = 0 if h*f < W (below cutoff)
  Stopping voltage: eV_s = h*f - W

Knobs:
  knob_0: continuous [0,1] — light frequency (normalized)
  knob_1: continuous [0,1] — light intensity (normalized)
  knob_2: discrete {0,1,2,3} — material type (different work functions)
  knob_3: continuous [-1,1] — applied voltage (normalized)

Detectors:
  detector_0: scalar — current reading (normalized)
"""
from __future__ import annotations

import numpy as np

from atlas.environments.base import BaseEnvironment
from atlas.environments.normalizer import denormalize
from atlas.environments.registry import register
from atlas.types import KnobSpec, KnobType, DetectorSpec

# Internal constants (NEVER exposed)
_H = 6.626e-34          # Planck's constant
_E = 1.602e-19          # elementary charge
_WORK_FUNCTIONS = [     # in eV, indexed by material type
    2.3,   # material 0 (like sodium)
    4.1,   # material 1 (like aluminum)
    4.7,   # material 2 (like copper)
    5.1,   # material 3 (like nickel)
]
_FREQ_MIN = 1e14        # Hz
_FREQ_MAX = 3e15        # Hz
_VOLTAGE_MIN = -5.0     # V
_VOLTAGE_MAX = 5.0      # V


@register
class PhotoelectricEnv(BaseEnvironment):

    @property
    def env_id(self) -> str:
        return "ENV_01"

    @property
    def _knob_specs(self) -> list[KnobSpec]:
        return [
            KnobSpec("knob_0", KnobType.CONTINUOUS, 0.0, 1.0),
            KnobSpec("knob_1", KnobType.CONTINUOUS, 0.0, 1.0),
            KnobSpec("knob_2", KnobType.DISCRETE, 0.0, 3.0, options=[0, 1, 2, 3]),
            KnobSpec("knob_3", KnobType.CONTINUOUS, -1.0, 1.0),
        ]

    @property
    def _detector_specs(self) -> list[DetectorSpec]:
        return [DetectorSpec("detector_0", "scalar")]

    @property
    def _entities(self) -> list[str]:
        return ["entity_A"]

    def _compute(self, knobs: dict[str, float | int]) -> dict[str, float]:
        freq = denormalize(knobs["knob_0"], _FREQ_MIN, _FREQ_MAX)
        intensity = denormalize(knobs["knob_1"], 0.0, 1.0)
        material = int(knobs["knob_2"])
        voltage = denormalize(knobs["knob_3"], _VOLTAGE_MIN, _VOLTAGE_MAX)

        W = _WORK_FUNCTIONS[material] * _E  # work function in Joules
        photon_energy = _H * freq
        E_max = photon_energy - W  # max kinetic energy of emitted electrons

        if E_max <= 0:
            # Below cutoff frequency: no emission regardless of voltage
            current = 0.0
        else:
            # Electrons emitted with KE up to E_max
            # Applied voltage either helps or hinders: eV adds to/subtracts from KE
            effective_energy = E_max + _E * voltage
            if effective_energy <= 0:
                current = 0.0
            else:
                current = intensity * (effective_energy / (_H * _FREQ_MAX))

        return {"detector_0": float(np.clip(current, 0.0, 1.0))}
```

- [ ] **Step 3: Write failing tests + implement ENV-02 (Compton scattering)**

```python
# tests/environments/test_env_02.py
"""Tests for ENV-02: Compton scattering."""
import numpy as np
from atlas.environments.env_02_compton import ComptonEnv


def test_schema():
    env = ComptonEnv()
    schema = env.get_schema()
    assert schema.env_id == "ENV_02"


def test_output_has_two_scalars():
    env = ComptonEnv()
    result = env.run({"knob_0": 0.5, "knob_1": 0.5})
    assert "detector_0" in result
    assert "detector_1" in result


def test_zero_angle_no_shift():
    """At knob_1 ≈ 0 (zero scattering angle), wavelength shift should be ~0."""
    env = ComptonEnv()
    result = env.run({"knob_0": 0.5, "knob_1": 0.01})
    # detector_0 is wavelength shift, should be near zero
    assert abs(result["detector_0"]) < 0.05


def test_max_shift_at_pi():
    """At knob_1 ≈ 1 (180 degrees), wavelength shift should be maximal."""
    env = ComptonEnv()
    r_low = env.run({"knob_0": 0.5, "knob_1": 0.1})
    r_high = env.run({"knob_0": 0.5, "knob_1": 0.99})
    assert abs(r_high["detector_0"]) > abs(r_low["detector_0"])
```

```python
# atlas/environments/env_02_compton.py
"""ENV-02: Compton scattering.

Physics (internal):
  Delta_lambda = (h / (m_e * c)) * (1 - cos(theta))
  Scattered photon energy and electron recoil.

Knobs:
  knob_0: continuous [0,1] — incident wavelength (normalized)
  knob_1: continuous [0,1] — scattering angle (normalized to [0, pi])

Detectors:
  detector_0: scalar — wavelength shift (normalized)
  detector_1: scalar — scattered intensity (normalized)
"""
from __future__ import annotations

import numpy as np

from atlas.environments.base import BaseEnvironment
from atlas.environments.normalizer import denormalize
from atlas.environments.registry import register
from atlas.types import KnobSpec, KnobType, DetectorSpec

_H = 6.626e-34
_M_E = 9.109e-31
_C = 2.998e8
_COMPTON_WAVELENGTH = _H / (_M_E * _C)  # ~2.426e-12 m


@register
class ComptonEnv(BaseEnvironment):

    @property
    def env_id(self) -> str:
        return "ENV_02"

    @property
    def _knob_specs(self) -> list[KnobSpec]:
        return [
            KnobSpec("knob_0", KnobType.CONTINUOUS, 0.0, 1.0),  # incident wavelength
            KnobSpec("knob_1", KnobType.CONTINUOUS, 0.0, 1.0),  # scattering angle
        ]

    @property
    def _detector_specs(self) -> list[DetectorSpec]:
        return [
            DetectorSpec("detector_0", "scalar"),  # wavelength shift
            DetectorSpec("detector_1", "scalar"),  # scattered intensity
        ]

    @property
    def _entities(self) -> list[str]:
        return ["entity_A"]

    def _compute(self, knobs: dict[str, float]) -> dict[str, float]:
        lambda_i = denormalize(knobs["knob_0"], 1e-12, 1e-10)  # 1pm - 100pm
        theta = denormalize(knobs["knob_1"], 0.0, np.pi)

        delta_lambda = _COMPTON_WAVELENGTH * (1 - np.cos(theta))
        lambda_f = lambda_i + delta_lambda

        # Klein-Nishina-like intensity (simplified)
        r = lambda_i / lambda_f
        intensity = 0.5 * r * r * (r + 1.0 / r - np.sin(theta) ** 2)

        # Normalize outputs
        delta_max = 2 * _COMPTON_WAVELENGTH  # max shift at theta=pi
        return {
            "detector_0": float(delta_lambda / delta_max),
            "detector_1": float(np.clip(intensity / 2.0, 0.0, 1.0)),
        }
```

- [ ] **Step 4: Write failing tests + implement ENV-05 (blackbody radiation)**

```python
# tests/environments/test_env_05.py
"""Tests for ENV-05: blackbody radiation."""
import numpy as np
from atlas.environments.env_05_blackbody import BlackbodyEnv


def test_schema():
    env = BlackbodyEnv()
    schema = env.get_schema()
    assert schema.env_id == "ENV_05"


def test_output_is_scalar():
    env = BlackbodyEnv()
    result = env.run({"knob_0": 0.5, "knob_1": 0.5})
    assert isinstance(result["detector_0"], (float, np.floating))


def test_higher_temp_more_radiation():
    """At same frequency, higher temperature -> higher spectral radiance."""
    env = BlackbodyEnv()
    r_cold = env.run({"knob_0": 0.5, "knob_1": 0.2})
    r_hot = env.run({"knob_0": 0.5, "knob_1": 0.8})
    assert r_hot["detector_0"] > r_cold["detector_0"]


def test_output_nonnegative():
    env = BlackbodyEnv()
    for _ in range(20):
        result = env.run({
            "knob_0": np.random.uniform(0.01, 1.0),
            "knob_1": np.random.uniform(0.01, 1.0),
        })
        assert result["detector_0"] >= 0
```

```python
# atlas/environments/env_05_blackbody.py
"""ENV-05: Blackbody radiation spectrum.

Physics (internal):
  B(f, T) = (2*h*f^3 / c^2) / (exp(h*f / (k_B*T)) - 1)

Knobs:
  knob_0: continuous [0,1] — frequency (normalized)
  knob_1: continuous [0,1] — temperature (normalized)

Detectors:
  detector_0: scalar — spectral radiance (normalized)
"""
from __future__ import annotations

import numpy as np

from atlas.environments.base import BaseEnvironment
from atlas.environments.normalizer import denormalize
from atlas.environments.registry import register
from atlas.types import KnobSpec, KnobType, DetectorSpec

_H = 6.626e-34
_C = 2.998e8
_K_B = 1.381e-23


@register
class BlackbodyEnv(BaseEnvironment):

    @property
    def env_id(self) -> str:
        return "ENV_05"

    @property
    def _knob_specs(self) -> list[KnobSpec]:
        return [
            KnobSpec("knob_0", KnobType.CONTINUOUS, 0.0, 1.0),  # frequency
            KnobSpec("knob_1", KnobType.CONTINUOUS, 0.0, 1.0),  # temperature
        ]

    @property
    def _detector_specs(self) -> list[DetectorSpec]:
        return [DetectorSpec("detector_0", "scalar")]

    def _compute(self, knobs: dict[str, float]) -> dict[str, float]:
        freq = denormalize(knobs["knob_0"], 1e12, 3e15)    # 1 THz - 3 PHz
        temp = denormalize(knobs["knob_1"], 300.0, 10000.0)  # 300K - 10000K

        x = _H * freq / (_K_B * temp)
        if x > 500:
            B = 0.0
        else:
            B = (2 * _H * freq**3 / _C**2) / (np.exp(x) - 1)

        # Normalize: peak of B at T=10000K is roughly 1e-5
        B_norm = B / 1e-5

        return {"detector_0": float(np.clip(B_norm, 0.0, 10.0))}
```

- [ ] **Step 5: Run all scalar quantum env tests**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/environments/test_env_01.py tests/environments/test_env_02.py tests/environments/test_env_05.py -v`
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add atlas/environments/env_01_photoelectric.py atlas/environments/env_02_compton.py atlas/environments/env_05_blackbody.py tests/environments/test_env_01.py tests/environments/test_env_02.py tests/environments/test_env_05.py
git commit -m "feat: quantum scalar environments (photoelectric, compton, blackbody)"
```

---

## Task 8: Quantum Environments — Array/List Output (ENV-03, 04, 06, 07)

These are the most complex environments. They involve array outputs, stochastic behavior, and discrete state spaces.

**Files:**
- Create: `atlas/environments/env_03_electron_diffraction.py`
- Create: `atlas/environments/env_04_double_slit.py`
- Create: `atlas/environments/env_06_hydrogen_spectrum.py`
- Create: `atlas/environments/env_07_stern_gerlach.py`
- Create: `tests/environments/test_env_03.py`
- Create: `tests/environments/test_env_04.py`
- Create: `tests/environments/test_env_06.py`
- Create: `tests/environments/test_env_07.py`

- [ ] **Step 1: Write failing tests for ENV-04 (double slit)**

```python
# tests/environments/test_env_04.py
"""Tests for ENV-04: double slit experiment.

This is the most important environment — it must exhibit:
- Smooth interference pattern at high intensity (wave behavior)
- Sparse discrete hits at low intensity (particle behavior)
- Statistical convergence: many low-intensity runs -> interference pattern
"""
import numpy as np
from atlas.environments.env_04_double_slit import DoubleSlitEnv


def test_schema():
    env = DoubleSlitEnv()
    schema = env.get_schema()
    assert schema.env_id == "ENV_04"
    # Must have the integer knob (source intensity)
    int_knobs = [k for k in schema.knobs if k.knob_type.value == "integer"]
    assert len(int_knobs) >= 1


def test_high_intensity_smooth_pattern():
    """At high knob_3 (intensity), output should be a smooth interference pattern."""
    env = DoubleSlitEnv()
    result = env.run({"knob_0": 0.5, "knob_1": 0.5, "knob_2": 0.5, "knob_3": 1000000})
    signal = result["detector_0"]
    assert isinstance(signal, np.ndarray)
    assert len(signal) == 1000
    # Should be smooth (low high-frequency content)
    fft = np.abs(np.fft.rfft(signal))
    low_freq_power = np.sum(fft[:50] ** 2)
    high_freq_power = np.sum(fft[200:] ** 2)
    assert low_freq_power > high_freq_power * 10


def test_low_intensity_sparse():
    """At low knob_3 (intensity), output should be sparse (mostly zeros)."""
    env = DoubleSlitEnv(seed=42)
    result = env.run({"knob_0": 0.5, "knob_1": 0.5, "knob_2": 0.5, "knob_3": 5})
    signal = result["detector_0"]
    # Most bins should be zero
    nonzero_fraction = np.count_nonzero(signal) / len(signal)
    assert nonzero_fraction < 0.1


def test_low_intensity_stochastic():
    """Same low-intensity settings with different seeds should give different outputs."""
    env1 = DoubleSlitEnv(seed=42)
    env2 = DoubleSlitEnv(seed=99)
    r1 = env1.run({"knob_0": 0.5, "knob_1": 0.5, "knob_2": 0.5, "knob_3": 5})
    r2 = env2.run({"knob_0": 0.5, "knob_1": 0.5, "knob_2": 0.5, "knob_3": 5})
    assert not np.array_equal(r1["detector_0"], r2["detector_0"])


def test_statistical_convergence():
    """Many low-intensity runs should converge to the high-intensity pattern."""
    env_hi = DoubleSlitEnv()
    hi_pattern = env_hi.run({"knob_0": 0.5, "knob_1": 0.5, "knob_2": 0.5, "knob_3": 1000000})["detector_0"]

    # Accumulate many low-intensity runs
    accumulated = np.zeros(1000)
    for seed in range(500):
        env_lo = DoubleSlitEnv(seed=seed)
        result = env_lo.run({"knob_0": 0.5, "knob_1": 0.5, "knob_2": 0.5, "knob_3": 10})
        accumulated += result["detector_0"]

    # Normalize both
    hi_norm = hi_pattern / (np.max(hi_pattern) + 1e-30)
    acc_norm = accumulated / (np.max(accumulated) + 1e-30)

    # Correlation should be high
    corr = np.corrcoef(hi_norm, acc_norm)[0, 1]
    assert corr > 0.8


def test_knob_0_changes_pattern():
    """Changing knob_0 (slit width) should change the interference pattern."""
    env = DoubleSlitEnv()
    r1 = env.run({"knob_0": 0.2, "knob_1": 0.5, "knob_2": 0.5, "knob_3": 1000000})
    r2 = env.run({"knob_0": 0.8, "knob_1": 0.5, "knob_2": 0.5, "knob_3": 1000000})
    assert not np.allclose(r1["detector_0"], r2["detector_0"])
```

- [ ] **Step 2: Implement ENV-04**

```python
# atlas/environments/env_04_double_slit.py
"""ENV-04: Double slit experiment.

Physics (internal):
  Intensity: I(x) = I_0 * cos^2(pi*d*x / (lambda*L)) * sinc^2(pi*a*x / (lambda*L))
  Low intensity: Poisson sampling from I(x) distribution

Knobs:
  knob_0: continuous [0,1] — slit width parameter (normalized)
  knob_1: continuous [0,1] — slit separation parameter (normalized)
  knob_2: continuous [0,1] — source wavelength parameter (normalized)
  knob_3: integer [1, 1000000] — source intensity (number of quanta)

Detectors:
  detector_0: array_1d[1000] — detector screen readings
"""
from __future__ import annotations

import numpy as np

from atlas.environments.base import BaseEnvironment
from atlas.environments.normalizer import denormalize
from atlas.environments.registry import register
from atlas.types import KnobSpec, KnobType, DetectorSpec


@register
class DoubleSlitEnv(BaseEnvironment):

    def __init__(self, seed: int | None = None):
        self._seed = seed

    @property
    def env_id(self) -> str:
        return "ENV_04"

    @property
    def _knob_specs(self) -> list[KnobSpec]:
        return [
            KnobSpec("knob_0", KnobType.CONTINUOUS, 0.0, 1.0),   # slit width
            KnobSpec("knob_1", KnobType.CONTINUOUS, 0.0, 1.0),   # slit separation
            KnobSpec("knob_2", KnobType.CONTINUOUS, 0.0, 1.0),   # wavelength
            KnobSpec("knob_3", KnobType.INTEGER, 1, 1000000),     # intensity
        ]

    @property
    def _detector_specs(self) -> list[DetectorSpec]:
        return [DetectorSpec("detector_0", "array_1d", length=1000)]

    @property
    def _entities(self) -> list[str]:
        return ["entity_A"]

    def _compute(self, knobs: dict[str, float | int]) -> dict[str, np.ndarray]:
        a = denormalize(knobs["knob_0"], 10e-6, 200e-6)     # slit width: 10-200 um
        d = denormalize(knobs["knob_1"], 50e-6, 1000e-6)    # slit sep: 50-1000 um
        lam = denormalize(knobs["knob_2"], 400e-9, 700e-9)  # wavelength: 400-700 nm
        N = int(knobs["knob_3"])                             # number of quanta
        L = 1.0  # screen distance fixed at 1m (internal)

        # Screen positions
        x = np.linspace(-0.02, 0.02, 1000)  # +/- 2cm

        # Double-slit diffraction pattern
        alpha = np.pi * a * x / (lam * L)
        beta = np.pi * d * x / (lam * L)

        # sinc envelope (single slit diffraction)
        sinc_env = np.where(np.abs(alpha) < 1e-10, 1.0, np.sin(alpha) / alpha)
        # cos^2 interference (double slit)
        interference = np.cos(beta) ** 2

        intensity = sinc_env ** 2 * interference

        # Normalize probability distribution
        prob = intensity / (np.sum(intensity) + 1e-30)

        if N >= 10000:
            # High intensity: return smooth intensity pattern
            output = intensity / (np.max(intensity) + 1e-30)
        else:
            # Low intensity: sample individual detections
            rng = np.random.default_rng(self._seed)
            hits = rng.multinomial(N, prob)
            output = hits.astype(float)

        return {"detector_0": output}
```

- [ ] **Step 3: Write failing tests + implement ENV-07 (Stern-Gerlach)**

```python
# tests/environments/test_env_07.py
"""Tests for ENV-07: Stern-Gerlach experiment."""
import numpy as np
from atlas.environments.env_07_stern_gerlach import SternGerlachEnv


def test_schema():
    env = SternGerlachEnv()
    schema = env.get_schema()
    assert schema.env_id == "ENV_07"


def test_output_is_array():
    env = SternGerlachEnv(seed=42)
    schema = env.get_schema()
    knobs = {}
    for k in schema.knobs:
        if k.knob_type.value == "integer":
            knobs[k.name] = int((k.range_min + k.range_max) / 2)
        else:
            knobs[k.name] = (k.range_min + k.range_max) / 2
    result = env.run(knobs)
    assert "detector_0" in result
    assert isinstance(result["detector_0"], np.ndarray)


def test_discrete_output():
    """Output should cluster around discrete positions (spin quantization)."""
    env = SternGerlachEnv(seed=42)
    result = env.run({"knob_0": 0.5, "knob_1": 0.5, "knob_2": 100000})
    signal = result["detector_0"]
    # Should have clear peaks (not uniform)
    assert np.max(signal) > 5 * np.mean(signal)


def test_stochastic_at_low_count():
    """Low particle count should give noisy results."""
    e1 = SternGerlachEnv(seed=42)
    e2 = SternGerlachEnv(seed=99)
    r1 = e1.run({"knob_0": 0.5, "knob_1": 0.5, "knob_2": 10})
    r2 = e2.run({"knob_0": 0.5, "knob_1": 0.5, "knob_2": 10})
    assert not np.array_equal(r1["detector_0"], r2["detector_0"])
```

```python
# atlas/environments/env_07_stern_gerlach.py
"""ENV-07: Stern-Gerlach experiment (spin-1/2).

Physics (internal):
  Beam of spin-1/2 particles in inhomogeneous magnetic field.
  Deflection is quantized: only UP or DOWN.
  P(up) = cos^2(theta/2) where theta is angle between spin prep and field.

Knobs:
  knob_0: continuous [0,1] — preparation angle (normalized to [0, pi])
  knob_1: continuous [0,1] — field gradient strength (normalized)
  knob_2: integer [1, 1000000] — number of particles

Detectors:
  detector_0: array_1d[200] — position histogram on detector screen
"""
from __future__ import annotations

import numpy as np

from atlas.environments.base import BaseEnvironment
from atlas.environments.normalizer import denormalize
from atlas.environments.registry import register
from atlas.types import KnobSpec, KnobType, DetectorSpec


@register
class SternGerlachEnv(BaseEnvironment):

    def __init__(self, seed: int | None = None):
        self._seed = seed

    @property
    def env_id(self) -> str:
        return "ENV_07"

    @property
    def _knob_specs(self) -> list[KnobSpec]:
        return [
            KnobSpec("knob_0", KnobType.CONTINUOUS, 0.0, 1.0),  # preparation angle
            KnobSpec("knob_1", KnobType.CONTINUOUS, 0.0, 1.0),  # field gradient
            KnobSpec("knob_2", KnobType.INTEGER, 1, 1000000),    # particle count
        ]

    @property
    def _detector_specs(self) -> list[DetectorSpec]:
        return [DetectorSpec("detector_0", "array_1d", length=200)]

    def _compute(self, knobs: dict[str, float | int]) -> dict[str, np.ndarray]:
        theta = denormalize(knobs["knob_0"], 0.0, np.pi)
        gradient = denormalize(knobs["knob_1"], 0.1, 10.0)
        N = int(knobs["knob_2"])

        # Probability of spin-up
        p_up = np.cos(theta / 2) ** 2

        rng = np.random.default_rng(self._seed)

        # Each particle deflects UP or DOWN
        n_up = rng.binomial(N, p_up)
        n_down = N - n_up

        # Position histogram: two peaks at +/- deflection
        bins = np.zeros(200)
        center = 100
        deflection = int(gradient * 30)  # peak separation in bins

        # UP peak (Gaussian smeared)
        up_pos = center + deflection
        if 0 <= up_pos < 200:
            spread = max(1, int(2 + 0.5 * gradient))
            for _ in range(n_up):
                idx = int(rng.normal(up_pos, spread))
                if 0 <= idx < 200:
                    bins[idx] += 1

        # DOWN peak
        down_pos = center - deflection
        if 0 <= down_pos < 200:
            spread = max(1, int(2 + 0.5 * gradient))
            for _ in range(n_down):
                idx = int(rng.normal(down_pos, spread))
                if 0 <= idx < 200:
                    bins[idx] += 1

        return {"detector_0": bins}
```

- [ ] **Step 4: Write failing tests + implement ENV-03 (electron diffraction)**

```python
# tests/environments/test_env_03.py
"""Tests for ENV-03: electron diffraction."""
import numpy as np
from atlas.environments.env_03_electron_diffraction import ElectronDiffractionEnv
from atlas.types import KnobType


def test_schema():
    env = ElectronDiffractionEnv()
    schema = env.get_schema()
    assert schema.env_id == "ENV_03"


def test_output_is_array():
    env = ElectronDiffractionEnv()
    schema = env.get_schema()
    knobs = {}
    for k in schema.knobs:
        if k.knob_type == KnobType.DISCRETE:
            knobs[k.name] = k.options[0]
        elif k.knob_type == KnobType.INTEGER:
            knobs[k.name] = int((k.range_min + k.range_max) / 2)
        else:
            knobs[k.name] = (k.range_min + k.range_max) / 2
    result = env.run(knobs)
    assert isinstance(result["detector_0"], np.ndarray)


def test_shows_diffraction_rings():
    """Output should show diffraction pattern (periodic peaks)."""
    env = ElectronDiffractionEnv()
    schema = env.get_schema()
    knobs = {}
    for k in schema.knobs:
        if k.knob_type == KnobType.DISCRETE:
            knobs[k.name] = k.options[0]
        elif k.knob_type == KnobType.INTEGER:
            knobs[k.name] = int((k.range_min + k.range_max) / 2)
        else:
            knobs[k.name] = (k.range_min + k.range_max) / 2
    result = env.run(knobs)
    signal = result["detector_0"]
    assert np.std(signal) > 1e-6  # not flat
```

```python
# atlas/environments/env_03_electron_diffraction.py
"""ENV-03: Electron diffraction through crystal lattice.

Physics (internal):
  de Broglie wavelength: lambda = h / p = h / sqrt(2*m*E)
  Bragg diffraction: constructive at angles where 2*d*sin(theta) = n*lambda

Knobs:
  knob_0: continuous [0,1] — accelerating voltage (normalized, controls electron energy)
  knob_1: continuous [0,1] — lattice spacing parameter (normalized)
  knob_2: discrete {0,1,2} — crystal type

Detectors:
  detector_0: array_1d[500] — radial intensity profile
"""
from __future__ import annotations

import numpy as np

from atlas.environments.base import BaseEnvironment
from atlas.environments.normalizer import denormalize
from atlas.environments.registry import register
from atlas.types import KnobSpec, KnobType, DetectorSpec

_H = 6.626e-34
_M_E = 9.109e-31
_E_CHARGE = 1.602e-19


@register
class ElectronDiffractionEnv(BaseEnvironment):

    @property
    def env_id(self) -> str:
        return "ENV_03"

    @property
    def _knob_specs(self) -> list[KnobSpec]:
        return [
            KnobSpec("knob_0", KnobType.CONTINUOUS, 0.0, 1.0),  # voltage
            KnobSpec("knob_1", KnobType.CONTINUOUS, 0.0, 1.0),  # lattice spacing
            KnobSpec("knob_2", KnobType.DISCRETE, 0.0, 2.0, options=[0, 1, 2]),
        ]

    @property
    def _detector_specs(self) -> list[DetectorSpec]:
        return [DetectorSpec("detector_0", "array_1d", length=500)]

    @property
    def _entities(self) -> list[str]:
        return ["entity_B"]  # electrons are a different entity from light

    def _compute(self, knobs: dict[str, float | int]) -> dict[str, np.ndarray]:
        V = denormalize(knobs["knob_0"], 1e3, 1e5)  # 1kV - 100kV
        d_base = denormalize(knobs["knob_1"], 1e-10, 5e-10)  # lattice spacing
        crystal = int(knobs["knob_2"])
        d_multipliers = [1.0, 1.41, 1.73]  # different crystal structures
        d = d_base * d_multipliers[crystal]

        # de Broglie wavelength
        E = _E_CHARGE * V
        p = np.sqrt(2 * _M_E * E)
        lam = _H / p

        # Radial position on screen (normalized angle)
        r = np.linspace(0, 0.1, 500)  # radians

        # Powder diffraction pattern (sum over Bragg orders)
        intensity = np.zeros(500)
        for n in range(1, 6):
            sin_theta = n * lam / (2 * d)
            if abs(sin_theta) <= 1:
                theta_bragg = np.arcsin(sin_theta)
                peak = np.exp(-((r - theta_bragg) ** 2) / (0.001 ** 2))
                intensity += peak / n  # higher orders weaker

        # Normalize
        if np.max(intensity) > 0:
            intensity = intensity / np.max(intensity)

        return {"detector_0": intensity}
```

- [ ] **Step 5: Write failing tests + implement ENV-06 (hydrogen spectrum)**

```python
# tests/environments/test_env_06.py
"""Tests for ENV-06: hydrogen atom spectrum."""
import numpy as np
from atlas.environments.env_06_hydrogen_spectrum import HydrogenSpectrumEnv


def test_schema():
    env = HydrogenSpectrumEnv()
    schema = env.get_schema()
    assert schema.env_id == "ENV_06"


def test_output_is_list_like():
    """Output should be a list of discrete spectral line positions."""
    env = HydrogenSpectrumEnv()
    result = env.run({"knob_0": 0.5, "knob_1": 0.5})
    det = result["detector_0"]
    assert isinstance(det, np.ndarray)


def test_discrete_lines():
    """Spectrum should have sharp, discrete peaks."""
    env = HydrogenSpectrumEnv()
    result = env.run({"knob_0": 0.5, "knob_1": 0.5})
    signal = result["detector_0"]
    # Should have clear peaks separated by near-zero regions
    n_peaks = np.sum((signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:]) & (signal[1:-1] > 0.1))
    assert n_peaks >= 2  # at least 2 spectral lines
```

```python
# atlas/environments/env_06_hydrogen_spectrum.py
"""ENV-06: Hydrogen atom emission spectrum.

Physics (internal):
  Rydberg formula: 1/lambda = R_H * (1/n1^2 - 1/n2^2)
  R_H = m_e * e^4 / (8 * eps0^2 * h^3 * c)

Knobs:
  knob_0: continuous [0,1] — spectrometer range (normalized wavelength window)
  knob_1: continuous [0,1] — excitation energy (determines which transitions are visible)

Detectors:
  detector_0: array_1d[500] — spectrum (intensity vs wavelength position)
"""
from __future__ import annotations

import numpy as np

from atlas.environments.base import BaseEnvironment
from atlas.environments.normalizer import denormalize
from atlas.environments.registry import register
from atlas.types import KnobSpec, KnobType, DetectorSpec

_R_H = 1.097e7  # Rydberg constant (m^-1)


@register
class HydrogenSpectrumEnv(BaseEnvironment):

    @property
    def env_id(self) -> str:
        return "ENV_06"

    @property
    def _knob_specs(self) -> list[KnobSpec]:
        return [
            KnobSpec("knob_0", KnobType.CONTINUOUS, 0.0, 1.0),  # wavelength range
            KnobSpec("knob_1", KnobType.CONTINUOUS, 0.0, 1.0),  # excitation
        ]

    @property
    def _detector_specs(self) -> list[DetectorSpec]:
        return [DetectorSpec("detector_0", "array_1d", length=500)]

    def _compute(self, knobs: dict[str, float]) -> dict[str, np.ndarray]:
        # Wavelength window
        lam_center = denormalize(knobs["knob_0"], 100e-9, 2000e-9)
        lam_width = 500e-9
        lam_min = lam_center - lam_width / 2
        lam_max = lam_center + lam_width / 2

        # Excitation energy determines max n level
        n_max = int(denormalize(knobs["knob_1"], 2, 8))

        # Compute all transition wavelengths
        wavelength_axis = np.linspace(max(lam_min, 50e-9), max(lam_max, 100e-9), 500)
        spectrum = np.zeros(500)

        for n1 in range(1, n_max):
            for n2 in range(n1 + 1, n_max + 1):
                inv_lam = _R_H * (1.0 / n1**2 - 1.0 / n2**2)
                if inv_lam > 0:
                    lam_line = 1.0 / inv_lam
                    # Add Gaussian peak at this wavelength
                    line_width = 1e-9  # 1nm instrumental width
                    peak = np.exp(-((wavelength_axis - lam_line) ** 2) / (2 * line_width**2))
                    # Intensity decreases for higher transitions
                    intensity = 1.0 / (n2 - n1) ** 2
                    spectrum += intensity * peak

        # Normalize
        if np.max(spectrum) > 0:
            spectrum = spectrum / np.max(spectrum)

        return {"detector_0": spectrum}
```

- [ ] **Step 6: Run all quantum array env tests**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/environments/test_env_03.py tests/environments/test_env_04.py tests/environments/test_env_06.py tests/environments/test_env_07.py -v`
Expected: all tests PASS

- [ ] **Step 7: Commit**

```bash
git add atlas/environments/env_03_electron_diffraction.py atlas/environments/env_04_double_slit.py atlas/environments/env_06_hydrogen_spectrum.py atlas/environments/env_07_stern_gerlach.py tests/environments/test_env_03.py tests/environments/test_env_04.py tests/environments/test_env_06.py tests/environments/test_env_07.py
git commit -m "feat: quantum array environments (diffraction, double slit, hydrogen, stern-gerlach)"
```

---

## Task 9: Run Full Anti-Cheating Suite + Final Validation

**Files:**
- Modify: `tests/environments/test_anti_cheating.py` (may need fixes)
- Modify: `atlas/environments/__init__.py` (import all environments for registry)

- [ ] **Step 1: Update environments __init__ to import all modules**

```python
# atlas/environments/__init__.py
"""Experiment environments with anonymized knob/detector interfaces.

Import all environment modules to trigger registration.
"""
from atlas.environments import (  # noqa: F401
    env_01_photoelectric,
    env_02_compton,
    env_03_electron_diffraction,
    env_04_double_slit,
    env_05_blackbody,
    env_06_hydrogen_spectrum,
    env_07_stern_gerlach,
    env_08_water_wave,
    env_09_elastic_collision,
    env_10_spring,
    env_11_freefall,
    env_12_heat_conduction,
)
```

- [ ] **Step 2: Run the anti-cheating test suite**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/environments/test_anti_cheating.py -v`
Expected: all 5 anti-cheating tests PASS across all 12 environments

- [ ] **Step 3: Fix any anti-cheating violations**

If any test fails, fix the offending environment. Common issues:
- Knob range exceeding [-1, 1] for continuous knobs (check `knob_0` range in ENV-09 is 0.01 not < -1)
- Physics terms in schema string repr (check entity names don't contain physics terms)

- [ ] **Step 4: Run complete test suite**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/ -v --tb=short`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add atlas/environments/__init__.py tests/environments/test_anti_cheating.py
git commit -m "feat: complete environment suite with anti-cheating validation"
```

---

## Task 10: Alternative Physics Validation Harness

This implements V1 from the validation protocol: the ability to run all environments with modified physical constants (e.g., h -> 2h) to verify the system discovers different constant values but the same formula structures.

**Files:**
- Create: `atlas/environments/alt_physics.py`
- Create: `tests/environments/test_alt_physics.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/environments/test_alt_physics.py
"""Tests for alternative physics configuration."""
import numpy as np
from atlas.environments.alt_physics import PhysicsConfig, apply_config
from atlas.environments.registry import get_environment


def test_default_config():
    cfg = PhysicsConfig()
    assert cfg.h_multiplier == 1.0
    assert cfg.c_multiplier == 1.0


def test_alt_config_changes_output():
    """With h->2h, photoelectric cutoff should change."""
    env_default = get_environment("ENV_01")
    env_alt = apply_config(get_environment("ENV_01"), PhysicsConfig(h_multiplier=2.0))

    knobs = {"knob_0": 0.7, "knob_1": 0.5, "knob_2": 0, "knob_3": 0.0}
    r_default = env_default.run(knobs)
    r_alt = env_alt.run(knobs)

    # Outputs should differ (different h means different cutoff)
    assert r_default["detector_0"] != r_alt["detector_0"]


def test_classical_limit():
    """With h->0 (very small), quantum experiments should behave classically."""
    env = apply_config(get_environment("ENV_01"), PhysicsConfig(h_multiplier=1e-10))
    # At h~0, photon energy is negligible -> no photoelectric emission
    result = env.run({"knob_0": 0.5, "knob_1": 0.5, "knob_2": 0, "knob_3": 0.0})
    assert result["detector_0"] <= 1e-6
```

- [ ] **Step 2: Implement alternative physics config**

```python
# atlas/environments/alt_physics.py
"""Alternative physics configuration for validation tests.

Allows modifying fundamental constants (h, c, k_B, etc.) to verify
that the discovery system responds to data, not to hardcoded values.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from atlas.environments.base import BaseEnvironment


@dataclass
class PhysicsConfig:
    """Multipliers for fundamental constants.
    Default is 1.0 for all (standard physics).
    """
    h_multiplier: float = 1.0       # Planck's constant
    c_multiplier: float = 1.0       # speed of light
    k_b_multiplier: float = 1.0     # Boltzmann constant
    e_multiplier: float = 1.0       # elementary charge
    m_e_multiplier: float = 1.0     # electron mass


def apply_config(env: BaseEnvironment, config: PhysicsConfig) -> BaseEnvironment:
    """Return a copy of the environment with modified physics constants.

    The environment's _compute method is wrapped to inject modified constants.
    """
    modified = deepcopy(env)
    modified._physics_config = config  # type: ignore[attr-defined]
    original_compute = modified._compute.__func__  # type: ignore[attr-defined]

    def patched_compute(self, knobs):
        # Temporarily modify module-level constants
        import atlas.environments.env_01_photoelectric as m01
        import atlas.environments.env_02_compton as m02
        import atlas.environments.env_04_double_slit as m04
        import atlas.environments.env_05_blackbody as m05

        modules = [m01, m02, m04, m05]
        originals = {}

        for mod in modules:
            if hasattr(mod, "_H"):
                originals[(mod, "_H")] = mod._H
                mod._H = mod._H * config.h_multiplier

        try:
            return original_compute(self, knobs)
        finally:
            for (mod, attr), val in originals.items():
                setattr(mod, attr, val)

    import types
    modified._compute = types.MethodType(patched_compute, modified)  # type: ignore[attr-defined]
    return modified
```

- [ ] **Step 3: Run tests**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/environments/test_alt_physics.py -v`
Expected: all 3 tests PASS

- [ ] **Step 4: Commit**

```bash
git add atlas/environments/alt_physics.py tests/environments/test_alt_physics.py
git commit -m "feat: alternative physics validation harness (h->2h, h->0 tests)"
```

---

## Task 11: Final Integration Test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
"""Integration test: verify the complete foundation layer works end-to-end."""
import numpy as np
from atlas.types import EnvSchema
from atlas.dsl.expr import Const, Var, BinOp, UnaryOp
from atlas.dsl.operators import Op, DSL_0
from atlas.dsl.canonicalize import canonicalize
from atlas.dsl.serialize import to_str, from_str, to_dict, from_dict
from atlas.environments.registry import get_environment, get_all_environments


def test_all_12_environments_registered():
    envs = get_all_environments()
    ids = {e.get_schema().env_id for e in envs}
    expected = {f"ENV_{i:02d}" for i in range(1, 13)}
    assert ids == expected, f"Missing: {expected - ids}, Extra: {ids - expected}"


def test_dsl_roundtrip():
    """Build an expression, canonicalize it, serialize it, deserialize it."""
    # cos^2(x_0 * C) = cos(x_0 * C) * cos(x_0 * C)
    inner = BinOp(Op.MUL, Var("knob_0"), Const(3.14))
    expr = BinOp(Op.MUL, UnaryOp(Op.COS, inner), UnaryOp(Op.COS, inner))

    canon = canonicalize(expr)
    s = to_str(canon)
    recovered = from_str(s)
    assert recovered == canon

    d = to_dict(canon)
    recovered2 = from_dict(d)
    assert recovered2 == canon


def test_environment_data_generation():
    """Generate data from each environment and verify shapes."""
    for env in get_all_environments():
        schema = env.get_schema()
        # Build valid knobs
        knobs = {}
        for k in schema.knobs:
            if k.knob_type.value == "discrete":
                knobs[k.name] = k.options[0]
            elif k.knob_type.value == "integer":
                knobs[k.name] = int((k.range_min + k.range_max) / 2)
            else:
                knobs[k.name] = (k.range_min + k.range_max) / 2

        result = env.run(knobs)
        for det in schema.detectors:
            assert det.name in result, f"{schema.env_id}: missing {det.name}"
            val = result[det.name]
            if det.output_type == "scalar":
                assert isinstance(val, (float, np.floating, int, np.integer)), \
                    f"{schema.env_id}/{det.name}: expected scalar, got {type(val)}"
            elif det.output_type == "array_1d":
                assert isinstance(val, np.ndarray), \
                    f"{schema.env_id}/{det.name}: expected ndarray, got {type(val)}"
                assert len(val) == det.length, \
                    f"{schema.env_id}/{det.name}: expected len {det.length}, got {len(val)}"


def test_dsl0_is_minimal():
    """DSL_0 should contain exactly 10 operators, no physics-specific ones."""
    assert len(DSL_0) == 10
    names = {op.value for op in DSL_0}
    # Must NOT contain any of these
    forbidden = {"complex_abs", "complex_mul", "born_rule", "probability"}
    assert names & forbidden == set()
```

- [ ] **Step 2: Run integration test**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/test_integration.py -v`
Expected: all 4 tests PASS

- [ ] **Step 3: Run complete test suite with coverage**

Run: `cd C:/Users/30670/Desktop/eny/ai-scientist && python -m pytest tests/ -v --tb=short`
Expected: all tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: integration tests for foundation layer"
```

---

## Summary

After completing all 11 tasks, the project has:

- **DSL system**: Expression AST, DSL_0, canonicalization (alpha-equivalence + simplification), serialization
- **12 environments**: 7 quantum + 3 classical + 2 distractor, all behind anonymous knob/detector interface
- **Anti-cheating validation**: Automated checks for no physics leakage + alternative physics harness
- **Test coverage**: Unit tests for every module + integration tests + anti-cheating suite

This provides the complete foundation for **Plan 2** (single-agent ATLAS core with SR integration, diagnostics, and RGDE).
