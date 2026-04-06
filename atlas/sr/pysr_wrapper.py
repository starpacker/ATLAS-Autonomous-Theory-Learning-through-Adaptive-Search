"""PySR wrapper: config, result types, and expression parser.

The primary public API:
- SRConfig          — PySR hyperparameters, buildable from a DSL operator set.
- SRResult          — Pareto front results returned by run_sr().
- run_sr()          — Runs PySR and returns SRResult (requires pysr installed).
- pysr_expr_to_atlas() — Converts a PySR string expression to an ATLAS Expr AST.
"""
from __future__ import annotations

import logging
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

from atlas.dsl.expr import BinOp, Const, Expr, UnaryOp, Var
from atlas.dsl.operators import DSL_0, Op

# ---------------------------------------------------------------------------
# Operator mappings
# ---------------------------------------------------------------------------

_BINARY_OP_TO_STR: dict[Op, str] = {
    Op.ADD: "+",
    Op.SUB: "-",
    Op.MUL: "*",
    Op.DIV: "/",
    Op.POW: "^",
}

_UNARY_OP_TO_STR: dict[Op, str] = {
    Op.SIN: "sin",
    Op.COS: "cos",
    Op.EXP: "exp",
    Op.LOG: "log",
    Op.NEG: "neg",
}

_STR_TO_BINARY_OP: dict[str, Op] = {v: k for k, v in _BINARY_OP_TO_STR.items()}
_STR_TO_UNARY_OP: dict[str, Op] = {v: k for k, v in _UNARY_OP_TO_STR.items()}

_UNARY_FUNC_NAMES = frozenset(_STR_TO_UNARY_OP.keys())

# ---------------------------------------------------------------------------
# SRConfig
# ---------------------------------------------------------------------------


@dataclass
class SRConfig:
    """Hyperparameters for a PySR run."""

    niterations: int = 40
    populations: int = 15
    maxsize: int = 25
    binary_operators: list[str] = field(
        default_factory=lambda: ["+", "-", "*", "/", "^"]
    )
    unary_operators: list[str] = field(
        default_factory=lambda: ["sin", "cos", "exp", "log", "neg"]
    )
    timeout_in_seconds: int = 300
    procs: int = -1  # -1 = don't pass to PySR (use its default); 0 = single-process
    deterministic: bool = False

    @classmethod
    def from_dsl(cls, ops: frozenset[Op]) -> "SRConfig":
        """Build an SRConfig restricted to the operators in *ops*."""
        binary_ops = [
            sym for op, sym in _BINARY_OP_TO_STR.items() if op in ops
        ]
        unary_ops = [
            sym for op, sym in _UNARY_OP_TO_STR.items() if op in ops
        ]
        return cls(binary_operators=binary_ops, unary_operators=unary_ops)


# ---------------------------------------------------------------------------
# SRResult
# ---------------------------------------------------------------------------


@dataclass
class SRResult:
    """Results from a PySR symbolic regression run.

    Attributes
    ----------
    formulas:
        All Pareto-front formulas as ATLAS Expr nodes (may be empty).
    best_formula:
        The best formula (highest R² on the Pareto front), or None.
    best_r_squared:
        R² of the best formula.
    best_mdl:
        MDL score of the best formula (lower is better).
    converged:
        True when PySR reported convergence.
    raw:
        Raw PySR DataFrame (present only when pysr is installed).
    """

    formulas: list[Expr]
    best_formula: Optional[Expr]
    best_r_squared: float
    best_mdl: float
    converged: bool
    raw: Any = None


# ---------------------------------------------------------------------------
# run_sr
# ---------------------------------------------------------------------------


def run_sr(
    X: np.ndarray,
    y: np.ndarray,
    var_names: list[str],
    config: Optional[SRConfig] = None,
) -> SRResult:
    """Run PySR symbolic regression and return an SRResult.

    Parameters
    ----------
    X:
        Shape (n_samples, n_features).
    y:
        Shape (n_samples,).
    var_names:
        Variable names for each column of X (mapped to x0, x1, …).
    config:
        PySR configuration. Defaults to SRConfig().

    Raises
    ------
    ImportError
        If pysr is not installed.
    """
    try:
        from pysr import PySRRegressor  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pysr is required for run_sr(). Install it with: pip install pysr"
        ) from exc

    if config is None:
        config = SRConfig()

    # PySR uses x0, x1, … internally; we map those back to var_names.
    pysr_var_names = [f"x{i}" for i in range(X.shape[1])]

    # Use a temporary directory for PySR artifacts to avoid clutter
    tmpdir = tempfile.mkdtemp(prefix="pysr_")

    # Build PySR kwargs; deterministic mode requires parallelism='serial'
    pysr_kwargs = dict(
        niterations=config.niterations,
        populations=config.populations,
        maxsize=config.maxsize,
        binary_operators=config.binary_operators,
        unary_operators=config.unary_operators,
        variable_names=pysr_var_names,
        timeout_in_seconds=config.timeout_in_seconds,
        tempdir=tmpdir,
    )
    if config.procs >= 0:
        pysr_kwargs["procs"] = config.procs
    if config.deterministic:
        pysr_kwargs["deterministic"] = True
        pysr_kwargs["parallelism"] = "serial"
        pysr_kwargs["random_state"] = 42

    model = PySRRegressor(**pysr_kwargs)

    _empty = SRResult(
        formulas=[], best_formula=None, best_r_squared=-1.0,
        best_mdl=float("inf"), converged=False, raw=None,
    )

    try:
        model.fit(X, y)
    except (RuntimeError, OSError, Exception) as exc:
        logger.warning("PySR crashed during fit: %s", exc)
        return _empty

    equations = model.equations_
    if equations is None:
        return _empty

    formulas: list[Expr] = []
    best_formula: Optional[Expr] = None
    best_r_squared = -1.0
    best_mdl = float("inf")

    for _, row in equations.iterrows():
        try:
            expr = pysr_expr_to_atlas(row["equation"], var_names=var_names)
        except Exception:
            continue
        formulas.append(expr)
        r2 = float(row.get("r2", -1.0))
        mdl = float(expr.mdl_cost())
        if r2 > best_r_squared:
            best_r_squared = r2
            best_formula = expr
            best_mdl = mdl

    converged = bool(getattr(model, "converged_", False))

    return SRResult(
        formulas=formulas,
        best_formula=best_formula,
        best_r_squared=best_r_squared,
        best_mdl=best_mdl,
        converged=converged,
        raw=equations,
    )


# ---------------------------------------------------------------------------
# Expression parser: PySR string -> ATLAS Expr
# ---------------------------------------------------------------------------


class _ParseError(Exception):
    """Raised when the expression string cannot be parsed."""


class _Parser:
    """Recursive-descent parser for PySR expression strings.

    Grammar (operator precedence, low to high):
        expr    ::= additive
        additive ::= multiplicative (('+' | '-') multiplicative)*
        multiplicative ::= power (('*' | '/') power)*
        power   ::= unary ('^' unary)*      (right-associative)
        unary   ::= '-' unary | primary
        primary ::= number | variable | func '(' expr ')' | '(' expr ')'

    Variable tokens: x0, x1, x2, …
    Function tokens: sin, cos, exp, log, neg
    Number tokens: optional sign, digits, optional decimal + exponent.
    """

    _TOKEN_RE = re.compile(
        r"""
        \s*                                          # skip whitespace
        (?:
            ([+\-*/^()])                             # single-char operators/parens
          | ((?:\d+\.?\d*|\.\d+)(?:[eE][+\-]?\d+)?) # numeric literal
          | ([a-zA-Z_]\w*)                            # identifier (variable or func)
        )
        \s*
        """,
        re.VERBOSE,
    )

    def __init__(self, text: str, var_names: list[str]) -> None:
        self._text = text
        self._var_names = var_names
        self._pos = 0
        self._tokens: list[str] = []
        self._tok_pos = 0
        self._tokenize()

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    def _tokenize(self) -> None:
        tokens: list[str] = []
        pos = 0
        text = self._text
        while pos < len(text):
            m = self._TOKEN_RE.match(text, pos)
            if not m:
                raise _ParseError(f"Unexpected character at position {pos}: {text[pos:]!r}")
            op, num, ident = m.group(1), m.group(2), m.group(3)
            if op is not None:
                tokens.append(op)
            elif num is not None:
                tokens.append(num)
            elif ident is not None:
                tokens.append(ident)
            pos = m.end()
        self._tokens = tokens

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------

    def _peek(self) -> Optional[str]:
        if self._tok_pos < len(self._tokens):
            return self._tokens[self._tok_pos]
        return None

    def _consume(self) -> str:
        tok = self._tokens[self._tok_pos]
        self._tok_pos += 1
        return tok

    def _expect(self, value: str) -> str:
        tok = self._peek()
        if tok != value:
            raise _ParseError(f"Expected {value!r}, got {tok!r}")
        return self._consume()

    # ------------------------------------------------------------------
    # Grammar rules
    # ------------------------------------------------------------------

    def parse(self) -> Expr:
        expr = self._additive()
        if self._peek() is not None:
            raise _ParseError(
                f"Unexpected token after expression: {self._peek()!r}"
            )
        return expr

    def _additive(self) -> Expr:
        left = self._multiplicative()
        while self._peek() in ("+", "-"):
            op_str = self._consume()
            right = self._multiplicative()
            op = Op.ADD if op_str == "+" else Op.SUB
            left = BinOp(op, left, right)
        return left

    def _multiplicative(self) -> Expr:
        left = self._power()
        while self._peek() in ("*", "/"):
            op_str = self._consume()
            right = self._power()
            op = Op.MUL if op_str == "*" else Op.DIV
            left = BinOp(op, left, right)
        return left

    def _power(self) -> Expr:
        base = self._unary()
        if self._peek() == "^":
            self._consume()
            exponent = self._power()  # right-associative: recurse into _power
            return BinOp(Op.POW, base, exponent)
        return base

    def _unary(self) -> Expr:
        if self._peek() == "-":
            self._consume()
            operand = self._unary()
            # Fold constant negation immediately for cleanliness
            if isinstance(operand, Const):
                return Const(-operand.value)
            return UnaryOp(Op.NEG, operand)
        return self._primary()

    def _primary(self) -> Expr:
        tok = self._peek()
        if tok is None:
            raise _ParseError("Unexpected end of expression")

        # Parenthesized sub-expression
        if tok == "(":
            self._consume()
            expr = self._additive()
            self._expect(")")
            return expr

        # Numeric literal
        if re.fullmatch(r"(?:\d+\.?\d*|\.\d+)(?:[eE][+\-]?\d+)?", tok):
            self._consume()
            return Const(float(tok))

        # Identifier: variable or function call
        if re.fullmatch(r"[a-zA-Z_]\w*", tok):
            self._consume()

            # Function call
            if tok in _UNARY_FUNC_NAMES and self._peek() == "(":
                self._consume()  # consume '('
                arg = self._additive()
                self._expect(")")
                return UnaryOp(_STR_TO_UNARY_OP[tok], arg)

            # Variable reference: x0, x1, …
            m = re.fullmatch(r"x(\d+)", tok)
            if m:
                idx = int(m.group(1))
                if idx >= len(self._var_names):
                    raise _ParseError(
                        f"Variable index {idx} out of range for var_names={self._var_names}"
                    )
                return Var(self._var_names[idx])

            raise _ParseError(f"Unknown identifier: {tok!r}")

        raise _ParseError(f"Cannot parse token: {tok!r}")


def pysr_expr_to_atlas(expr_str: str, var_names: list[str]) -> Expr:
    """Parse a PySR expression string into an ATLAS Expr AST.

    Parameters
    ----------
    expr_str:
        A string as produced by PySR, e.g. ``"sin(x0 * 3.14) + x1"``.
    var_names:
        Ordered list of ATLAS variable names. ``x0`` maps to ``var_names[0]``,
        ``x1`` to ``var_names[1]``, etc.

    Returns
    -------
    Expr
        The corresponding ATLAS expression tree.

    Raises
    ------
    _ParseError
        If the string cannot be parsed.
    """
    parser = _Parser(expr_str.strip(), var_names)
    return parser.parse()
