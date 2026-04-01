"""FormulaStore: store, query, and compare discovered formulas."""
from __future__ import annotations
from dataclasses import dataclass
from atlas.dsl.expr import Expr, Const, Var, BinOp, UnaryOp, NAryOp
from atlas.types import FitMetrics


@dataclass
class StoredFormula:
    env_id: str
    expr: Expr
    fit: FitMetrics


class FormulaStore:
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
        constants: list[float] = []
        for formulas in self._formulas.values():
            for sf in formulas:
                constants.extend(_extract_constants(sf.expr))
        return constants

    def pareto_front(self, env_id: str) -> list[StoredFormula]:
        formulas = self.get(env_id)
        if not formulas:
            return []
        pareto = []
        for f in formulas:
            dominated = False
            for other in formulas:
                if (other.fit.r_squared > f.fit.r_squared and other.fit.mdl < f.fit.mdl):
                    dominated = True
                    break
            if not dominated:
                pareto.append(f)
        return pareto


def _extract_constants(expr: Expr) -> list[float]:
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
