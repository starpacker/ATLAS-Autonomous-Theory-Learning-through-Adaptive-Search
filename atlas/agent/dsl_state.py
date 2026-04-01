"""DSL state: tracks current operators, concepts, and extensions."""
from __future__ import annotations
from atlas.dsl.operators import Op, DSL_0
from atlas.dsl.expr import Expr


class DSLState:
    def __init__(self):
        self.operators: frozenset[Op] = DSL_0
        self.concepts: dict[str, Expr] = {}
        self.extensions: list[dict] = []
        self._history: list[dict] = []

    def add_concept(self, name: str, expr: Expr) -> None:
        self.concepts[name] = expr
        self._history.append({"action": "add_concept", "name": name})

    def add_extension(self, name: str, ext_type: str, definition: dict, trigger: str) -> None:
        ext = {"name": name, "type": ext_type, "definition": definition, "trigger": trigger}
        self.extensions.append(ext)
        self._history.append({"action": "add_extension", "name": name, "type": ext_type})

    def mdl_cost(self) -> float:
        cost = float(len(self.operators))
        for expr in self.concepts.values():
            cost += expr.size()
        for ext in self.extensions:
            cost += 5.0
        return cost

    def snapshot(self) -> dict:
        return {"operators": self.operators, "concepts": dict(self.concepts),
                "extensions": list(self.extensions)}

    def restore(self, snap: dict) -> None:
        self.operators = snap["operators"]
        self.concepts = dict(snap["concepts"])
        self.extensions = list(snap["extensions"])

    @property
    def history(self) -> list[dict]:
        return list(self._history)
