"""RGDE Step 4d: Build DSLType from encoder formulas and constraints."""
from __future__ import annotations
from dataclasses import dataclass
from atlas.dsl.expr import Expr
from atlas.dsl.serialize import to_str
from atlas.rgde.constraint_finder import Constraint


@dataclass
class DSLType:
    name: str
    dimension: int
    encoding: dict[int, Expr]
    constraints: list[Constraint]
    source_env: str

    def mdl_cost(self) -> float:
        cost = 1.0
        for expr in self.encoding.values():
            cost += expr.size()
        cost += len(self.constraints) * 3.0
        return cost

    def to_dict(self) -> dict:
        return {"name": self.name, "dimension": self.dimension,
                "encoding": {k: to_str(v) for k, v in self.encoding.items()},
                "constraints": [{"terms": c.terms, "constant": c.constant,
                                 "residual": c.residual, "type": c.constraint_type}
                                for c in self.constraints],
                "source_env": self.source_env}


def build_type(env_id: str, encoder_formulas: dict[int, Expr],
               constraints: list[Constraint]) -> DSLType:
    return DSLType(name=f"State_{env_id}", dimension=len(encoder_formulas),
                   encoding=dict(encoder_formulas), constraints=list(constraints),
                   source_env=env_id)
