"""Theory output structure with layered compression accounting."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LawTemplate:
    """A discovered law template shared across experiments."""
    template_id: str
    template_str: str
    shared_constants: list[str]
    applies_to: list[str]
    compression_savings: float


@dataclass
class CompressionLayer:
    """One layer in the compression chain."""
    level: int
    total_mdl: float
    label: str
    delta: float


class Theory:
    """The final theory output — a layered compression of all experimental data."""

    def __init__(self):
        self.law_templates: list[LawTemplate] = []
        self.shared_constants: list[dict] = []
        self.shared_types: list[dict] = []
        self.experiment_bindings: dict[str, dict] = {}
        self.compression_chain: list[CompressionLayer] = []
        self.extension_lineage: list[dict] = []
        self.fit_metrics: dict[str, dict] = {}

    def add_law_template(self, law: LawTemplate) -> None:
        self.law_templates.append(law)

    def add_shared_constant(self, symbol: str, value: float, uncertainty: float,
                            appearances: list[str], chi2_consistency: float) -> None:
        self.shared_constants.append({
            "symbol": symbol, "value": value, "uncertainty": uncertainty,
            "appearances": appearances, "chi2_consistency": chi2_consistency,
        })

    def add_shared_type(self, name: str, dimension: int, constraints: list[str],
                        appears_in: list[str], compression_savings: float) -> None:
        self.shared_types.append({
            "name": name, "dimension": dimension, "constraints": constraints,
            "appears_in": appears_in, "compression_savings": compression_savings,
        })

    def add_compression_layer(self, layer: CompressionLayer) -> None:
        self.compression_chain.append(layer)

    def compression_ratio(self) -> float:
        if len(self.compression_chain) < 2:
            return 1.0
        first = self.compression_chain[0].total_mdl
        last = self.compression_chain[-1].total_mdl
        return first / last if last > 0 else 1.0

    def to_dict(self) -> dict:
        return {
            "law_templates": [
                {"id": l.template_id, "template": l.template_str,
                 "shared_constants": l.shared_constants, "applies_to": l.applies_to,
                 "compression_savings": l.compression_savings}
                for l in self.law_templates
            ],
            "shared_constants": self.shared_constants,
            "shared_types": self.shared_types,
            "experiment_bindings": self.experiment_bindings,
            "compression_chain": [
                {"level": c.level, "total_mdl": c.total_mdl,
                 "label": c.label, "delta": c.delta}
                for c in self.compression_chain
            ],
            "compression_ratio": self.compression_ratio(),
            "extension_lineage": self.extension_lineage,
            "fit_metrics": self.fit_metrics,
        }
