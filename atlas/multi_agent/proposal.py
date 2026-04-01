"""Proposal dataclass and pool management."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ProposalStatus(Enum):
    PENDING = "pending"
    ADOPTED = "adopted"
    REJECTED = "rejected"


@dataclass
class Proposal:
    """A DSL extension proposal from an agent."""
    proposal_id: str
    source_agent: str
    source_env: str
    trigger: str
    extension_type: str          # "new_type", "new_operator", "prob_mode"
    extension_definition: dict
    evidence: dict
    status: ProposalStatus = ProposalStatus.PENDING
    delta_total_mdl: float | None = None
    verification_details: dict = field(default_factory=dict)


class ProposalPool:
    """Manages all proposals across epochs."""

    def __init__(self):
        self._proposals: dict[str, Proposal] = {}

    def add(self, proposal: Proposal) -> None:
        self._proposals[proposal.proposal_id] = proposal

    def get(self, proposal_id: str) -> Proposal | None:
        return self._proposals.get(proposal_id)

    def pending(self) -> list[Proposal]:
        return [p for p in self._proposals.values() if p.status == ProposalStatus.PENDING]

    def adopted(self) -> list[Proposal]:
        return [p for p in self._proposals.values() if p.status == ProposalStatus.ADOPTED]

    def rejected(self) -> list[Proposal]:
        return [p for p in self._proposals.values() if p.status == ProposalStatus.REJECTED]

    def set_status(self, proposal_id: str, status: ProposalStatus,
                   delta_total_mdl: float | None = None,
                   verification_details: dict | None = None) -> None:
        p = self._proposals[proposal_id]
        p.status = status
        if delta_total_mdl is not None:
            p.delta_total_mdl = delta_total_mdl
        if verification_details is not None:
            p.verification_details = verification_details

    def all_proposals(self) -> list[Proposal]:
        return list(self._proposals.values())
