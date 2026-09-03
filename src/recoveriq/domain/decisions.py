"""Domain models for economic evaluations, recovery proposals, and policy gate decisions."""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import List, Optional

from recoveriq.domain.actions import Action


@dataclass(frozen=True)
class CandidateActionEV:
    """Expected value calculation breakdown for an individual candidate recovery action."""

    action: Action
    estimated_probability: Decimal
    gross_expected_value: Decimal
    intervention_cost: Decimal
    friction_penalty: Decimal
    net_expected_value: Decimal

    @classmethod
    def calculate(
        cls,
        action: Action,
        probability: Decimal,
        payment_amount: Decimal,
        cost: Decimal,
        penalty: Decimal,
    ) -> "CandidateActionEV":
        """Compute EV = P * V - Cost - Penalty with Decimal precision."""
        if not (Decimal("0.0") <= probability <= Decimal("1.0")):
            raise ValueError(f"Probability must be within [0.0, 1.0], got: {probability}")

        gross_ev = probability * payment_amount
        net_ev = gross_ev - cost - penalty

        return cls(
            action=action,
            estimated_probability=probability,
            gross_expected_value=gross_ev,
            intervention_cost=cost,
            friction_penalty=penalty,
            net_expected_value=net_ev,
        )


@dataclass(frozen=True)
class RecoveryDecision:
    """Action recommendation proposed by the economic engine prior to policy authorization."""

    payment_id: str
    proposed_action: Action
    candidate_evaluations: List[CandidateActionEV] = field(default_factory=list)
    rationale: str = ""

    @property
    def best_candidate(self) -> Optional[CandidateActionEV]:
        """Retrieve candidate evaluation matching the proposed action."""
        for candidate in self.candidate_evaluations:
            if candidate.action == self.proposed_action:
                return candidate
        return None


@dataclass(frozen=True)
class PolicyRuleResult:
    """Individual rule evaluation outcome within the deterministic policy gate."""

    rule_name: str
    passed: bool
    message: str = ""


@dataclass(frozen=True)
class PolicyDecision:
    """Final deterministic authorization verdict from the Policy Gate."""

    payment_id: str
    proposed_action: Action
    authorized_action: Action
    is_authorized: bool
    rejection_reason: Optional[str] = None
    rule_results: List[PolicyRuleResult] = field(default_factory=list)

    @classmethod
    def authorize(
        cls,
        payment_id: str,
        action: Action,
        rule_results: Optional[List[PolicyRuleResult]] = None,
    ) -> "PolicyDecision":
        """Factory for an authorized policy decision."""
        return cls(
            payment_id=payment_id,
            proposed_action=action,
            authorized_action=action,
            is_authorized=True,
            rejection_reason=None,
            rule_results=rule_results or [],
        )

    @classmethod
    def reject_and_clamp(
        cls,
        payment_id: str,
        proposed_action: Action,
        fallback_action: Action,
        reason: str,
        rule_results: Optional[List[PolicyRuleResult]] = None,
    ) -> "PolicyDecision":
        """Factory for a rejected policy decision overridden to a safe fallback (STOP/ESCALATE)."""
        return cls(
            payment_id=payment_id,
            proposed_action=proposed_action,
            authorized_action=fallback_action,
            is_authorized=False,
            rejection_reason=reason,
            rule_results=rule_results or [],
        )
