"""Statistical Recovery Probability Model contracts and protocols."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Protocol, runtime_checkable

from recoveriq.domain.actions import Action
from recoveriq.domain.models import PaymentContext


@dataclass(frozen=True)
class ProbabilityEstimate:
    """Estimated probability of successful payment recovery for a candidate action."""

    action: Action
    probability: Decimal
    model_version: str = "stub-v1"

    def __post_init__(self) -> None:
        if not isinstance(self.probability, Decimal):
            object.__setattr__(self, "probability", Decimal(str(self.probability)))
        if not (Decimal("0.0") <= self.probability <= Decimal("1.0")):
            raise ValueError(f"Probability must be within [0.0, 1.0], got {self.probability}")


@runtime_checkable
class RecoveryProbabilityModel(Protocol):
    """Protocol for statistical models estimating recovery probabilities."""

    def estimate_probabilities(
        self,
        context: PaymentContext,
    ) -> Dict[Action, ProbabilityEstimate]:
        """Estimate recovery probability P(recovery | context, action) for all candidate actions."""
        ...


class StubProbabilityModel:
    """Stub statistical model for Sprint 1 foundation contracts.

    Sprint 4 will implement trained, calibrated Logistic Regression models.
    """

    def __init__(self, model_version: str = "stub-v1"):
        self.model_version = model_version

    def estimate_probabilities(
        self,
        context: PaymentContext,
    ) -> Dict[Action, ProbabilityEstimate]:
        """Return baseline placeholder probabilities based on failure category."""
        # Simple contract placeholder:
        estimates: Dict[Action, ProbabilityEstimate] = {}
        for action in Action:
            if action == Action.STOP:
                prob = Decimal("0.00")
            elif context.failure_category.is_hard_decline:
                prob = Decimal("0.00")
            elif action == Action.ESCALATE:
                prob = Decimal("0.85")
            elif action == Action.RETRY_NOW:
                prob = Decimal("0.40")
            elif action == Action.RETRY_LATER:
                prob = Decimal("0.55")
            elif action == Action.SEND_LINK:
                prob = Decimal("0.60")
            elif action == Action.NUDGE:
                prob = Decimal("0.30")
            else:
                prob = Decimal("0.10")

            estimates[action] = ProbabilityEstimate(
                action=action,
                probability=prob,
                model_version=self.model_version,
            )
        return estimates
