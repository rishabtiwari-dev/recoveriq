"""Economic optimization engine contracts and Expected Value (EV) calculation."""

from decimal import Decimal
from typing import Dict, List, Protocol, runtime_checkable

from recoveriq.config.settings import EconomicConfig
from recoveriq.domain.actions import Action
from recoveriq.domain.decisions import CandidateActionEV, RecoveryDecision
from recoveriq.domain.models import PaymentContext
from recoveriq.model.probability import ProbabilityEstimate


@runtime_checkable
class EconomicEngine(Protocol):
    """Protocol for evaluating candidate actions and calculating net expected value."""

    def evaluate_actions(
        self,
        context: PaymentContext,
        payment_amount: Decimal,
        probabilities: Dict[Action, ProbabilityEstimate],
    ) -> RecoveryDecision:
        """Evaluate candidate actions and propose candidate action with highest positive EV."""
        ...


class DefaultEconomicEngine:
    """Calculates Net Expected Value (EV) for each action and nominates the top candidate.

    EV(a) = P(recovery | context, a) * V - Cost(a) - Penalty(a, context)
    """

    def __init__(self, config: EconomicConfig = None):
        self.config = config or EconomicConfig()

    def evaluate_actions(
        self,
        context: PaymentContext,
        payment_amount: Decimal,
        probabilities: Dict[Action, ProbabilityEstimate],
    ) -> RecoveryDecision:
        if not isinstance(payment_amount, Decimal):
            payment_amount = Decimal(str(payment_amount))

        evaluations: List[CandidateActionEV] = []

        for action in Action:
            prob_est = probabilities.get(
                action,
                ProbabilityEstimate(action=action, probability=Decimal("0.0")),
            )
            prob = prob_est.probability

            cost = self.config.cost_config.get_cost(action)
            penalty = self.config.penalty_config.get_penalty(action, context.customer_tier)

            ev_record = CandidateActionEV.calculate(
                action=action,
                probability=prob,
                payment_amount=payment_amount,
                cost=cost,
                penalty=penalty,
            )
            evaluations.append(ev_record)

        # Sort descending by Net EV
        evaluations.sort(key=lambda x: x.net_expected_value, reverse=True)

        best = evaluations[0]

        # If best EV is non-positive or below threshold, propose STOP
        if best.net_expected_value <= self.config.min_ev_threshold:
            proposed_action = Action.STOP
            rationale = (
                f"Maximum Net EV ({best.net_expected_value:.2f}) <= threshold "
                f"({self.config.min_ev_threshold:.2f}); defaulting to STOP."
            )
        else:
            proposed_action = best.action
            rationale = (
                f"Selected {best.action.value} with highest Net EV ({best.net_expected_value:.2f}) "
                f"[P={best.estimated_probability:.2%}, Cost={best.intervention_cost:.2f}, "
                f"Penalty={best.friction_penalty:.2f}]."
            )

        return RecoveryDecision(
            payment_id=context.payment_id,
            proposed_action=proposed_action,
            candidate_evaluations=evaluations,
            rationale=rationale,
        )
