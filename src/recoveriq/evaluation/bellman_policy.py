"""Sprint 13 — Native Bellman Option Value & Sequential Economic Policy.

Formulates sequential recovery optimization using finite-horizon dynamic programming:
Q_t(a, x) = Immediate_EV(a, x) + Future_Option_Value(a, x)

Where:
- Immediate_EV(a, x) = P_success(a | x) * V - Cost(a) - Penalty(a, x)
- Future_Option_Value(a, x):
    * If a is terminal (STOP or ESCALATE in automated recovery): 0.0
    * If t == max_attempts (final attempt): 0.0
    * If a is non-terminal (RETRY_NOW, RETRY_LATER, SEND_LINK, NUDGE) and t < max_attempts:
        P_failure(a | x) * J_{t+1}(x')
        where J_{t+1}(x') = max_{a'} Q_{t+1}(a', x')
- Option_Value(a) = Q_t(a) - Immediate_EV(a)
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from recoveriq.config.settings import ActionCostConfig, EconomicConfig, PenaltyConfig
from recoveriq.domain.actions import Action
from recoveriq.domain.decisions import CandidateActionEV, RecoveryDecision
from recoveriq.domain.models import CustomerTier, PaymentContext
from recoveriq.domain.state import PaymentState
from recoveriq.economics.engine import DefaultEconomicEngine, EconomicEngine
from recoveriq.model.probability import ProbabilityEstimate, RecoveryProbabilityModel
from recoveriq.simulation.schema import SyntheticPaymentRecord


@dataclass(frozen=True)
class BellmanActionEvaluation:
    """Detailed decomposition of a candidate action evaluated by Bellman dynamic programming."""

    action: Action
    probability: Decimal
    payment_amount: Decimal
    cost: Decimal
    penalty: Decimal
    immediate_ev: Decimal
    future_option_value: Decimal
    total_q_value: Decimal
    option_value: Decimal  # total_q_value - immediate_ev


@dataclass(frozen=True)
class BellmanDecision:
    """Decision output produced by BellmanRecoverIQStrategy with full value decomposition."""

    payment_id: str
    attempt_count: int
    max_attempts: int
    proposed_action: Action
    evaluations: List[BellmanActionEvaluation]
    rationale: str

    @property
    def selected_evaluation(self) -> BellmanActionEvaluation:
        for ev in self.evaluations:
            if ev.action == self.proposed_action:
                return ev
        return self.evaluations[0]


class BellmanRecoverIQStrategy:
    """Native Sequential Bellman Option Value Strategy (Sprint 13).

    Calculates:
        Q_t(a, x) = Immediate_EV(a, x) + Future_Option_Value(a, x)
    and selects:
        a*_t = argmax_a Q_t(a, x)
    """

    name: str = "RecoverIQ-Bellman"

    def __init__(
        self,
        probability_model: RecoveryProbabilityModel,
        economic_config: Optional[EconomicConfig] = None,
        max_attempts: int = 3,
        planning_horizon: Optional[int] = None,
        enable_future_value: bool = True,
    ):
        self.probability_model = probability_model
        self.config = economic_config or EconomicConfig()
        self.max_attempts = max_attempts
        self.planning_horizon = planning_horizon if planning_horizon is not None else max_attempts
        self.enable_future_value = enable_future_value

        # Internal container for logging recent decisions for diagnostic inspection
        self.last_decision: Optional[BellmanDecision] = None

    def evaluate_q_values(
        self,
        record: SyntheticPaymentRecord,
        context: PaymentContext,
        current_attempt: int,
        effective_horizon: int,
    ) -> List[BellmanActionEvaluation]:
        """Compute Q-values for all candidate actions via finite-horizon dynamic programming."""
        amount_decimal = record.amount if isinstance(record.amount, Decimal) else Decimal(str(record.amount))
        probabilities = self.probability_model.estimate_probabilities(context)

        # Base case: if current_attempt >= effective_horizon or future value is disabled, future value is 0
        can_have_future = self.enable_future_value and (current_attempt < effective_horizon)

        # Compute next-stage optimal value J_{t+1}(x') if future transitions are allowed
        next_stage_optimal_j = Decimal("0.00")
        if can_have_future:
            # Construct counterfactual next-attempt context x' (attempt_count + 1)
            next_context = PaymentContext(
                payment_id=context.payment_id,
                customer_id=context.customer_id,
                customer_tier=context.customer_tier,
                payment_method=context.payment_method,
                raw_error_code=context.raw_error_code,
                raw_error_message=context.raw_error_message,
                failure_category=context.failure_category,
                failure_severity=context.failure_severity,
                attempt_count=current_attempt + 1,
                last_attempt_timestamp=context.last_attempt_timestamp,
                extra_metadata=context.extra_metadata,
            )
            # Recursively compute Q_{t+1}(a', x')
            next_evaluations = self.evaluate_q_values(
                record=record,
                context=next_context,
                current_attempt=current_attempt + 1,
                effective_horizon=effective_horizon,
            )
            # J_{t+1}(x') = max_{a'} Q_{t+1}(a', x')
            # If all Q-values are non-positive, agent can choose STOP (terminal value 0)
            max_next_q = max(ev.total_q_value for ev in next_evaluations)
            next_stage_optimal_j = max(Decimal("0.00"), max_next_q)

        evaluations: List[BellmanActionEvaluation] = []

        for action in Action:
            prob_est = probabilities.get(
                action,
                ProbabilityEstimate(action=action, probability=Decimal("0.0")),
            )
            p_success = prob_est.probability
            p_failure = Decimal("1.0") - p_success

            cost = self.config.cost_config.get_cost(action)
            penalty = self.config.penalty_config.get_penalty(action, context.customer_tier)

            # Immediate EV = P(success)*V - Cost - Penalty
            immediate_ev = p_success * amount_decimal - cost - penalty

            # Determine Future Option Value:
            # - STOP and ESCALATE are terminal in automated recovery -> future automated value = 0
            # - Non-terminal retry actions transition to next attempt upon failure with probability (1 - p_success)
            if action in (Action.STOP, Action.ESCALATE) or not can_have_future:
                future_option_value = Decimal("0.00")
            else:
                # Value to go: if failed, we proceed to attempt t+1 and earn J_{t+1}(x')
                future_option_value = p_failure * next_stage_optimal_j

            total_q_value = immediate_ev + future_option_value
            option_value = future_option_value

            evaluations.append(
                BellmanActionEvaluation(
                    action=action,
                    probability=p_success,
                    payment_amount=amount_decimal,
                    cost=cost,
                    penalty=penalty,
                    immediate_ev=immediate_ev,
                    future_option_value=future_option_value,
                    total_q_value=total_q_value,
                    option_value=option_value,
                )
            )

        # Sort descending by Total Q-Value
        evaluations.sort(key=lambda x: x.total_q_value, reverse=True)
        return evaluations

    def propose_action(
        self,
        record: SyntheticPaymentRecord,
        context: PaymentContext,
    ) -> Action:
        """Select action maximizing Bellman Q-value at current attempt."""
        current_attempt = context.attempt_count if context.attempt_count > 0 else 1
        evaluations = self.evaluate_q_values(
            record=record,
            context=context,
            current_attempt=current_attempt,
            effective_horizon=self.planning_horizon,
        )

        best = evaluations[0]

        # If best total Q-value is non-positive or below threshold, propose STOP
        if best.total_q_value <= self.config.min_ev_threshold:
            proposed_action = Action.STOP
            rationale = (
                f"Maximum Bellman Q ({best.total_q_value:.2f}) <= threshold "
                f"({self.config.min_ev_threshold:.2f}); defaulting to STOP."
            )
        else:
            proposed_action = best.action
            rationale = (
                f"Selected {best.action.value} with highest Bellman Q ({best.total_q_value:.2f}) "
                f"[Immediate EV={best.immediate_ev:.2f}, Future Option Value={best.future_option_value:.2f}, "
                f"P={best.probability:.2%}, Cost={best.cost:.2f}]."
            )

        self.last_decision = BellmanDecision(
            payment_id=context.payment_id,
            attempt_count=current_attempt,
            max_attempts=self.max_attempts,
            proposed_action=proposed_action,
            evaluations=evaluations,
            rationale=rationale,
        )

        return proposed_action
