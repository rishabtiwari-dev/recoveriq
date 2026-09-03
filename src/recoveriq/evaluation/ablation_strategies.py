"""Ablation strategies for RecoverIQ evaluation and component attribution (SPEC.md Section 17).

ABLATION EXPERIMENTS:
1. A1 — Context-Source Ablation (RecoverIQCtxAblationStrategy):
   Replaces generator-oracle categorical context with deterministic rule-based
   extraction via RuleBasedContextExtractor applied to raw error codes/messages.
   RESEARCH NOTE: Sprint 5's full RecoverIQ directly passed generator-assigned
   failure_category into PaymentContext (an oracle shortcut). This ablation isolates
   the performance impact of rule-based context extraction vs generator-oracle context,
   NOT LLM vs rules.

2. A2 — Economic Engine Ablation (RecoverIQNoEconStrategy):
   Replaces Net Expected Value (EV) optimization (argmax_a EV(a)) with greedy
   probability maximization (argmax_a P̂(a)), ignoring direct intervention costs
   and friction penalties during action selection.
   Post-hoc financial outcome accounting remains identical.
"""

import uuid
from decimal import Decimal
from typing import Dict, Optional

from recoveriq.context.extractor import ContextExtractor, RuleBasedContextExtractor
from recoveriq.domain.actions import Action
from recoveriq.domain.events import EventType, PaymentFailedEvent
from recoveriq.domain.models import PaymentContext
from recoveriq.economics.engine import DefaultEconomicEngine, EconomicEngine
from recoveriq.model.probability import ProbabilityEstimate, RecoveryProbabilityModel
from recoveriq.simulation.schema import SyntheticPaymentRecord


class GreedyProbabilitySelector:
    """Greedy probability action selector for economic engine ablation (A2).

    Selects the candidate action that maximizes estimated recovery probability:
        a_selected = argmax_{a in Actions \\ {STOP}} P̂(recovery | x, a)

    Domain invariant:
        STOP is excluded from probability argmax because P(STOP) = 0.00 by definition.
        If all candidate actions have estimated probability 0.00, STOP is proposed.
    """

    @staticmethod
    def select(probabilities: Dict[Action, ProbabilityEstimate]) -> Action:
        """Select action with highest estimated recovery probability (excluding STOP)."""
        best_action: Optional[Action] = None
        best_prob: Decimal = Decimal("-1.0")

        # Evaluate all candidate actions excluding STOP
        for action in Action:
            if action == Action.STOP:
                continue
            prob_est = probabilities.get(action)
            prob = prob_est.probability if prob_est else Decimal("0.0")
            if prob > best_prob:
                best_prob = prob
                best_action = action

        # If best non-STOP probability is non-positive or unavailable, propose STOP
        if best_action is None or best_prob <= Decimal("0.0"):
            return Action.STOP

        return best_action


class RecoverIQCtxAblationStrategy:
    """Ablation A1 — Context-Source Ablation (Generator-Oracle vs Rule-Based Extraction).

    In the Sprint 5 baseline, PaymentContext is constructed directly with the generator's
    assigned failure_category and failure_severity (oracle context).
    This strategy reconstructs a synthetic PaymentFailedEvent from the record's raw fields
    and executes RuleBasedContextExtractor to obtain categorical signals.
    """

    name: str = "RecoverIQ-CtxAblation"

    def __init__(
        self,
        probability_model: RecoveryProbabilityModel,
        economic_engine: Optional[EconomicEngine] = None,
        context_extractor: Optional[ContextExtractor] = None,
    ):
        self.probability_model = probability_model
        self.economic_engine = economic_engine or DefaultEconomicEngine()
        self.context_extractor = context_extractor or RuleBasedContextExtractor()

    def propose_action(
        self,
        record: SyntheticPaymentRecord,
        context: PaymentContext,
    ) -> Action:
        """Extract context via rule-based keyword parser on raw error fields and optimize EV."""
        # Synthesize PaymentFailedEvent strictly from observable record fields
        event = PaymentFailedEvent(
            event_id=str(uuid.uuid4()),
            payment_id=record.payment_id,
            event_type=EventType.PAYMENT_FAILED,
            timestamp=record.failure_timestamp,
            customer_id=record.customer_id,
            amount=record.amount if isinstance(record.amount, Decimal) else Decimal(str(record.amount)),
            currency=record.currency,
            customer_tier=record.customer_tier,
            payment_method=record.payment_method,
            raw_error_code=record.raw_error_code,
            raw_error_message=record.raw_error_message,
            attempt_count=record.attempt_count,
        )

        # 1. Context extraction via RuleBasedContextExtractor
        extracted_context = self.context_extractor.extract_context(event)

        # Preserve amount metadata for preprocessor
        amount_decimal = record.amount if isinstance(record.amount, Decimal) else Decimal(str(record.amount))
        augmented_context = PaymentContext(
            payment_id=extracted_context.payment_id,
            customer_id=extracted_context.customer_id,
            customer_tier=extracted_context.customer_tier,
            payment_method=extracted_context.payment_method,
            raw_error_code=extracted_context.raw_error_code,
            raw_error_message=extracted_context.raw_error_message,
            failure_category=extracted_context.failure_category,
            failure_severity=extracted_context.failure_severity,
            attempt_count=extracted_context.attempt_count,
            last_attempt_timestamp=extracted_context.last_attempt_timestamp,
            diagnostic_explanation=extracted_context.diagnostic_explanation,
            extra_metadata={"amount": float(amount_decimal)},
        )

        # 2. Probability estimation using the ablated context
        probabilities = self.probability_model.estimate_probabilities(augmented_context)

        # 3. Expected Value optimization with standard economic engine
        decision = self.economic_engine.evaluate_actions(
            context=augmented_context,
            payment_amount=amount_decimal,
            probabilities=probabilities,
        )
        return decision.proposed_action


class RecoverIQNoEconStrategy:
    """Ablation A2 — Economic Engine Ablation (Greedy Probability vs EV Optimization).

    Preserves the standard PaymentContext and trained probability model, but replaces
    the Net Expected Value optimizer with GreedyProbabilitySelector (argmax_a P̂(a)).
    Cost and penalty terms are ignored during action selection.
    """

    name: str = "RecoverIQ-NoEcon"

    def __init__(
        self,
        probability_model: RecoveryProbabilityModel,
        selector: Optional[GreedyProbabilitySelector] = None,
    ):
        self.probability_model = probability_model
        self.selector = selector or GreedyProbabilitySelector()

    def propose_action(
        self,
        record: SyntheticPaymentRecord,
        context: PaymentContext,
    ) -> Action:
        """Estimate recovery probabilities and select greedy probability maximizer."""
        probabilities = self.probability_model.estimate_probabilities(context)
        return self.selector.select(probabilities)
