"""Strategy definitions for RecoverIQ evaluation and baseline benchmarking.

THREE EVALUATION STRATEGIES (SPEC.md Section 14):
1. RecoverIQStrategy: Full system combining context, statistical probability model,
   economic expected value (EV) optimization, and policy gating.
2. FixedRetryStrategy: Static, context-agnostic baseline that unconditionally proposes
   Action.RETRY_NOW at the current decision point.
3. RuleBasedStrategy: Deterministic heuristic baseline mapping error categories to actions
   without probabilistic estimation or economic optimization.

HARD SAFETY INVARIANT:
All strategies must submit their proposed action to the common PolicyGate.
No baseline is permitted to bypass the policy gate.
"""

from typing import Dict, Optional, Protocol, runtime_checkable
from decimal import Decimal

from recoveriq.domain.actions import Action
from recoveriq.domain.decisions import CandidateActionEV, RecoveryDecision
from recoveriq.domain.models import CustomerTier, FailureCategory, PaymentContext
from recoveriq.economics.engine import DefaultEconomicEngine, EconomicEngine
from recoveriq.model.probability import RecoveryProbabilityModel
from recoveriq.simulation.schema import SyntheticPaymentRecord


@runtime_checkable
class RecoveryStrategy(Protocol):
    """Protocol for recovery decision strategies evaluated in Sprint 5."""

    name: str

    def propose_action(
        self,
        record: SyntheticPaymentRecord,
        context: PaymentContext,
    ) -> Action:
        """Propose a candidate action prior to policy authorization."""
        ...


class FixedRetryStrategy:
    """Fixed-Retry Baseline.

    In the single-decision-point scope, this strategy unconditionally proposes Action.RETRY_NOW.
    LIMITATION NOTE: This represents the initial attempt of a fixed-retry schedule;
    multi-round cooldown trajectories are out of scope for Sprint 5.
    """

    name: str = "Fixed-Retry"

    def propose_action(
        self,
        record: SyntheticPaymentRecord,
        context: PaymentContext,
    ) -> Action:
        """Unconditionally propose immediate gateway retry."""
        return Action.RETRY_NOW


class RuleBasedStrategy:
    """Rule-Based Baseline (SPEC.md Section 14).

    Applies deterministic heuristics mirroring standard industry payment operations:
    - INSUFFICIENT_FUNDS -> RETRY_LATER
    - NETWORK_TIMEOUT -> RETRY_NOW
    - CARD_EXPIRED -> SEND_LINK
    - AUTHENTICATION_FAILED -> SEND_LINK
    - AUTHENTICATION_REJECTED -> SEND_LINK
    - VELOCITY_EXCEEDED -> RETRY_LATER
    - HARD_DECLINE / INVALID_DETAILS -> STOP
    - UNKNOWN / other -> STOP
    """

    name: str = "Rule-Based"

    def __init__(self, rule_mappings: Optional[Dict[FailureCategory, Action]] = None):
        self.rule_mappings = rule_mappings or {
            FailureCategory.INSUFFICIENT_FUNDS: Action.RETRY_LATER,
            FailureCategory.NETWORK_TIMEOUT: Action.RETRY_NOW,
            FailureCategory.CARD_EXPIRED: Action.SEND_LINK,
            FailureCategory.AUTHENTICATION_FAILED: Action.SEND_LINK,
            FailureCategory.AUTHENTICATION_REJECTED: Action.SEND_LINK,
            FailureCategory.VELOCITY_EXCEEDED: Action.RETRY_LATER,
            FailureCategory.HARD_DECLINE: Action.STOP,
            FailureCategory.INVALID_DETAILS: Action.STOP,
            FailureCategory.UNKNOWN: Action.STOP,
        }

    def propose_action(
        self,
        record: SyntheticPaymentRecord,
        context: PaymentContext,
    ) -> Action:
        """Evaluate deterministic rules on failure category."""
        return self.rule_mappings.get(context.failure_category, Action.STOP)


class RecoverIQStrategy:
    """RecoverIQ Full System Strategy.

    Evaluates recovery probabilities using the trained statistical model,
    optimizes Expected Value (EV) via the Economic Engine, and proposes
    argmax_a EV(a).
    """

    name: str = "RecoverIQ"

    def __init__(
        self,
        probability_model: RecoveryProbabilityModel,
        economic_engine: Optional[EconomicEngine] = None,
    ):
        self.probability_model = probability_model
        self.economic_engine = economic_engine or DefaultEconomicEngine()

    def propose_action(
        self,
        record: SyntheticPaymentRecord,
        context: PaymentContext,
    ) -> Action:
        """Compute EV for all candidate actions and propose the EV-maximizing action."""
        probabilities = self.probability_model.estimate_probabilities(context)
        amount_decimal = record.amount if isinstance(record.amount, Decimal) else Decimal(str(record.amount))

        decision = self.economic_engine.evaluate_actions(
            context=context,
            payment_amount=amount_decimal,
            probabilities=probabilities,
        )
        return decision.proposed_action
