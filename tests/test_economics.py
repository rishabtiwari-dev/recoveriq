"""Tests for the Economic Engine and Expected Value calculations."""

from decimal import Decimal
import pytest

from recoveriq.config.settings import EconomicConfig
from recoveriq.domain.actions import Action
from recoveriq.domain.decisions import CandidateActionEV
from recoveriq.domain.models import (
    CustomerTier,
    FailureCategory,
    FailureSeverity,
    PaymentContext,
    PaymentMethod,
)
from recoveriq.economics.engine import DefaultEconomicEngine
from recoveriq.model.probability import ProbabilityEstimate


def test_candidate_action_ev_calculation():
    """Verify EV = P * V - Cost - Penalty calculation accuracy."""
    ev_record = CandidateActionEV.calculate(
        action=Action.RETRY_NOW,
        probability=Decimal("0.60"),
        payment_amount=Decimal("100.00"),
        cost=Decimal("0.15"),
        penalty=Decimal("0.05"),
    )
    # Gross EV = 0.60 * 100.00 = 60.00
    # Net EV = 60.00 - 0.15 - 0.05 = 59.80
    assert ev_record.gross_expected_value == Decimal("60.00")
    assert ev_record.intervention_cost == Decimal("0.15")
    assert ev_record.friction_penalty == Decimal("0.05")
    assert ev_record.net_expected_value == Decimal("59.80")


def test_candidate_action_ev_invalid_probability():
    """Verify invalid probability ranges are rejected."""
    with pytest.raises(ValueError, match="within"):
        CandidateActionEV.calculate(
            action=Action.RETRY_NOW,
            probability=Decimal("1.5"),
            payment_amount=Decimal("100.00"),
            cost=Decimal("0.15"),
            penalty=Decimal("0.05"),
        )


def test_economic_engine_selects_highest_ev():
    """Verify engine ranks actions and proposes the argmax EV."""
    engine = DefaultEconomicEngine()
    context = PaymentContext(
        payment_id="pay_ev_1",
        customer_id="cust_ev_1",
        customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="TIMEOUT",
        raw_error_message="Gateway timeout",
        failure_category=FailureCategory.NETWORK_TIMEOUT,
        failure_severity=FailureSeverity.TRANSIENT,
    )
    probabilities = {
        Action.RETRY_NOW: ProbabilityEstimate(Action.RETRY_NOW, Decimal("0.70")),
        Action.RETRY_LATER: ProbabilityEstimate(Action.RETRY_LATER, Decimal("0.50")),
        Action.SEND_LINK: ProbabilityEstimate(Action.SEND_LINK, Decimal("0.30")),
        Action.NUDGE: ProbabilityEstimate(Action.NUDGE, Decimal("0.20")),
        Action.ESCALATE: ProbabilityEstimate(Action.ESCALATE, Decimal("0.80")),
        Action.STOP: ProbabilityEstimate(Action.STOP, Decimal("0.00")),
    }
    # Payment amount = $100
    # RETRY_NOW: 0.70 * 100 - 0.15 - 0.05 = 69.80
    # ESCALATE: 0.80 * 100 - 3.50 - 0.00 = 76.50
    decision = engine.evaluate_actions(context, Decimal("100.00"), probabilities)
    assert decision.proposed_action == Action.ESCALATE
    assert decision.best_candidate is not None
    assert decision.best_candidate.action == Action.ESCALATE


def test_economic_engine_defaults_to_stop_on_negative_ev():
    """Verify engine defaults to STOP when all actions produce non-positive EV."""
    engine = DefaultEconomicEngine()
    context = PaymentContext(
        payment_id="pay_ev_2",
        customer_id="cust_ev_2",
        customer_tier=CustomerTier.VIP,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="HARD_DECLINE",
        raw_error_message="Stolen card",
        failure_category=FailureCategory.HARD_DECLINE,
        failure_severity=FailureSeverity.FATAL,
    )
    probabilities = {action: ProbabilityEstimate(action, Decimal("0.00")) for action in Action}

    decision = engine.evaluate_actions(context, Decimal("50.00"), probabilities)
    assert decision.proposed_action == Action.STOP


def test_ev_stop_is_zero_when_probability_is_zero():
    """T5: Verify STOP with probability 0 has gross EV = 0 and net EV = 0."""
    ev_record = CandidateActionEV.calculate(
        action=Action.STOP,
        probability=Decimal("0.00"),
        payment_amount=Decimal("500.00"),
        cost=Decimal("0.00"),
        penalty=Decimal("0.00"),
    )
    assert ev_record.gross_expected_value == Decimal("0.00")
    assert ev_record.net_expected_value == Decimal("0.00")
    assert ev_record.intervention_cost == Decimal("0.00")
    assert ev_record.friction_penalty == Decimal("0.00")


def test_penalty_scales_with_customer_tier():
    """T6: Verify penalty scales by customer tier multipliers (VIP=3.0, STANDARD=1.0)."""
    from recoveriq.config.settings import PenaltyConfig

    penalty_cfg = PenaltyConfig()
    # Base NUDGE penalty = 0.25
    # VIP multiplier = 3.0 -> 0.25 * 3.0 = 0.75
    assert penalty_cfg.get_penalty(Action.NUDGE, CustomerTier.VIP) == Decimal("0.75")
    # STANDARD multiplier = 1.0 -> 0.25 * 1.0 = 0.25
    assert penalty_cfg.get_penalty(Action.NUDGE, CustomerTier.STANDARD) == Decimal("0.25")
    # Base ESCALATE penalty = 0.00 -> 0.00 * 3.0 = 0.00
    assert penalty_cfg.get_penalty(Action.ESCALATE, CustomerTier.VIP) == Decimal("0.00")


def test_economic_engine_all_candidates_populated():
    """T7: Verify candidate evaluations list contains exactly all Action space members (6)."""
    engine = DefaultEconomicEngine()
    context = PaymentContext(
        payment_id="pay_pop_1",
        customer_id="cust_pop_1",
        customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="TIMEOUT",
        raw_error_message="Gateway timeout",
        failure_category=FailureCategory.NETWORK_TIMEOUT,
        failure_severity=FailureSeverity.TRANSIENT,
    )
    probabilities = {action: ProbabilityEstimate(action, Decimal("0.20")) for action in Action}
    decision = engine.evaluate_actions(context, Decimal("100.00"), probabilities)

    assert len(decision.candidate_evaluations) == 6
    evaluated_actions = {c.action for c in decision.candidate_evaluations}
    assert evaluated_actions == set(Action)


def test_economic_engine_rationale_not_empty():
    """T8: Verify decision rationale is a non-empty string explaining action selection."""
    engine = DefaultEconomicEngine()
    context = PaymentContext(
        payment_id="pay_rat_1",
        customer_id="cust_rat_1",
        customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="TIMEOUT",
        raw_error_message="Gateway timeout",
        failure_category=FailureCategory.NETWORK_TIMEOUT,
        failure_severity=FailureSeverity.TRANSIENT,
    )
    probabilities = {
        Action.RETRY_NOW: ProbabilityEstimate(Action.RETRY_NOW, Decimal("0.80")),
        Action.RETRY_LATER: ProbabilityEstimate(Action.RETRY_LATER, Decimal("0.50")),
        Action.SEND_LINK: ProbabilityEstimate(Action.SEND_LINK, Decimal("0.30")),
        Action.NUDGE: ProbabilityEstimate(Action.NUDGE, Decimal("0.20")),
        Action.ESCALATE: ProbabilityEstimate(Action.ESCALATE, Decimal("0.10")),
        Action.STOP: ProbabilityEstimate(Action.STOP, Decimal("0.00")),
    }
    decision = engine.evaluate_actions(context, Decimal("100.00"), probabilities)
    assert isinstance(decision.rationale, str)
    assert len(decision.rationale.strip()) > 0
    assert "RETRY_NOW" in decision.rationale


def test_economic_engine_stop_fallback_on_exactly_zero_ev():
    """T9: Verify proposed action is STOP when best net EV is exactly zero (<= threshold)."""
    engine = DefaultEconomicEngine()
    context = PaymentContext(
        payment_id="pay_zero_1",
        customer_id="cust_zero_1",
        customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="TIMEOUT",
        raw_error_message="Gateway timeout",
        failure_category=FailureCategory.NETWORK_TIMEOUT,
        failure_severity=FailureSeverity.TRANSIENT,
    )
    # Payment amount = 100.00.
    # For RETRY_NOW: cost = 0.15, penalty = 0.05 (STANDARD). Total deduction = 0.20.
    # To get net EV = exactly 0.00: Gross EV = 0.20 => prob = 0.20 / 100.00 = 0.002.
    # Gross EV = Decimal("0.002") * Decimal("100.00") = Decimal("0.200").
    # Net EV = Decimal("0.200") - Decimal("0.15") - Decimal("0.05") = Decimal("0.000").
    # For all other actions: prob = 0.00 -> net EV < 0 (due to positive costs/penalties) or = 0 (STOP).
    probabilities = {
        Action.RETRY_NOW: ProbabilityEstimate(Action.RETRY_NOW, Decimal("0.002")),
        Action.RETRY_LATER: ProbabilityEstimate(Action.RETRY_LATER, Decimal("0.00")),
        Action.SEND_LINK: ProbabilityEstimate(Action.SEND_LINK, Decimal("0.00")),
        Action.NUDGE: ProbabilityEstimate(Action.NUDGE, Decimal("0.00")),
        Action.ESCALATE: ProbabilityEstimate(Action.ESCALATE, Decimal("0.00")),
        Action.STOP: ProbabilityEstimate(Action.STOP, Decimal("0.00")),
    }
    decision = engine.evaluate_actions(context, Decimal("100.00"), probabilities)
    # Top evaluation has net EV of 0.00, which satisfies net_expected_value <= min_ev_threshold (0.00)
    assert decision.candidate_evaluations[0].net_expected_value == Decimal("0.000")
    assert decision.proposed_action == Action.STOP
    assert "defaulting to STOP" in decision.rationale


def test_economic_engine_custom_ev_threshold():
    """T10: Verify custom min_ev_threshold forces STOP if all candidate EVs are below threshold."""
    custom_config = EconomicConfig(min_ev_threshold=Decimal("50.00"))
    engine = DefaultEconomicEngine(config=custom_config)
    context = PaymentContext(
        payment_id="pay_thresh_1",
        customer_id="cust_thresh_1",
        customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="TIMEOUT",
        raw_error_message="Gateway timeout",
        failure_category=FailureCategory.NETWORK_TIMEOUT,
        failure_severity=FailureSeverity.TRANSIENT,
    )
    # V = 100.00.
    # RETRY_NOW: prob = 0.40 -> Gross = 40.00, Net = 40.00 - 0.20 = 39.80 < 50.00.
    # All candidates have net EV < 50.00.
    probabilities = {
        Action.RETRY_NOW: ProbabilityEstimate(Action.RETRY_NOW, Decimal("0.40")),
        Action.RETRY_LATER: ProbabilityEstimate(Action.RETRY_LATER, Decimal("0.30")),
        Action.SEND_LINK: ProbabilityEstimate(Action.SEND_LINK, Decimal("0.20")),
        Action.NUDGE: ProbabilityEstimate(Action.NUDGE, Decimal("0.10")),
        Action.ESCALATE: ProbabilityEstimate(Action.ESCALATE, Decimal("0.10")),
        Action.STOP: ProbabilityEstimate(Action.STOP, Decimal("0.00")),
    }
    decision = engine.evaluate_actions(context, Decimal("100.00"), probabilities)
    assert decision.proposed_action == Action.STOP
    assert decision.candidate_evaluations[0].net_expected_value < Decimal("50.00")
    assert "defaulting to STOP" in decision.rationale


def test_economic_engine_vip_nudge_penalty_affects_selection():
    """T11: Verify cost/penalty awareness causes selection to diverge from greedy probability.
    
    For a VIP customer:
    - NUDGE has highest probability (0.50), but VIP penalty (0.75) + cost (0.10) = 0.85 deduction.
      On a 10.00 payment: Net EV = 5.00 - 0.85 = 4.15.
    - RETRY_LATER has lower probability (0.45), but VIP penalty (0.06) + cost (0.15) = 0.21 deduction.
      On a 10.00 payment: Net EV = 4.50 - 0.21 = 4.29.
    Engine must choose RETRY_LATER over NUDGE due to net EV maximization.
    """
    engine = DefaultEconomicEngine()
    context = PaymentContext(
        payment_id="pay_vip_1",
        customer_id="cust_vip_1",
        customer_tier=CustomerTier.VIP,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="INSUFFICIENT_FUNDS",
        raw_error_message="Low balance",
        failure_category=FailureCategory.INSUFFICIENT_FUNDS,
        failure_severity=FailureSeverity.RECOVERABLE,
    )
    probabilities = {
        Action.NUDGE: ProbabilityEstimate(Action.NUDGE, Decimal("0.50")),
        Action.RETRY_LATER: ProbabilityEstimate(Action.RETRY_LATER, Decimal("0.45")),
        Action.RETRY_NOW: ProbabilityEstimate(Action.RETRY_NOW, Decimal("0.10")),
        Action.SEND_LINK: ProbabilityEstimate(Action.SEND_LINK, Decimal("0.10")),
        Action.ESCALATE: ProbabilityEstimate(Action.ESCALATE, Decimal("0.10")),
        Action.STOP: ProbabilityEstimate(Action.STOP, Decimal("0.00")),
    }
    decision = engine.evaluate_actions(context, Decimal("10.00"), probabilities)
    assert decision.proposed_action == Action.RETRY_LATER
    # Confirm NUDGE had higher probability
    nudge_eval = next(c for c in decision.candidate_evaluations if c.action == Action.NUDGE)
    retry_eval = next(c for c in decision.candidate_evaluations if c.action == Action.RETRY_LATER)
    assert nudge_eval.estimated_probability > retry_eval.estimated_probability
    assert retry_eval.net_expected_value > nudge_eval.net_expected_value


def test_economic_engine_missing_probability_defaults_to_zero():
    """T12: Verify missing actions in probability map default gracefully to probability 0.0."""
    engine = DefaultEconomicEngine()
    context = PaymentContext(
        payment_id="pay_miss_1",
        customer_id="cust_miss_1",
        customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="TIMEOUT",
        raw_error_message="Gateway timeout",
        failure_category=FailureCategory.NETWORK_TIMEOUT,
        failure_severity=FailureSeverity.TRANSIENT,
    )
    # Partial map missing NUDGE, ESCALATE, SEND_LINK
    partial_probabilities = {
        Action.RETRY_NOW: ProbabilityEstimate(Action.RETRY_NOW, Decimal("0.70")),
        Action.RETRY_LATER: ProbabilityEstimate(Action.RETRY_LATER, Decimal("0.40")),
        Action.STOP: ProbabilityEstimate(Action.STOP, Decimal("0.00")),
    }
    decision = engine.evaluate_actions(context, Decimal("100.00"), partial_probabilities)
    assert decision.proposed_action == Action.RETRY_NOW
    # Check default assigned to missing action
    missing_eval = next(c for c in decision.candidate_evaluations if c.action == Action.NUDGE)
    assert missing_eval.estimated_probability == Decimal("0.0")


def test_economic_engine_best_candidate_matches_proposed_action():
    """T13: Verify decision.best_candidate corresponds to decision.proposed_action."""
    engine = DefaultEconomicEngine()
    context = PaymentContext(
        payment_id="pay_match_1",
        customer_id="cust_match_1",
        customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="TIMEOUT",
        raw_error_message="Gateway timeout",
        failure_category=FailureCategory.NETWORK_TIMEOUT,
        failure_severity=FailureSeverity.TRANSIENT,
    )
    probabilities = {
        Action.RETRY_NOW: ProbabilityEstimate(Action.RETRY_NOW, Decimal("0.85")),
        Action.RETRY_LATER: ProbabilityEstimate(Action.RETRY_LATER, Decimal("0.50")),
        Action.SEND_LINK: ProbabilityEstimate(Action.SEND_LINK, Decimal("0.20")),
        Action.NUDGE: ProbabilityEstimate(Action.NUDGE, Decimal("0.10")),
        Action.ESCALATE: ProbabilityEstimate(Action.ESCALATE, Decimal("0.05")),
        Action.STOP: ProbabilityEstimate(Action.STOP, Decimal("0.00")),
    }
    decision = engine.evaluate_actions(context, Decimal("100.00"), probabilities)
    assert decision.proposed_action == Action.RETRY_NOW
    assert decision.best_candidate is not None
    assert decision.best_candidate.action == decision.proposed_action
    assert decision.best_candidate.net_expected_value == decision.candidate_evaluations[0].net_expected_value


def test_economic_engine_satisfies_protocol():
    """T14: Verify DefaultEconomicEngine runtime satisfies the EconomicEngine Protocol."""
    from recoveriq.economics.engine import EconomicEngine

    engine = DefaultEconomicEngine()
    assert isinstance(engine, EconomicEngine)


def test_candidate_action_ev_zero_probability_is_valid():
    """T15: Verify Decimal('0.00') is within valid probability bounds [0.0, 1.0]."""
    ev_record = CandidateActionEV.calculate(
        action=Action.STOP,
        probability=Decimal("0.00"),
        payment_amount=Decimal("100.00"),
        cost=Decimal("0.00"),
        penalty=Decimal("0.00"),
    )
    assert ev_record.estimated_probability == Decimal("0.00")
    assert ev_record.gross_expected_value == Decimal("0.00")
    assert ev_record.net_expected_value == Decimal("0.00")


def test_candidate_action_ev_unit_probability():
    """T16: Verify Decimal('1.00') produces gross expected value equal to full payment amount."""
    ev_record = CandidateActionEV.calculate(
        action=Action.RETRY_NOW,
        probability=Decimal("1.00"),
        payment_amount=Decimal("250.00"),
        cost=Decimal("0.15"),
        penalty=Decimal("0.05"),
    )
    assert ev_record.estimated_probability == Decimal("1.00")
    assert ev_record.gross_expected_value == Decimal("250.00")
    assert ev_record.net_expected_value == Decimal("249.80")

