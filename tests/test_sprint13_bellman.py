"""Sprint 13 Tests — Bellman Option Value & Native Sequential Economic Policy."""

from datetime import datetime, timezone
from decimal import Decimal
import pytest

from recoveriq.config.settings import ActionCostConfig, EconomicConfig, PenaltyConfig
from recoveriq.domain.actions import Action
from recoveriq.domain.models import CustomerTier, FailureCategory, FailureSeverity, PaymentContext, PaymentMethod
from recoveriq.domain.state import PaymentState
from recoveriq.evaluation.bellman_policy import (
    BellmanActionEvaluation,
    BellmanDecision,
    BellmanRecoverIQStrategy,
)
from recoveriq.evaluation.robustness import calculate_human_ops_valuation_sweep
from recoveriq.evaluation.sequential_policy import TieredRecoverIQStrategy
from recoveriq.evaluation.strategies import RecoverIQStrategy
from recoveriq.evaluation.trajectory import (
    TrajectoryEpisode,
    TrajectoryEvaluationRunner,
    TrajectoryStep,
    TrajectoryStrategyMetrics,
)
from recoveriq.model.probability import ProbabilityEstimate, RecoveryProbabilityModel
from recoveriq.policy.gate import InvariantPolicyGate
from recoveriq.simulation.environment import SimulationEnvironment
from recoveriq.simulation.schema import (
    GroundTruthRecord,
    RecoverabilityProfile,
    SyntheticPaymentRecord,
)


class MockRecoverIQModel(RecoveryProbabilityModel):
    """Mock probability model where ESCALATE has highest single-step probability, but RETRY_LATER is strong."""

    def __init__(self, p_esc="0.55", p_retry_later="0.45", p_retry_now="0.30"):
        self.p_esc = Decimal(p_esc)
        self.p_retry_later = Decimal(p_retry_later)
        self.p_retry_now = Decimal(p_retry_now)

    def estimate_probabilities(self, context: PaymentContext):
        return {
            Action.RETRY_NOW: ProbabilityEstimate(Action.RETRY_NOW, self.p_retry_now),
            Action.RETRY_LATER: ProbabilityEstimate(Action.RETRY_LATER, self.p_retry_later),
            Action.SEND_LINK: ProbabilityEstimate(Action.SEND_LINK, Decimal("0.25")),
            Action.NUDGE: ProbabilityEstimate(Action.NUDGE, Decimal("0.20")),
            Action.ESCALATE: ProbabilityEstimate(Action.ESCALATE, self.p_esc),
            Action.STOP: ProbabilityEstimate(Action.STOP, Decimal("0.00")),
        }


@pytest.fixture
def sample_payment_record():
    return SyntheticPaymentRecord(
        payment_id="pay_b13_001",
        customer_id="cust_001",
        amount=Decimal("1000.00"),
        currency="INR",
        failure_category=FailureCategory.INSUFFICIENT_FUNDS,
        failure_severity=FailureSeverity.RECOVERABLE,
        customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.UPI,
        raw_error_code="NSF",
        raw_error_message="Insufficient funds",
        failure_timestamp=datetime.now(timezone.utc),
        attempt_count=1,
    )


def test_bellman_strategy_determinism(sample_payment_record):
    """Test 1: Verify Bellman strategy produces 100% deterministic evaluations given identical inputs."""
    model = MockRecoverIQModel()
    strat = BellmanRecoverIQStrategy(probability_model=model, max_attempts=3)

    ctx = PaymentContext(
        payment_id=sample_payment_record.payment_id,
        customer_id=sample_payment_record.customer_id,
        customer_tier=sample_payment_record.customer_tier,
        payment_method=sample_payment_record.payment_method,
        raw_error_code=sample_payment_record.raw_error_code,
        raw_error_message=sample_payment_record.raw_error_message,
        failure_category=sample_payment_record.failure_category,
        failure_severity=sample_payment_record.failure_severity,
        attempt_count=1,
    )

    evals1 = strat.evaluate_q_values(sample_payment_record, ctx, 1, 3)
    evals2 = strat.evaluate_q_values(sample_payment_record, ctx, 1, 3)

    assert len(evals1) == len(evals2)
    for e1, e2 in zip(evals1, evals2):
        assert e1.action == e2.action
        assert e1.immediate_ev == e2.immediate_ev
        assert e1.future_option_value == e2.future_option_value
        assert e1.total_q_value == e2.total_q_value


def test_terminal_horizon_behavior(sample_payment_record):
    """Test 2: At attempt == max_attempts (final attempt), future option value is exactly 0 for ALL actions."""
    model = MockRecoverIQModel()
    strat = BellmanRecoverIQStrategy(probability_model=model, max_attempts=3)

    ctx = PaymentContext(
        payment_id=sample_payment_record.payment_id,
        customer_id=sample_payment_record.customer_id,
        customer_tier=sample_payment_record.customer_tier,
        payment_method=sample_payment_record.payment_method,
        raw_error_code=sample_payment_record.raw_error_code,
        raw_error_message=sample_payment_record.raw_error_message,
        failure_category=sample_payment_record.failure_category,
        failure_severity=sample_payment_record.failure_severity,
        attempt_count=3,  # Final attempt
    )

    evals = strat.evaluate_q_values(sample_payment_record, ctx, current_attempt=3, effective_horizon=3)
    for ev in evals:
        assert ev.future_option_value == Decimal("0.00")
        assert ev.option_value == Decimal("0.00")
        assert ev.total_q_value == ev.immediate_ev


def test_future_value_is_zero_after_terminal_actions(sample_payment_record):
    """Test 3: Terminal actions (STOP and ESCALATE) must have exactly 0.00 automated future value even at attempt 1."""
    model = MockRecoverIQModel()
    strat = BellmanRecoverIQStrategy(probability_model=model, max_attempts=3)

    ctx = PaymentContext(
        payment_id=sample_payment_record.payment_id,
        customer_id=sample_payment_record.customer_id,
        customer_tier=sample_payment_record.customer_tier,
        payment_method=sample_payment_record.payment_method,
        raw_error_code=sample_payment_record.raw_error_code,
        raw_error_message=sample_payment_record.raw_error_message,
        failure_category=sample_payment_record.failure_category,
        failure_severity=sample_payment_record.failure_severity,
        attempt_count=1,
    )

    evals = strat.evaluate_q_values(sample_payment_record, ctx, current_attempt=1, effective_horizon=3)
    ev_map = {e.action: e for e in evals}

    assert ev_map[Action.STOP].future_option_value == Decimal("0.00")
    assert ev_map[Action.ESCALATE].future_option_value == Decimal("0.00")
    # Non-terminal retry action has positive option value
    assert ev_map[Action.RETRY_LATER].future_option_value > Decimal("0.00")


def test_bellman_policy_does_not_mutate_recoveriq_strategy(sample_payment_record):
    """Test 4: Verify original RecoverIQStrategy is untouched and behaves purely myopically."""
    model = MockRecoverIQModel()
    orig_strat = RecoverIQStrategy(probability_model=model)

    ctx = PaymentContext(
        payment_id=sample_payment_record.payment_id,
        customer_id=sample_payment_record.customer_id,
        customer_tier=sample_payment_record.customer_tier,
        payment_method=sample_payment_record.payment_method,
        raw_error_code=sample_payment_record.raw_error_code,
        raw_error_message=sample_payment_record.raw_error_message,
        failure_category=sample_payment_record.failure_category,
        failure_severity=sample_payment_record.failure_severity,
        attempt_count=1,
    )

    # Myopic strategy chooses ESCALATE because EV(ESCALATE) = 0.55*1000 - 3.50 = 546.50 > EV(RETRY_LATER) = 449.83
    act = orig_strat.propose_action(sample_payment_record, ctx)
    assert act == Action.ESCALATE


def test_policy_gate_remains_authoritative(sample_payment_record):
    """Test 5: Policy gate intercepts and clamps Bellman actions on hard declines."""
    hard_rec = SyntheticPaymentRecord(
        payment_id="pay_hard_01",
        customer_id="cust_001",
        amount=Decimal("500.00"),
        currency="INR",
        failure_category=FailureCategory.HARD_DECLINE,
        failure_severity=FailureSeverity.FATAL,
        customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="STOLEN",
        raw_error_message="Card stolen",
        failure_timestamp=datetime.now(timezone.utc),
        attempt_count=1,
    )
    model = MockRecoverIQModel()
    strat = BellmanRecoverIQStrategy(probability_model=model, max_attempts=3)

    gt = GroundTruthRecord(hard_rec.payment_id, RecoverabilityProfile.VERY_LOW, {a: 0.0 for a in Action})
    env = SimulationEnvironment([gt], seed=42)
    runner = TrajectoryEvaluationRunner(max_attempts=3)

    ep = runner.evaluate_episode(hard_rec, strat, env)
    assert ep.terminal_state == PaymentState.FAILED_TERMINAL
    assert ep.steps[0].authorized_action == Action.STOP
    assert ep.steps[0].is_authorized is False


def test_horizon_1_collapses_to_myopic_evaluation(sample_payment_record):
    """Test 6: With planning_horizon=1 or enable_future_value=False, Bellman Q-values equal immediate EV."""
    model = MockRecoverIQModel()
    strat_h1 = BellmanRecoverIQStrategy(probability_model=model, max_attempts=3, planning_horizon=1)
    strat_no_fut = BellmanRecoverIQStrategy(probability_model=model, max_attempts=3, enable_future_value=False)

    ctx = PaymentContext(
        payment_id=sample_payment_record.payment_id,
        customer_id=sample_payment_record.customer_id,
        customer_tier=sample_payment_record.customer_tier,
        payment_method=sample_payment_record.payment_method,
        raw_error_code=sample_payment_record.raw_error_code,
        raw_error_message=sample_payment_record.raw_error_message,
        failure_category=sample_payment_record.failure_category,
        failure_severity=sample_payment_record.failure_severity,
        attempt_count=1,
    )

    evals_h1 = strat_h1.evaluate_q_values(sample_payment_record, ctx, 1, 1)
    evals_no_fut = strat_no_fut.evaluate_q_values(sample_payment_record, ctx, 1, 3)

    for ev1, ev2 in zip(evals_h1, evals_no_fut):
        assert ev1.future_option_value == Decimal("0.00")
        assert ev2.future_option_value == Decimal("0.00")
        assert ev1.total_q_value == ev1.immediate_ev
        assert ev2.total_q_value == ev2.immediate_ev

    # Under Horizon 1, ESCALATE has highest immediate EV, so it is proposed
    assert strat_h1.propose_action(sample_payment_record, ctx) == Action.ESCALATE
    assert strat_no_fut.propose_action(sample_payment_record, ctx) == Action.ESCALATE


def test_horizon_3_preserves_retry_option_value(sample_payment_record):
    """Test 7: Under Horizon 3, future option value lifts RETRY_LATER Q-value above ESCALATE."""
    model = MockRecoverIQModel()
    strat = BellmanRecoverIQStrategy(probability_model=model, max_attempts=3, planning_horizon=3)

    ctx = PaymentContext(
        payment_id=sample_payment_record.payment_id,
        customer_id=sample_payment_record.customer_id,
        customer_tier=sample_payment_record.customer_tier,
        payment_method=sample_payment_record.payment_method,
        raw_error_code=sample_payment_record.raw_error_code,
        raw_error_message=sample_payment_record.raw_error_message,
        failure_category=sample_payment_record.failure_category,
        failure_severity=sample_payment_record.failure_severity,
        attempt_count=1,
    )

    evals = strat.evaluate_q_values(sample_payment_record, ctx, 1, 3)
    ev_map = {e.action: e for e in evals}

    # Immediate EV: ESCALATE (546.50) > RETRY_LATER (449.83)
    assert ev_map[Action.ESCALATE].immediate_ev > ev_map[Action.RETRY_LATER].immediate_ev

    # Future Option Value: RETRY_LATER > 0, ESCALATE == 0
    assert ev_map[Action.RETRY_LATER].future_option_value > Decimal("0.00")
    assert ev_map[Action.ESCALATE].future_option_value == Decimal("0.00")

    # Total Q-value: RETRY_LATER Q > ESCALATE Q
    # Q(RETRY_LATER) = 449.83 + (1 - 0.45) * J_2 ≈ 449.83 + 0.55 * 546.50 = 449.83 + 300.57 = 750.40 > 546.50
    assert ev_map[Action.RETRY_LATER].total_q_value > ev_map[Action.ESCALATE].total_q_value

    # Native Bellman selects RETRY_LATER without any hardcoded rule
    proposed = strat.propose_action(sample_payment_record, ctx)
    assert proposed == Action.RETRY_LATER


def test_immediate_ev_and_future_option_value_separately_observable(sample_payment_record):
    """Test 8: Verify BellmanActionEvaluation cleanly exposes immediate_ev, future_option_value, and total_q_value."""
    model = MockRecoverIQModel()
    strat = BellmanRecoverIQStrategy(probability_model=model, max_attempts=3)

    ctx = PaymentContext(
        payment_id=sample_payment_record.payment_id,
        customer_id=sample_payment_record.customer_id,
        customer_tier=sample_payment_record.customer_tier,
        payment_method=sample_payment_record.payment_method,
        raw_error_code=sample_payment_record.raw_error_code,
        raw_error_message=sample_payment_record.raw_error_message,
        failure_category=sample_payment_record.failure_category,
        failure_severity=sample_payment_record.failure_severity,
        attempt_count=1,
    )

    strat.propose_action(sample_payment_record, ctx)
    dec = strat.last_decision
    assert dec is not None
    sel = dec.selected_evaluation

    assert sel.total_q_value == sel.immediate_ev + sel.future_option_value
    assert sel.option_value == sel.future_option_value


def test_bellman_continuation_in_trajectory(sample_payment_record):
    """Test 9: Verify Bellman strategy executes multi-step trajectories in TrajectoryEvaluationRunner."""
    # Simulation: 0% recovery on retries, 80% on escalate
    gt = GroundTruthRecord(
        sample_payment_record.payment_id,
        RecoverabilityProfile.MEDIUM,
        {
            Action.RETRY_NOW: 0.0,
            Action.RETRY_LATER: 0.0,
            Action.SEND_LINK: 0.0,
            Action.NUDGE: 0.0,
            Action.ESCALATE: 0.80,
            Action.STOP: 0.0,
        },
    )
    env = SimulationEnvironment([gt], seed=42)
    # Model with RETRY_NOW highest among automated so it doesn't wait on cooldown
    model = MockRecoverIQModel(p_esc="0.60", p_retry_later="0.30", p_retry_now="0.45")
    strat = BellmanRecoverIQStrategy(probability_model=model, max_attempts=3)
    runner = TrajectoryEvaluationRunner(max_attempts=3)

    ep = runner.evaluate_episode(sample_payment_record, strat, env)

    # Attempt 1: RETRY_NOW (option value keeps it alive)
    # Attempt 2: RETRY_NOW (option value keeps it alive)
    # Attempt 3: ESCALATE (final attempt, no future option value, ESCALATE has highest immediate EV)
    assert ep.attempt_count == 3
    assert ep.steps[0].authorized_action == Action.RETRY_NOW
    assert ep.steps[1].authorized_action == Action.RETRY_NOW
    assert ep.steps[2].authorized_action == Action.ESCALATE
    assert ep.terminal_state == PaymentState.ESCALATED
