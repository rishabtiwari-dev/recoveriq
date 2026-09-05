"""Sprint 11 Tests — Controlled Sequential Policy & Escalation Valuation Ablation."""

from datetime import datetime, timezone
from decimal import Decimal
import pytest

from recoveriq.domain.actions import Action
from recoveriq.domain.models import CustomerTier, FailureCategory, FailureSeverity, PaymentContext, PaymentMethod
from recoveriq.domain.state import PaymentState
from recoveriq.evaluation.sequential_policy import (
    TieredRecoverIQStrategy,
    calculate_human_ops_valuation,
)
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


class MockHighEscalateModel(RecoveryProbabilityModel):
    """Mock probability model where ESCALATE always has highest probability."""

    def estimate_probabilities(self, context: PaymentContext):
        return {
            Action.RETRY_NOW: ProbabilityEstimate(Action.RETRY_NOW, Decimal("0.45")),
            Action.RETRY_LATER: ProbabilityEstimate(Action.RETRY_LATER, Decimal("0.40")),
            Action.SEND_LINK: ProbabilityEstimate(Action.SEND_LINK, Decimal("0.35")),
            Action.NUDGE: ProbabilityEstimate(Action.NUDGE, Decimal("0.20")),
            Action.ESCALATE: ProbabilityEstimate(Action.ESCALATE, Decimal("0.85")),
            Action.STOP: ProbabilityEstimate(Action.STOP, Decimal("0.00")),
        }


@pytest.fixture
def sample_payment_record():
    return SyntheticPaymentRecord(
        payment_id="pay_s11_001",
        customer_id="cust_001",
        amount=Decimal("1000.00"),
        currency="INR",
        failure_category=FailureCategory.INSUFFICIENT_FUNDS,
        failure_severity=FailureSeverity.RECOVERABLE,
        customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="NSF",
        raw_error_message="Insufficient funds",
        failure_timestamp=datetime.now(timezone.utc),
        attempt_count=1,
    )


def test_early_escalation_is_blocked(sample_payment_record):
    """Test 1: Verify Attempt 1 and Attempt 2 cannot result in proposed ESCALATE for TieredRecoverIQStrategy."""
    model = MockHighEscalateModel()
    strat = TieredRecoverIQStrategy(probability_model=model, max_attempts=3)

    # Attempt 1
    ctx1 = PaymentContext(
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
    act1 = strat.propose_action(sample_payment_record, ctx1)
    assert act1 != Action.ESCALATE
    # Highest among automated actions is RETRY_NOW (P=0.45)
    assert act1 == Action.RETRY_NOW

    # Attempt 2
    ctx2 = PaymentContext(
        payment_id=sample_payment_record.payment_id,
        customer_id=sample_payment_record.customer_id,
        customer_tier=sample_payment_record.customer_tier,
        payment_method=sample_payment_record.payment_method,
        raw_error_code=sample_payment_record.raw_error_code,
        raw_error_message=sample_payment_record.raw_error_message,
        failure_category=sample_payment_record.failure_category,
        failure_severity=sample_payment_record.failure_severity,
        attempt_count=2,
    )
    act2 = strat.propose_action(sample_payment_record, ctx2)
    assert act2 != Action.ESCALATE
    assert act2 == Action.RETRY_NOW


def test_final_escalation_is_permitted(sample_payment_record):
    """Test 2: Verify Attempt 3 (final attempt) can select ESCALATE when it has highest EV."""
    model = MockHighEscalateModel()
    strat = TieredRecoverIQStrategy(probability_model=model, max_attempts=3)

    # Attempt 3 (Final attempt)
    ctx3 = PaymentContext(
        payment_id=sample_payment_record.payment_id,
        customer_id=sample_payment_record.customer_id,
        customer_tier=sample_payment_record.customer_tier,
        payment_method=sample_payment_record.payment_method,
        raw_error_code=sample_payment_record.raw_error_code,
        raw_error_message=sample_payment_record.raw_error_message,
        failure_category=sample_payment_record.failure_category,
        failure_severity=sample_payment_record.failure_severity,
        attempt_count=3,
    )
    act3 = strat.propose_action(sample_payment_record, ctx3)
    assert act3 == Action.ESCALATE


def test_existing_recoveriq_remains_unchanged(sample_payment_record):
    """Test 3: Verify the original RecoverIQStrategy still proposes ESCALATE on attempt 1."""
    model = MockHighEscalateModel()
    orig_strat = RecoverIQStrategy(probability_model=model)

    ctx1 = PaymentContext(
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
    act1 = orig_strat.propose_action(sample_payment_record, ctx1)
    # Original strategy must unconstrainedly select ESCALATE on attempt 1
    assert act1 == Action.ESCALATE


def test_policy_gate_still_applies(sample_payment_record):
    """Test 4: Verify Tiered strategy actions go through the normal Policy Gate and get clamped on hard decline."""
    hard_rec = SyntheticPaymentRecord(
        payment_id="pay_hard_01",
        customer_id="cust_001",
        amount=Decimal("500.00"),
        currency="INR",
        failure_category=FailureCategory.HARD_DECLINE,
        failure_severity=FailureSeverity.FATAL,
        customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="FRAUD",
        raw_error_message="Suspected fraud",
        failure_timestamp=datetime.now(timezone.utc),
        attempt_count=1,
    )
    # Model proposing RETRY_NOW on attempt 1
    model = MockHighEscalateModel()
    strat = TieredRecoverIQStrategy(probability_model=model, max_attempts=3)

    gt = GroundTruthRecord(
        payment_id=hard_rec.payment_id,
        latent_recoverability_profile=RecoverabilityProfile.VERY_LOW,
        action_base_probabilities={a: 0.0 for a in Action},
    )
    env = SimulationEnvironment([gt], seed=42)
    runner = TrajectoryEvaluationRunner(max_attempts=3)

    episode = runner.evaluate_episode(hard_rec, strat, env)
    assert episode.terminal_state == PaymentState.FAILED_TERMINAL
    assert episode.steps[0].authorized_action == Action.STOP
    assert episode.steps[0].is_authorized is False
    assert "hard decline" in episode.steps[0].rejection_reason.lower()


def test_human_ops_valuation_calculation():
    """Test 5: Verify expected_human_value = P_escalate * payment_amount for escalated payments."""
    gt1 = GroundTruthRecord("p1", RecoverabilityProfile.HIGH, {Action.ESCALATE: 0.75})
    gt2 = GroundTruthRecord("p2", RecoverabilityProfile.LOW, {Action.ESCALATE: 0.25})

    episodes = [
        # Episode 1: Escalated, amount = 1000.00 -> expected = 0.75 * 1000 = 750.00
        TrajectoryEpisode(
            payment_id="p1",
            steps=[TrajectoryStep(1, Action.ESCALATE, Action.ESCALATE, True, False, Decimal("3.50"), Decimal("0.00"), PaymentState.ESCALATED)],
            terminal_state=PaymentState.ESCALATED,
            final_recovered=False,
            payment_amount=Decimal("1000.00"),
            total_cost=Decimal("3.50"),
            total_penalty=Decimal("0.00"),
            net_recovered_value=Decimal("-3.50"),
            attempt_count=1,
        ),
        # Episode 2: Recovered automated, amount = 500.00 -> human value = 0
        TrajectoryEpisode(
            payment_id="p2",
            steps=[TrajectoryStep(1, Action.RETRY_NOW, Action.RETRY_NOW, True, True, Decimal("0.15"), Decimal("0.05"), PaymentState.RECOVERED)],
            terminal_state=PaymentState.RECOVERED,
            final_recovered=True,
            payment_amount=Decimal("500.00"),
            total_cost=Decimal("0.15"),
            total_penalty=Decimal("0.05"),
            net_recovered_value=Decimal("499.80"),
            attempt_count=1,
        ),
    ]

    auto_metrics = TrajectoryStrategyMetrics.compute("TestStrat", seed=42, episodes=episodes, max_attempts=3)
    val_record = calculate_human_ops_valuation(episodes, [gt1, gt2], auto_metrics)

    assert val_record.n_escalated == 1
    assert val_record.escalation_rate == 0.5
    assert val_record.expected_human_ops_value == Decimal("750.00")
    # Automated NRV = -3.50 + 499.80 = 496.30
    assert val_record.automated_nrv == Decimal("496.30")
    # Full system NRV = 496.30 + 750.00 = 1246.30
    assert val_record.full_system_expected_nrv == Decimal("1246.30")


def test_automated_accounting_is_unchanged():
    """Test 6: Verify an escalated trajectory remains terminal_state == ESCALATED and final_recovered == False."""
    gt = GroundTruthRecord("p_esc", RecoverabilityProfile.HIGH, {Action.ESCALATE: 0.80})
    rec = SyntheticPaymentRecord(
        payment_id="p_esc",
        customer_id="c1",
        amount=Decimal("200.00"),
        currency="INR",
        failure_category=FailureCategory.NETWORK_TIMEOUT,
        failure_severity=FailureSeverity.TRANSIENT,
        customer_tier=CustomerTier.VIP,
        payment_method=PaymentMethod.UPI,
        raw_error_code="TIMEOUT",
        raw_error_message="Timeout",
        failure_timestamp=datetime.now(timezone.utc),
        attempt_count=1,
    )
    model = MockHighEscalateModel()
    strat = RecoverIQStrategy(probability_model=model)
    env = SimulationEnvironment([gt], seed=42)
    runner = TrajectoryEvaluationRunner(max_attempts=3)

    ep = runner.evaluate_episode(rec, strat, env)

    assert ep.terminal_state == PaymentState.ESCALATED
    assert ep.final_recovered is False
    assert ep.net_recovered_value == Decimal("-3.50")


def test_tiered_strategy_determinism(sample_payment_record):
    """Test 7: Verify running Tiered strategy twice with same seed produces 100% identical outputs."""
    model = MockHighEscalateModel()
    strat = TieredRecoverIQStrategy(probability_model=model, max_attempts=3)

    gt = GroundTruthRecord(
        sample_payment_record.payment_id,
        RecoverabilityProfile.MEDIUM,
        {Action.RETRY_LATER: 0.50, Action.ESCALATE: 0.80},
    )

    env1 = SimulationEnvironment([gt], seed=999)
    runner1 = TrajectoryEvaluationRunner(max_attempts=3)
    ep1 = runner1.evaluate_episode(sample_payment_record, strat, env1)

    env2 = SimulationEnvironment([gt], seed=999)
    runner2 = TrajectoryEvaluationRunner(max_attempts=3)
    ep2 = runner2.evaluate_episode(sample_payment_record, strat, env2)

    assert ep1.final_recovered == ep2.final_recovered
    assert ep1.terminal_state == ep2.terminal_state
    assert ep1.attempt_count == ep2.attempt_count
    assert ep1.net_recovered_value == ep2.net_recovered_value
    assert len(ep1.steps) == len(ep2.steps)


def test_tiered_strategy_allows_retry_continuation(sample_payment_record):
    """Test 8: Verify Tiered strategy attempts retry on step 1 and step 2 before escalating on step 3 if unrecovered."""
    # World model: 0% recovery on automated retries, 80% on escalate
    gt = GroundTruthRecord(
        sample_payment_record.payment_id,
        RecoverabilityProfile.MEDIUM,
        {
            Action.RETRY_LATER: 0.0,
            Action.RETRY_NOW: 0.0,
            Action.SEND_LINK: 0.0,
            Action.NUDGE: 0.0,
            Action.ESCALATE: 0.80,
            Action.STOP: 0.0,
        },
    )
    env = SimulationEnvironment([gt], seed=42)
    model = MockHighEscalateModel()
    strat = TieredRecoverIQStrategy(probability_model=model, max_attempts=3)
    runner = TrajectoryEvaluationRunner(max_attempts=3)

    ep = runner.evaluate_episode(sample_payment_record, strat, env)

    # Must have executed 3 steps: Step 1 = RETRY_NOW, Step 2 = RETRY_NOW, Step 3 = ESCALATE
    assert ep.attempt_count == 3
    assert len(ep.steps) == 3
    assert ep.steps[0].authorized_action == Action.RETRY_NOW
    assert ep.steps[1].authorized_action == Action.RETRY_NOW
    assert ep.steps[2].authorized_action == Action.ESCALATE
    assert ep.terminal_state == PaymentState.ESCALATED
    assert ep.final_recovered is False
