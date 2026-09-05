"""Sprint 10 Tests — Sequential Recovery Trajectory Evaluation."""

from datetime import datetime, timezone
from decimal import Decimal
import pytest

from recoveriq.config.settings import ActionCostConfig, PenaltyConfig, PolicyConfig
from recoveriq.domain.actions import Action
from recoveriq.domain.models import CustomerTier, FailureCategory, FailureSeverity, PaymentMethod
from recoveriq.domain.state import PaymentState
from recoveriq.evaluation.strategies import FixedRetryStrategy, RecoverIQStrategy, RuleBasedStrategy
from recoveriq.evaluation.trajectory import (
    AlwaysStopStrategy,
    TrajectoryEpisode,
    TrajectoryEvaluationRunner,
    TrajectoryStep,
    TrajectoryStrategyMetrics,
)
from recoveriq.model.trainer import ModelTrainer
from recoveriq.policy.gate import InvariantPolicyGate
from recoveriq.simulation.config import SimulationConfig
from recoveriq.simulation.environment import SimulationEnvironment
from recoveriq.simulation.generator import SyntheticPaymentGenerator
from recoveriq.simulation.partitioner import partition_dataset
from recoveriq.simulation.schema import (
    GroundTruthRecord,
    RecoverabilityProfile,
    SyntheticPaymentRecord,
)


@pytest.fixture
def mock_payment_record():
    """Create a standard synthetic payment record for unit testing trajectories."""
    return SyntheticPaymentRecord(
        payment_id="pay_traj_001",
        customer_id="cust_001",
        amount=Decimal("100.00"),
        currency="INR",
        failure_category=FailureCategory.NETWORK_TIMEOUT,
        failure_severity=FailureSeverity.TRANSIENT,
        customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="TIMEOUT",
        raw_error_message="Gateway timeout",
        failure_timestamp=datetime.now(timezone.utc),
        attempt_count=1,
    )


@pytest.fixture
def mock_hard_decline_record():
    """Create a hard decline payment record."""
    return SyntheticPaymentRecord(
        payment_id="pay_traj_hard",
        customer_id="cust_hard",
        amount=Decimal("200.00"),
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


def test_trajectory_terminates_on_first_success(mock_payment_record):
    """Test 1: Verify trajectory stops immediately upon successful recovery with no redundant attempts."""
    # Ground truth: 100% recovery on RETRY_NOW
    gt = GroundTruthRecord(
        payment_id=mock_payment_record.payment_id,
        latent_recoverability_profile=RecoverabilityProfile.VERY_HIGH,
        action_base_probabilities={a: 1.0 for a in Action},
    )
    env = SimulationEnvironment([gt], seed=42)
    runner = TrajectoryEvaluationRunner(max_attempts=3)
    strategy = FixedRetryStrategy()

    episode = runner.evaluate_episode(mock_payment_record, strategy, env)

    assert episode.final_recovered is True
    assert episode.terminal_state == PaymentState.RECOVERED
    assert episode.attempt_count == 1
    assert len(episode.steps) == 1
    assert episode.steps[0].recovered is True
    assert episode.steps[0].resulting_state == PaymentState.RECOVERED
    assert episode.net_recovered_value == Decimal("100.00") - Decimal("0.15") - Decimal("0.05")


def test_trajectory_respects_max_retry_budget(mock_payment_record):
    """Test 2: Verify unrecoverable payment executes exactly max_attempts (3) then terminates FAILED_TERMINAL."""
    # Ground truth: 0% recovery on all actions
    gt = GroundTruthRecord(
        payment_id=mock_payment_record.payment_id,
        latent_recoverability_profile=RecoverabilityProfile.VERY_LOW,
        action_base_probabilities={a: 0.0 for a in Action},
    )
    env = SimulationEnvironment([gt], seed=42)
    runner = TrajectoryEvaluationRunner(max_attempts=3)
    strategy = FixedRetryStrategy()

    episode = runner.evaluate_episode(mock_payment_record, strategy, env)

    assert episode.final_recovered is False
    assert episode.terminal_state == PaymentState.FAILED_TERMINAL
    assert episode.attempt_count == 3
    assert len(episode.steps) == 3
    for step in episode.steps:
        assert step.recovered is False

    # Check that costs accumulated across all 3 steps: 3 * 0.15 = 0.45 cost, 3 * 0.05 = 0.15 penalty
    assert episode.total_cost == Decimal("0.45")
    assert episode.total_penalty == Decimal("0.15")
    assert episode.net_recovered_value == Decimal("-0.60")


def test_trajectory_stop_action_terminates_immediately(mock_payment_record):
    """Test 3: Verify STOP action terminates trajectory immediately at step 1 without further steps."""
    gt = GroundTruthRecord(
        payment_id=mock_payment_record.payment_id,
        latent_recoverability_profile=RecoverabilityProfile.MEDIUM,
        action_base_probabilities={a: 0.5 for a in Action},
    )
    env = SimulationEnvironment([gt], seed=42)
    runner = TrajectoryEvaluationRunner(max_attempts=3)
    strategy = AlwaysStopStrategy()

    episode = runner.evaluate_episode(mock_payment_record, strategy, env)

    assert episode.final_recovered is False
    assert episode.terminal_state == PaymentState.FAILED_TERMINAL
    assert episode.attempt_count == 1
    assert len(episode.steps) == 1
    assert episode.steps[0].authorized_action == Action.STOP
    assert episode.total_cost == Decimal("0.00")
    assert episode.total_penalty == Decimal("0.00")
    assert episode.net_recovered_value == Decimal("0.00")


def test_trajectory_escalate_terminates_as_escalated(mock_payment_record):
    """Test 4: Verify ESCALATE action immediately terminates trajectory with PaymentState.ESCALATED."""
    class MockEscalateStrategy:
        name = "Mock-Escalate"
        def propose_action(self, record, context):
            return Action.ESCALATE

    gt = GroundTruthRecord(
        payment_id=mock_payment_record.payment_id,
        latent_recoverability_profile=RecoverabilityProfile.HIGH,
        action_base_probabilities={a: 0.8 for a in Action},
    )
    env = SimulationEnvironment([gt], seed=42)
    runner = TrajectoryEvaluationRunner(max_attempts=3)
    strategy = MockEscalateStrategy()

    episode = runner.evaluate_episode(mock_payment_record, strategy, env)

    assert episode.terminal_state == PaymentState.ESCALATED
    assert episode.attempt_count == 1
    assert len(episode.steps) == 1
    assert episode.steps[0].authorized_action == Action.ESCALATE
    # ESCALATE direct cost = 3.50, penalty = 0.00
    assert episode.total_cost == Decimal("3.50")
    assert episode.total_penalty == Decimal("0.00")


def test_trajectory_crn_determinism(mock_payment_record):
    """Test 5: Verify identical seeds and strategy yield 100% deterministic trajectory outcomes."""
    gt = GroundTruthRecord(
        payment_id=mock_payment_record.payment_id,
        latent_recoverability_profile=RecoverabilityProfile.MEDIUM,
        action_base_probabilities={a: 0.5 for a in Action},
    )
    strategy = FixedRetryStrategy()

    # Run 1
    env1 = SimulationEnvironment([gt], seed=777)
    runner1 = TrajectoryEvaluationRunner(max_attempts=3)
    ep1 = runner1.evaluate_episode(mock_payment_record, strategy, env1)

    # Run 2 with same seed
    env2 = SimulationEnvironment([gt], seed=777)
    runner2 = TrajectoryEvaluationRunner(max_attempts=3)
    ep2 = runner2.evaluate_episode(mock_payment_record, strategy, env2)

    assert ep1.final_recovered == ep2.final_recovered
    assert ep1.terminal_state == ep2.terminal_state
    assert ep1.attempt_count == ep2.attempt_count
    assert ep1.net_recovered_value == ep2.net_recovered_value
    assert len(ep1.steps) == len(ep2.steps)
    for s1, s2 in zip(ep1.steps, ep2.steps):
        assert s1.recovered == s2.recovered
        assert s1.step_cost == s2.step_cost
        assert s1.step_penalty == s2.step_penalty


def test_trajectory_economic_accumulation():
    """Test 6: Verify step costs, penalties, and NRV accumulate correctly across multiple steps."""
    rec = SyntheticPaymentRecord(
        payment_id="pay_accum_01",
        customer_id="cust_accum",
        amount=Decimal("500.00"),
        currency="INR",
        failure_category=FailureCategory.NETWORK_TIMEOUT,
        failure_severity=FailureSeverity.TRANSIENT,
        customer_tier=CustomerTier.VIP,  # VIP penalty multiplier = 3.0
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="TIMEOUT",
        raw_error_message="Timeout",
        failure_timestamp=datetime.now(timezone.utc),
        attempt_count=1,
    )
    # Recover only on attempt 2
    class SecondAttemptSuccessEnv:
        def __init__(self):
            self.calls = 0
        def apply_action(self, pid, act):
            self.calls += 1
            from recoveriq.simulation.environment import ActionOutcome
            recovered = (self.calls == 2)
            return ActionOutcome(payment_id=pid, action=act, recovered=recovered, true_probability=0.5)

    env = SecondAttemptSuccessEnv()
    runner = TrajectoryEvaluationRunner(max_attempts=3)
    strategy = FixedRetryStrategy()

    ep = runner.evaluate_episode(rec, strategy, env)

    assert ep.final_recovered is True
    assert ep.terminal_state == PaymentState.RECOVERED
    assert ep.attempt_count == 2
    assert len(ep.steps) == 2

    # VIP RETRY_NOW: cost = 0.15, penalty = 0.05 * 3.0 = 0.15
    # Total cost = 2 * 0.15 = 0.30
    # Total penalty = 2 * 0.15 = 0.30
    # Net NRV = 500.00 - 0.30 - 0.30 = 499.40
    assert ep.total_cost == Decimal("0.30")
    assert ep.total_penalty == Decimal("0.30")
    assert ep.net_recovered_value == Decimal("499.40")


def test_trajectory_hard_decline_clamped_at_step_one(mock_hard_decline_record):
    """Verify hard decline is clamped to STOP by the Policy Gate on attempt 1."""
    gt = GroundTruthRecord(
        payment_id=mock_hard_decline_record.payment_id,
        latent_recoverability_profile=RecoverabilityProfile.VERY_LOW,
        action_base_probabilities={a: 0.0 for a in Action},
    )
    env = SimulationEnvironment([gt], seed=42)
    runner = TrajectoryEvaluationRunner(max_attempts=3)
    # FixedRetry proposes RETRY_NOW, which violates hard decline policy
    strategy = FixedRetryStrategy()

    episode = runner.evaluate_episode(mock_hard_decline_record, strategy, env)

    assert episode.final_recovered is False
    assert episode.terminal_state == PaymentState.FAILED_TERMINAL
    assert episode.attempt_count == 1
    assert len(episode.steps) == 1
    assert episode.steps[0].proposed_action == Action.RETRY_NOW
    assert episode.steps[0].authorized_action == Action.STOP
    assert episode.steps[0].is_authorized is False
    assert "hard decline" in episode.steps[0].rejection_reason.lower()


def test_trajectory_metrics_computation():
    """Verify TrajectoryStrategyMetrics computes survival rate, lift, and terminal distributions."""
    episodes = [
        # Episode 1: recovered at step 1
        TrajectoryEpisode(
            payment_id="p1",
            steps=[TrajectoryStep(1, Action.RETRY_NOW, Action.RETRY_NOW, True, True, Decimal("0.15"), Decimal("0.05"), PaymentState.RECOVERED)],
            terminal_state=PaymentState.RECOVERED,
            final_recovered=True,
            payment_amount=Decimal("100.00"),
            total_cost=Decimal("0.15"),
            total_penalty=Decimal("0.05"),
            net_recovered_value=Decimal("99.80"),
            attempt_count=1,
        ),
        # Episode 2: failed 3 times
        TrajectoryEpisode(
            payment_id="p2",
            steps=[
                TrajectoryStep(1, Action.RETRY_NOW, Action.RETRY_NOW, True, False, Decimal("0.15"), Decimal("0.05"), PaymentState.RECOVERING),
                TrajectoryStep(2, Action.RETRY_NOW, Action.RETRY_NOW, True, False, Decimal("0.15"), Decimal("0.05"), PaymentState.RECOVERING),
                TrajectoryStep(3, Action.RETRY_NOW, Action.RETRY_NOW, True, False, Decimal("0.15"), Decimal("0.05"), PaymentState.FAILED_TERMINAL),
            ],
            terminal_state=PaymentState.FAILED_TERMINAL,
            final_recovered=False,
            payment_amount=Decimal("100.00"),
            total_cost=Decimal("0.45"),
            total_penalty=Decimal("0.15"),
            net_recovered_value=Decimal("-0.60"),
            attempt_count=3,
        ),
        # Episode 3: escalated at step 1
        TrajectoryEpisode(
            payment_id="p3",
            steps=[TrajectoryStep(1, Action.ESCALATE, Action.ESCALATE, True, False, Decimal("3.50"), Decimal("0.00"), PaymentState.ESCALATED)],
            terminal_state=PaymentState.ESCALATED,
            final_recovered=False,
            payment_amount=Decimal("100.00"),
            total_cost=Decimal("3.50"),
            total_penalty=Decimal("0.00"),
            net_recovered_value=Decimal("-3.50"),
            attempt_count=1,
        ),
    ]

    metrics = TrajectoryStrategyMetrics.compute("TestStrategy", seed=42, episodes=episodes, max_attempts=3)

    assert metrics.n_payments == 3
    assert metrics.n_recovered == 1
    assert metrics.n_escalated == 1
    assert metrics.n_failed_terminal == 1
    assert pytest.approx(metrics.recovery_rate) == 1 / 3
    assert pytest.approx(metrics.escalation_rate) == 1 / 3
    assert pytest.approx(metrics.average_attempts_per_payment) == (1 + 3 + 1) / 3

    # Survival rate by step:
    # Step 1: all 3 active
    # Step 2: only p2 active (1/3)
    # Step 3: only p2 active (1/3)
    assert metrics.survival_rate_by_step[1] == 1.0
    assert pytest.approx(metrics.survival_rate_by_step[2]) == 1 / 3
    assert pytest.approx(metrics.survival_rate_by_step[3]) == 1 / 3

    # Recovery lift by attempt:
    # Attempt 1: p1 recovered (1/3)
    # Attempt 2: 0 (0/3)
    # Attempt 3: 0 (0/3)
    assert pytest.approx(metrics.recovery_lift_by_attempt[1]) == 1 / 3
    assert metrics.recovery_lift_by_attempt[2] == 0.0
    assert metrics.recovery_lift_by_attempt[3] == 0.0
