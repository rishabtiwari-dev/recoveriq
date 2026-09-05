"""Sprint 12 Tests — Robustness, Sensitivity & Statistical Validation."""

from datetime import datetime, timezone
from decimal import Decimal
import pytest

from recoveriq.config.settings import ActionCostConfig, PenaltyConfig
from recoveriq.domain.actions import Action
from recoveriq.domain.models import CustomerTier, FailureCategory, FailureSeverity, PaymentContext, PaymentMethod
from recoveriq.domain.state import PaymentState
from recoveriq.evaluation.robustness import (
    SPRINT12_EXPANDED_SEEDS,
    BreakEvenDiagnostic,
    calculate_human_ops_valuation_sweep,
    compute_break_even_diagnostic,
    compute_paired_crn_differences,
    stratify_payments_by_value,
)
from recoveriq.evaluation.sequential_policy import TieredRecoverIQStrategy
from recoveriq.evaluation.strategies import FixedRetryStrategy, RecoverIQStrategy
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


class MockConstantModel(RecoveryProbabilityModel):
    """Mock probability model with known fixed estimates."""

    def __init__(self, probs=None):
        self.probs = probs or {
            Action.RETRY_NOW: Decimal("0.30"),
            Action.RETRY_LATER: Decimal("0.40"),
            Action.SEND_LINK: Decimal("0.35"),
            Action.NUDGE: Decimal("0.20"),
            Action.ESCALATE: Decimal("0.55"),
            Action.STOP: Decimal("0.00"),
        }

    def estimate_probabilities(self, context: PaymentContext):
        return {
            act: ProbabilityEstimate(act, prob) for act, prob in self.probs.items()
        }


@pytest.fixture
def mock_payment_records():
    """Create sample payments spanning multiple values for stratification."""
    return [
        SyntheticPaymentRecord(
            payment_id=f"pay_strat_{i}",
            customer_id=f"cust_{i}",
            amount=Decimal(str(val)),
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
        for i, val in enumerate([100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0])
    ]


def test_phuman_zero_produces_no_human_ops_value():
    """Test 1: Contract test verifying Phuman = 0 produces exactly 0.00 human-ops value."""
    episodes = [
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
        )
    ]
    auto_metrics = TrajectoryStrategyMetrics.compute("Test", seed=42, episodes=episodes, max_attempts=3)
    h_val, full_nrv = calculate_human_ops_valuation_sweep(episodes, p_human=0.0, automated_metrics=auto_metrics)

    assert h_val == Decimal("0.00")
    assert full_nrv == auto_metrics.total_nrv


def test_phuman_one_produces_exact_gross_amount():
    """Test 2: Contract test verifying Phuman = 1.0 produces exactly sum(payment_amount) for escalated payments."""
    episodes = [
        TrajectoryEpisode(
            payment_id="p1",
            steps=[TrajectoryStep(1, Action.ESCALATE, Action.ESCALATE, True, False, Decimal("3.50"), Decimal("0.00"), PaymentState.ESCALATED)],
            terminal_state=PaymentState.ESCALATED,
            final_recovered=False,
            payment_amount=Decimal("1500.00"),
            total_cost=Decimal("3.50"),
            total_penalty=Decimal("0.00"),
            net_recovered_value=Decimal("-3.50"),
            attempt_count=1,
        )
    ]
    auto_metrics = TrajectoryStrategyMetrics.compute("Test", seed=42, episodes=episodes, max_attempts=3)
    h_val, full_nrv = calculate_human_ops_valuation_sweep(episodes, p_human=1.0, automated_metrics=auto_metrics)

    assert h_val == Decimal("1500.00")
    assert full_nrv == auto_metrics.total_nrv + Decimal("1500.00")


def test_human_ops_valuation_does_not_mutate_automated_accounting():
    """Test 3: Contract test verifying human-ops valuation leaves TrajectoryEpisode completely unmutated."""
    ep = TrajectoryEpisode(
        payment_id="p1",
        steps=[TrajectoryStep(1, Action.ESCALATE, Action.ESCALATE, True, False, Decimal("3.50"), Decimal("0.00"), PaymentState.ESCALATED)],
        terminal_state=PaymentState.ESCALATED,
        final_recovered=False,
        payment_amount=Decimal("1000.00"),
        total_cost=Decimal("3.50"),
        total_penalty=Decimal("0.00"),
        net_recovered_value=Decimal("-3.50"),
        attempt_count=1,
    )
    auto_metrics = TrajectoryStrategyMetrics.compute("Test", seed=42, episodes=[ep], max_attempts=3)
    calculate_human_ops_valuation_sweep([ep], p_human=0.75, automated_metrics=auto_metrics)

    assert ep.final_recovered is False
    assert ep.terminal_state == PaymentState.ESCALATED
    assert ep.net_recovered_value == Decimal("-3.50")
    assert auto_metrics.total_nrv == Decimal("-3.50")


def test_tiered_strategy_remains_unchanged_across_valuations():
    """Test 4: Verify TieredRecoverIQStrategy decisions are independent of valuation assumptions."""
    model = MockConstantModel()
    strat = TieredRecoverIQStrategy(probability_model=model, max_attempts=3)
    rec = SyntheticPaymentRecord(
        payment_id="p_test",
        customer_id="c1",
        amount=Decimal("2000.00"),
        currency="INR",
        failure_category=FailureCategory.NETWORK_TIMEOUT,
        failure_severity=FailureSeverity.TRANSIENT,
        customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.UPI,
        raw_error_code="TIMEOUT",
        raw_error_message="Timeout",
        failure_timestamp=datetime.now(timezone.utc),
        attempt_count=1,
    )
    ctx1 = PaymentContext(
        payment_id=rec.payment_id,
        customer_id=rec.customer_id,
        customer_tier=rec.customer_tier,
        payment_method=rec.payment_method,
        raw_error_code=rec.raw_error_code,
        raw_error_message=rec.raw_error_message,
        failure_category=rec.failure_category,
        failure_severity=rec.failure_severity,
        attempt_count=1,
    )
    # Proposes best automated (RETRY_LATER) regardless of any external evaluation parameter
    assert strat.propose_action(rec, ctx1) == Action.RETRY_LATER


def test_existing_recoveriq_strategy_remains_unchanged():
    """Test 5: Verify original RecoverIQStrategy continues to select ESCALATE when EV is highest."""
    model = MockConstantModel()
    strat = RecoverIQStrategy(probability_model=model)
    rec = SyntheticPaymentRecord(
        payment_id="p_test",
        customer_id="c1",
        amount=Decimal("2000.00"),
        currency="INR",
        failure_category=FailureCategory.NETWORK_TIMEOUT,
        failure_severity=FailureSeverity.TRANSIENT,
        customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.UPI,
        raw_error_code="TIMEOUT",
        raw_error_message="Timeout",
        failure_timestamp=datetime.now(timezone.utc),
        attempt_count=1,
    )
    ctx1 = PaymentContext(
        payment_id=rec.payment_id,
        customer_id=rec.customer_id,
        customer_tier=rec.customer_tier,
        payment_method=rec.payment_method,
        raw_error_code=rec.raw_error_code,
        raw_error_message=rec.raw_error_message,
        failure_category=rec.failure_category,
        failure_severity=rec.failure_severity,
        attempt_count=1,
    )
    # Original strategy selects ESCALATE unconstrained
    assert strat.propose_action(rec, ctx1) == Action.ESCALATE


def test_seed_expansion_is_deterministic():
    """Test 7: Verify SPRINT12_EXPANDED_SEEDS contains exactly 20 seeds and retains original 5."""
    assert len(SPRINT12_EXPANDED_SEEDS) == 20
    assert len(set(SPRINT12_EXPANDED_SEEDS)) == 20  # No duplicates
    assert SPRINT12_EXPANDED_SEEDS[:5] == [42, 100, 777, 999, 2024]


def test_payment_value_stratification_coverage(mock_payment_records):
    """Test 8: Verify stratification partitions payments into Lower, Middle, Higher with 100% coverage and zero overlap."""
    strata = stratify_payments_by_value(mock_payment_records)

    assert set(strata.keys()) == {"Lower-Value", "Middle-Value", "Higher-Value"}

    all_partitioned_pids = []
    for name, (min_amt, max_amt, recs) in strata.items():
        assert len(recs) > 0
        all_partitioned_pids.extend([r.payment_id for r in recs])
        for r in recs:
            assert min_amt <= r.amount <= max_amt

    # Exactly covers all original payments
    assert len(all_partitioned_pids) == len(mock_payment_records)
    assert set(all_partitioned_pids) == set(r.payment_id for r in mock_payment_records)


def test_break_even_equation_positive_gap():
    """Test 9: Verify theoretical V* calculation when P(ESCALATE) > P(best_auto)."""
    # P(ESCALATE) = 0.55, P(RETRY_LATER) = 0.40 -> Gap = 0.15
    # C(ESCALATE) = 3.50, C(RETRY_LATER) = 0.15, pen_esc = 0, pen_auto = 0.02 -> Cost gap = 3.50 - 0.17 = 3.33
    # V* = 3.33 / 0.15 = 22.20
    model = MockConstantModel()
    rec = SyntheticPaymentRecord(
        payment_id="p_be",
        customer_id="c1",
        amount=Decimal("100.00"),  # 100.00 > V* (22.20)
        currency="INR",
        failure_category=FailureCategory.INSUFFICIENT_FUNDS,
        failure_severity=FailureSeverity.RECOVERABLE,
        customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.UPI,
        raw_error_code="NSF",
        raw_error_message="NSF",
        failure_timestamp=datetime.now(timezone.utc),
        attempt_count=1,
    )
    ctx = PaymentContext(
        payment_id=rec.payment_id,
        customer_id=rec.customer_id,
        customer_tier=rec.customer_tier,
        payment_method=rec.payment_method,
        raw_error_code=rec.raw_error_code,
        raw_error_message=rec.raw_error_message,
        failure_category=rec.failure_category,
        failure_severity=rec.failure_severity,
        attempt_count=1,
    )

    diag = compute_break_even_diagnostic(rec, ctx, model)
    assert diag.theoretical_v_star is not None
    assert pytest.approx(diag.theoretical_v_star, 0.01) == 22.20
    # Because V=100.00 > V*=22.20, RecoverIQ should propose ESCALATE
    assert diag.actual_proposed_action == Action.ESCALATE
    assert diag.agrees_with_prediction is True


def test_break_even_equation_negative_gap():
    """Test 10: Verify theoretical V* returns None when P(ESCALATE) <= P(best_auto)."""
    probs = {
        Action.RETRY_NOW: Decimal("0.30"),
        Action.RETRY_LATER: Decimal("0.60"),  # Best auto > ESCALATE
        Action.SEND_LINK: Decimal("0.35"),
        Action.NUDGE: Decimal("0.20"),
        Action.ESCALATE: Decimal("0.50"),
        Action.STOP: Decimal("0.00"),
    }
    model = MockConstantModel(probs)
    rec = SyntheticPaymentRecord(
        payment_id="p_be_neg",
        customer_id="c1",
        amount=Decimal("1000.00"),
        currency="INR",
        failure_category=FailureCategory.INSUFFICIENT_FUNDS,
        failure_severity=FailureSeverity.RECOVERABLE,
        customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.UPI,
        raw_error_code="NSF",
        raw_error_message="NSF",
        failure_timestamp=datetime.now(timezone.utc),
        attempt_count=1,
    )
    ctx = PaymentContext(
        payment_id=rec.payment_id,
        customer_id=rec.customer_id,
        customer_tier=rec.customer_tier,
        payment_method=rec.payment_method,
        raw_error_code=rec.raw_error_code,
        raw_error_message=rec.raw_error_message,
        failure_category=rec.failure_category,
        failure_severity=rec.failure_severity,
        attempt_count=1,
    )

    diag = compute_break_even_diagnostic(rec, ctx, model)
    assert diag.theoretical_v_star is None
    # Because ESCALATE has lower prob and higher cost, it should NOT be proposed
    assert diag.actual_proposed_action != Action.ESCALATE
    assert diag.agrees_with_prediction is True


def test_paired_observations_preserve_payment_identity():
    """Test 11: Verify paired differences accurately match records by payment_id."""
    gt = [
        GroundTruthRecord("p1", RecoverabilityProfile.HIGH, {Action.ESCALATE: 0.80}),
        GroundTruthRecord("p2", RecoverabilityProfile.LOW, {Action.ESCALATE: 0.20}),
    ]
    # Strategy A: recovered p1, failed p2
    eps_a = [
        TrajectoryEpisode("p1", [], PaymentState.RECOVERED, True, Decimal("100.0"), Decimal("0.15"), Decimal("0.05"), Decimal("99.80"), 1),
        TrajectoryEpisode("p2", [], PaymentState.FAILED_TERMINAL, False, Decimal("100.0"), Decimal("0.45"), Decimal("0.15"), Decimal("-0.60"), 3),
    ]
    # Strategy B: failed p1, recovered p2
    eps_b = [
        TrajectoryEpisode("p1", [], PaymentState.FAILED_TERMINAL, False, Decimal("100.0"), Decimal("0.45"), Decimal("0.15"), Decimal("-0.60"), 3),
        TrajectoryEpisode("p2", [], PaymentState.RECOVERED, True, Decimal("100.0"), Decimal("0.15"), Decimal("0.05"), Decimal("99.80"), 1),
    ]

    res = compute_paired_crn_differences(eps_a, eps_b, gt, "StratA", "StratB")
    assert res.n_observations == 2
    # p1 diff = 99.80 - (-0.60) = 100.40; p2 diff = -0.60 - 99.80 = -100.40
    # Mean diff = 0.00
    assert pytest.approx(res.mean_nrv_diff_per_payment, 0.01) == 0.00
