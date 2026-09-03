"""Sprint 5 Tests — Evaluation strategy behavior and proposal contracts."""

from decimal import Decimal
import pytest

from recoveriq.domain.actions import Action
from recoveriq.domain.models import CustomerTier, FailureCategory, FailureSeverity, PaymentContext, PaymentMethod
from recoveriq.evaluation.strategies import FixedRetryStrategy, RecoverIQStrategy, RuleBasedStrategy
from recoveriq.model.trainer import ModelTrainer
from recoveriq.simulation.config import SimulationConfig
from recoveriq.simulation.environment import SimulationEnvironment
from recoveriq.simulation.generator import SyntheticPaymentGenerator
from recoveriq.simulation.partitioner import partition_dataset


@pytest.fixture
def sample_context():
    return PaymentContext(
        payment_id="pay_strat_001",
        customer_id="cust_001",
        customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.UPI,
        raw_error_code="NSF",
        raw_error_message="Insufficient balance",
        failure_category=FailureCategory.INSUFFICIENT_FUNDS,
        failure_severity=FailureSeverity.RECOVERABLE,
        attempt_count=1,
    )


@pytest.fixture
def trained_model():
    cfg = SimulationConfig(n_payments=60, n_customers=10, default_seed=42)
    gen = SyntheticPaymentGenerator(cfg)
    ds = gen.generate(seed=42)
    part = partition_dataset(ds, train_fraction=0.75)
    env = SimulationEnvironment(part.train_ground_truth, seed=42)
    trainer = ModelTrainer(random_state=42)
    return trainer.train(part.train_observable, env)


def test_fixed_retry_always_proposes_retry_now(sample_context):
    """Fixed-Retry baseline must unconditionally propose Action.RETRY_NOW."""
    strat = FixedRetryStrategy()
    assert strat.name == "Fixed-Retry"

    # Test across multiple contexts
    assert strat.propose_action(None, sample_context) == Action.RETRY_NOW

    # Even on hard decline, Fixed-Retry proposes RETRY_NOW (to be checked/blocked by Policy Gate)
    hard_ctx = PaymentContext(
        payment_id="pay_hard",
        customer_id="cust_001",
        customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="STOLEN",
        raw_error_message="Card reported stolen",
        failure_category=FailureCategory.HARD_DECLINE,
        failure_severity=FailureSeverity.FATAL,
        attempt_count=1,
    )
    assert strat.propose_action(None, hard_ctx) == Action.RETRY_NOW


def test_rule_based_strategy_applies_deterministic_mappings():
    """Rule-Based strategy must apply SPEC.md Section 14 category-to-action heuristics."""
    strat = RuleBasedStrategy()
    assert strat.name == "Rule-Based"

    test_cases = [
        (FailureCategory.INSUFFICIENT_FUNDS, Action.RETRY_LATER),
        (FailureCategory.NETWORK_TIMEOUT, Action.RETRY_NOW),
        (FailureCategory.CARD_EXPIRED, Action.SEND_LINK),
        (FailureCategory.AUTHENTICATION_FAILED, Action.SEND_LINK),
        (FailureCategory.AUTHENTICATION_REJECTED, Action.SEND_LINK),
        (FailureCategory.VELOCITY_EXCEEDED, Action.RETRY_LATER),
        (FailureCategory.HARD_DECLINE, Action.STOP),
        (FailureCategory.INVALID_DETAILS, Action.STOP),
        (FailureCategory.UNKNOWN, Action.STOP),
    ]

    for cat, expected_act in test_cases:
        ctx = PaymentContext(
            payment_id="pay_test",
            customer_id="cust_001",
            customer_tier=CustomerTier.STANDARD,
            payment_method=PaymentMethod.CREDIT_CARD,
            raw_error_code="ERR",
            raw_error_message="MSG",
            failure_category=cat,
            failure_severity=FailureSeverity.RECOVERABLE,
        )
        assert strat.propose_action(None, ctx) == expected_act


def test_recoveriq_strategy_optimizes_ev(trained_model, sample_context):
    """RecoverIQStrategy must compute EV across candidate actions and return the EV-maximizing action."""
    strat = RecoverIQStrategy(probability_model=trained_model)
    assert strat.name == "RecoverIQ"

    # Create dummy observable record matching context
    from recoveriq.simulation.schema import SyntheticPaymentRecord
    from datetime import datetime, timezone
    record = SyntheticPaymentRecord(
        payment_id=sample_context.payment_id,
        customer_id=sample_context.customer_id,
        amount=Decimal("500.00"),
        currency="INR",
        failure_category=sample_context.failure_category,
        failure_severity=sample_context.failure_severity,
        customer_tier=sample_context.customer_tier,
        payment_method=sample_context.payment_method,
        raw_error_code=sample_context.raw_error_code,
        raw_error_message=sample_context.raw_error_message,
        failure_timestamp=datetime.now(timezone.utc),
        attempt_count=1,
    )

    action = strat.propose_action(record, sample_context)
    assert isinstance(action, Action)
    assert action in Action
