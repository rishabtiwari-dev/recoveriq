"""Sprint 4 Tests — Seamless integration of TrainedRecoveryProbabilityModel with RecoverIQEngine."""

from decimal import Decimal
import pytest

from recoveriq.domain.actions import Action
from recoveriq.domain.events import EventType, PaymentFailedEvent
from recoveriq.domain.models import CustomerTier, PaymentMethod
from recoveriq.engine import RecoverIQEngine
from recoveriq.executor.executor import ExecutionStatus
from recoveriq.model.trainer import ModelTrainer
from recoveriq.simulation.config import SimulationConfig
from recoveriq.simulation.environment import SimulationEnvironment
from recoveriq.simulation.generator import SyntheticPaymentGenerator
from recoveriq.simulation.partitioner import partition_dataset


@pytest.fixture
def trained_model():
    cfg = SimulationConfig(n_payments=100, n_customers=20, default_seed=42)
    gen = SyntheticPaymentGenerator(cfg)
    ds = gen.generate(seed=42)
    partitioned = partition_dataset(ds, train_fraction=0.75)
    env = SimulationEnvironment(partitioned.train_ground_truth, seed=42)

    trainer = ModelTrainer(c_regularization=1.0, random_state=42)
    return trainer.train(partitioned.train_observable, env)


def test_engine_executes_with_trained_probability_model(trained_model):
    """RecoverIQEngine accepts TrainedRecoveryProbabilityModel via dependency injection with zero engine modifications."""
    engine = RecoverIQEngine(probability_model=trained_model)

    event = PaymentFailedEvent(
        event_id="evt_trained_001",
        payment_id="pay_trained_001",
        event_type=EventType.PAYMENT_FAILED,
        customer_id="cust_trained_001",
        amount=Decimal("250.00"),
        currency="USD",
        customer_tier=CustomerTier.PREMIUM,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="INSUFFICIENT_FUNDS",
        raw_error_message="Account balance is too low",
        attempt_count=1,
    )

    result = engine.process_failure_event(event)

    assert result.payment_id == "pay_trained_001"
    assert result.status == ExecutionStatus.SUCCESS
    assert result.action in Action

    # Confirm probability estimation audit log captured trained model probabilities
    audit_events = engine.audit_logger.get_events_for_payment("pay_trained_001")
    prob_events = [e for e in audit_events if e.event_type.value == "PROBABILITY_ESTIMATION"]
    assert len(prob_events) == 1
    details = prob_events[0].details
    assert "STOP" in details
    assert details["STOP"] == 0.0
