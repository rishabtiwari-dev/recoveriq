"""End-to-end integration tests verifying pipeline contracts and module boundaries."""

from decimal import Decimal
from recoveriq.domain.actions import Action
from recoveriq.domain.events import EventType, PaymentFailedEvent
from recoveriq.domain.models import CustomerTier, PaymentMethod
from recoveriq.domain.state import PaymentState
from recoveriq.engine import RecoverIQEngine
from recoveriq.executor.executor import ExecutionStatus


def test_pipeline_end_to_end_success():
    """Verify complete failure event handling through all decoupled pipeline layers."""
    engine = RecoverIQEngine()

    event = PaymentFailedEvent(
        event_id="evt_pipe_001",
        payment_id="pay_pipe_001",
        event_type=EventType.PAYMENT_FAILED,
        customer_id="cust_pipe_001",
        amount=Decimal("120.00"),
        currency="USD",
        customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="504_GATEWAY_TIMEOUT",
        raw_error_message="Issuer switch timeout",
        attempt_count=1,
    )

    result = engine.process_failure_event(event)

    # In our baseline contracts, 504_GATEWAY_TIMEOUT -> NETWORK_TIMEOUT -> ESCALATE (highest net EV for $120) or RETRY_NOW
    assert result.payment_id == "pay_pipe_001"
    assert result.status == ExecutionStatus.SUCCESS
    assert result.idempotency_key is not None

    # Check complete audit trail is produced
    audit_events = engine.audit_logger.get_events_for_payment("pay_pipe_001")
    event_types = [e.event_type.value for e in audit_events]

    assert "INGESTION" in event_types
    assert "CONTEXT_EXTRACTION" in event_types
    assert "PROBABILITY_ESTIMATION" in event_types
    assert "ECONOMIC_EVALUATION" in event_types
    assert ("POLICY_AUTHORIZATION" in event_types) or ("POLICY_REJECTION" in event_types)
    assert "ACTION_EXECUTION" in event_types
    assert "STATE_TRANSITION" in event_types


def test_pipeline_prevents_ai_execution_authority():
    """Verify AI layer has no direct execution authority and only extracts context."""
    engine = RecoverIQEngine()

    # The AI layer does not have an execute() method
    assert not hasattr(engine.ai_layer, "execute")
    assert not hasattr(engine.ai_layer, "dispatch")
    assert hasattr(engine.ai_layer, "interpret_failure")
