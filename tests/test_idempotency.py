"""Tests for idempotency contracts and deterministic key calculation."""

from decimal import Decimal
from recoveriq.domain.actions import Action
from recoveriq.domain.decisions import PolicyDecision
from recoveriq.domain.idempotency import (
    IdempotencyRecord,
    IdempotencyStatus,
    generate_idempotency_key,
)
from recoveriq.domain.models import Payment
from recoveriq.executor.executor import ExecutionStatus, InMemoryActionExecutor


def test_idempotency_key_deterministic():
    """Verify idempotency key calculation is deterministic and sensitive to parameters."""
    key1 = generate_idempotency_key("pay_1", Action.RETRY_NOW, 1, "evt_100")
    key2 = generate_idempotency_key("pay_1", Action.RETRY_NOW, 1, "evt_100")
    key_diff_attempt = generate_idempotency_key("pay_1", Action.RETRY_NOW, 2, "evt_100")
    key_diff_action = generate_idempotency_key("pay_1", Action.RETRY_LATER, 1, "evt_100")

    assert key1 == key2
    assert key1 != key_diff_attempt
    assert key1 != key_diff_action
    assert len(key1) == 64  # SHA-256 hex digest length


def test_idempotency_record_lifecycle():
    """Verify IdempotencyRecord status transitions."""
    record = IdempotencyRecord.create("pay_1", Action.SEND_LINK, 1, "evt_100")
    assert record.status == IdempotencyStatus.PENDING
    assert record.completed_at is None

    record.mark_completed({"url": "https://pay.example.com/link_123"})
    assert record.status == IdempotencyStatus.COMPLETED
    assert record.completed_at is not None
    assert record.response_payload == {"url": "https://pay.example.com/link_123"}


def test_executor_idempotency_guard():
    """Verify executor prevents duplicate execution of the same idempotency key."""
    executor = InMemoryActionExecutor()
    payment = Payment(payment_id="pay_10", customer_id="cust_10", amount=Decimal("100.00"))
    decision = PolicyDecision.authorize("pay_10", Action.RETRY_NOW)

    # First dispatch -> SUCCESS
    result1 = executor.execute(payment, decision, "evt_999")
    assert result1.status == ExecutionStatus.SUCCESS

    # Second dispatch with identical parameters -> SKIPPED_IDEMPOTENT
    result2 = executor.execute(payment, decision, "evt_999")
    assert result2.status == ExecutionStatus.SKIPPED_IDEMPOTENT
    assert result1.idempotency_key == result2.idempotency_key
