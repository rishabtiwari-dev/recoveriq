"""Tests for domain models and data integrity."""

from decimal import Decimal
import pytest

from recoveriq.domain.models import (
    CustomerTier,
    FailureCategory,
    FailureSeverity,
    Payment,
    PaymentContext,
    PaymentMethod,
)
from recoveriq.domain.state import PaymentState
from recoveriq.domain.events import PaymentFailedEvent, OutcomeReceivedEvent, EventType
from recoveriq.domain.actions import Action


def test_payment_creation_valid():
    """Verify Payment initialization and Decimal amount coercion."""
    payment = Payment(
        payment_id="pay_1001",
        customer_id="cust_2001",
        amount=Decimal("150.75"),
        currency="USD",
    )
    assert payment.payment_id == "pay_1001"
    assert payment.customer_id == "cust_2001"
    assert payment.amount == Decimal("150.75")
    assert payment.state == PaymentState.FAILED_INITIAL
    assert payment.attempt_count == 0
    assert payment.is_terminal is False


def test_payment_creation_from_float_or_int():
    """Verify numeric conversion to Decimal."""
    payment = Payment(
        payment_id="pay_1002",
        customer_id="cust_2002",
        amount="49.99",  # type: ignore
    )
    assert isinstance(payment.amount, Decimal)
    assert payment.amount == Decimal("49.99")


def test_payment_rejects_negative_or_zero_amount():
    """Verify Payment rejects non-positive amounts."""
    with pytest.raises(ValueError, match="strictly positive"):
        Payment(
            payment_id="pay_invalid",
            customer_id="cust_invalid",
            amount=Decimal("0.00"),
        )

    with pytest.raises(ValueError, match="strictly positive"):
        Payment(
            payment_id="pay_invalid",
            customer_id="cust_invalid",
            amount=Decimal("-10.00"),
        )


def test_payment_context_creation():
    """Verify PaymentContext properties and failure category helpers."""
    ctx = PaymentContext(
        payment_id="pay_1003",
        customer_id="cust_2003",
        customer_tier=CustomerTier.VIP,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="INSUFFICIENT_FUNDS",
        raw_error_message="Card issuer reported low balance",
        failure_category=FailureCategory.INSUFFICIENT_FUNDS,
        failure_severity=FailureSeverity.RECOVERABLE,
    )
    assert ctx.failure_category == FailureCategory.INSUFFICIENT_FUNDS
    assert ctx.failure_category.is_hard_decline is False
    assert ctx.customer_tier == CustomerTier.VIP


def test_hard_decline_category_identification():
    """Verify failure category hard decline recognition."""
    assert FailureCategory.HARD_DECLINE.is_hard_decline is True
    assert FailureCategory.INVALID_DETAILS.is_hard_decline is True
    assert FailureCategory.NETWORK_TIMEOUT.is_hard_decline is False
    assert FailureCategory.INSUFFICIENT_FUNDS.is_hard_decline is False


def test_payment_failed_event():
    """Verify PaymentFailedEvent construction and type checks."""
    evt = PaymentFailedEvent(
        event_id="evt_001",
        payment_id="pay_1004",
        event_type=EventType.PAYMENT_FAILED,
        customer_id="cust_2004",
        amount=Decimal("250.00"),
        customer_tier=CustomerTier.PREMIUM,
        payment_method=PaymentMethod.UPI,
        raw_error_code="504_TIMEOUT",
        raw_error_message="Gateway timeout",
    )
    assert evt.event_id == "evt_001"
    assert evt.amount == Decimal("250.00")
    assert evt.event_type == EventType.PAYMENT_FAILED
