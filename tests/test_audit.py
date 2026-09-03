"""Tests for the Structured Audit Logging subsystem."""

from datetime import datetime, timezone
from recoveriq.audit.logger import (
    AuditEvent,
    AuditEventType,
    InMemoryAuditLogger,
)


def test_audit_event_creation():
    """Verify AuditEvent construction and uuid generation."""
    event = AuditEvent.create(
        event_type=AuditEventType.INGESTION,
        payment_id="pay_audit_1",
        details={"amount": "100.00"},
    )
    assert event.payment_id == "pay_audit_1"
    assert event.event_type == AuditEventType.INGESTION
    assert event.details["amount"] == "100.00"
    assert event.audit_id is not None
    assert event.timestamp is not None


def test_in_memory_audit_logger():
    """Verify appending and querying audit records."""
    logger = InMemoryAuditLogger()
    evt1 = AuditEvent.create(AuditEventType.INGESTION, "pay_1", {"key": "1"})
    evt2 = AuditEvent.create(AuditEventType.POLICY_AUTHORIZATION, "pay_1", {"key": "2"})
    evt3 = AuditEvent.create(AuditEventType.INGESTION, "pay_2", {"key": "3"})

    logger.log(evt1)
    logger.log(evt2)
    logger.log(evt3)

    pay1_events = logger.get_events_for_payment("pay_1")
    assert len(pay1_events) == 2
    assert pay1_events[0].event_type == AuditEventType.INGESTION
    assert pay1_events[1].event_type == AuditEventType.POLICY_AUTHORIZATION

    pay2_events = logger.get_events_for_payment("pay_2")
    assert len(pay2_events) == 1
    assert pay2_events[0].payment_id == "pay_2"
