"""Structured Audit Logging interfaces and append-only event models."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, unique
from typing import Any, Dict, List, Protocol, runtime_checkable


@unique
class AuditEventType(str, Enum):
    """Lifecycle stages captured in the append-only audit trail."""

    INGESTION = "INGESTION"
    CONTEXT_EXTRACTION = "CONTEXT_EXTRACTION"
    PROBABILITY_ESTIMATION = "PROBABILITY_ESTIMATION"
    ECONOMIC_EVALUATION = "ECONOMIC_EVALUATION"
    POLICY_AUTHORIZATION = "POLICY_AUTHORIZATION"
    POLICY_REJECTION = "POLICY_REJECTION"
    ACTION_EXECUTION = "ACTION_EXECUTION"
    STATE_TRANSITION = "STATE_TRANSITION"


@dataclass(frozen=True)
class AuditEvent:
    """Immutable structured audit record capturing decision steps and outcomes."""

    audit_id: str
    event_type: AuditEventType
    payment_id: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        event_type: AuditEventType,
        payment_id: str,
        details: Dict[str, Any],
        timestamp: datetime = None,
    ) -> "AuditEvent":
        return cls(
            audit_id=str(uuid.uuid4()),
            event_type=event_type,
            payment_id=payment_id,
            timestamp=timestamp or datetime.now(timezone.utc),
            details=details,
        )


@runtime_checkable
class AuditLogger(Protocol):
    """Protocol for structured audit event logging."""

    def log(self, event: AuditEvent) -> None:
        """Append an audit event to the ledger."""
        ...

    def get_events_for_payment(self, payment_id: str) -> List[AuditEvent]:
        """Retrieve full chronological audit trail for a payment."""
        ...


class InMemoryAuditLogger:
    """Append-only in-memory audit logger for testing and simulation contracts."""

    def __init__(self):
        self._events: List[AuditEvent] = []

    def log(self, event: AuditEvent) -> None:
        """Append event to in-memory immutable ledger."""
        self._events.append(event)

    def get_events_for_payment(self, payment_id: str) -> List[AuditEvent]:
        """Return all events for a specific payment in chronological order."""
        return [e for e in self._events if e.payment_id == payment_id]

    @property
    def all_events(self) -> List[AuditEvent]:
        """Return shallow copy of all logged events."""
        return list(self._events)
