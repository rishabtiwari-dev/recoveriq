"""Idempotency domain contracts and deterministic key calculation."""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, unique
from typing import Any, Dict, Optional

from recoveriq.domain.actions import Action


@unique
class IdempotencyStatus(str, Enum):
    """Lifecycle status of an idempotent action execution."""

    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


def generate_idempotency_key(
    payment_id: str,
    action: Action,
    attempt_number: int,
    event_id: str,
) -> str:
    """Generate deterministic SHA-256 idempotency key as defined in SPEC.md.

    IdempotencyKey = SHA256(payment_id:action_type:attempt_number:event_id)
    """
    raw_key = f"{payment_id}:{action.value}:{attempt_number}:{event_id}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


@dataclass
class IdempotencyRecord:
    """Record tracking an executed or in-flight action to prevent duplicate execution."""

    key: str
    payment_id: str
    action: Action
    attempt_number: int
    event_id: str
    status: IdempotencyStatus = IdempotencyStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    response_payload: Optional[Dict[str, Any]] = None

    @classmethod
    def create(
        cls,
        payment_id: str,
        action: Action,
        attempt_number: int,
        event_id: str,
    ) -> "IdempotencyRecord":
        key = generate_idempotency_key(payment_id, action, attempt_number, event_id)
        return cls(
            key=key,
            payment_id=payment_id,
            action=action,
            attempt_number=attempt_number,
            event_id=event_id,
        )

    def mark_completed(self, response: Optional[Dict[str, Any]] = None) -> None:
        """Mark record as successfully completed."""
        self.status = IdempotencyStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        self.response_payload = response

    def mark_failed(self, error_details: Optional[Dict[str, Any]] = None) -> None:
        """Mark record as failed."""
        self.status = IdempotencyStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.response_payload = error_details
