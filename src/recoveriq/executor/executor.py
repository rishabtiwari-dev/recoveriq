"""Action Executor interfaces, execution results, and idempotency tracking."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, unique
from typing import Dict, Optional, Protocol, runtime_checkable

from recoveriq.domain.actions import Action
from recoveriq.domain.decisions import PolicyDecision
from recoveriq.domain.idempotency import (
    IdempotencyRecord,
    IdempotencyStatus,
    generate_idempotency_key,
)
from recoveriq.domain.models import Payment
from recoveriq.domain.state import PaymentState


@unique
class ExecutionStatus(str, Enum):
    """Status of an executed recovery action."""

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    SKIPPED_IDEMPOTENT = "SKIPPED_IDEMPOTENT"


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome and metadata of a dispatched recovery action."""

    payment_id: str
    action: Action
    status: ExecutionStatus
    idempotency_key: str
    resulting_state: PaymentState
    executed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: Optional[str] = None


@runtime_checkable
class ActionExecutor(Protocol):
    """Protocol for executing authorized payment recovery actions."""

    def execute(
        self,
        payment: Payment,
        policy_decision: PolicyDecision,
        event_id: str,
    ) -> ExecutionResult:
        """Execute the authorized action under strict idempotency guarantees."""
        ...


class InMemoryActionExecutor:
    """In-memory reference action executor with built-in idempotency store for Sprint 1 contracts."""

    def __init__(self):
        self._idempotency_store: Dict[str, IdempotencyRecord] = {}

    def get_idempotency_record(self, key: str) -> Optional[IdempotencyRecord]:
        """Look up existing idempotency record by key."""
        return self._idempotency_store.get(key)

    def execute(
        self,
        payment: Payment,
        policy_decision: PolicyDecision,
        event_id: str,
    ) -> ExecutionResult:
        action = policy_decision.authorized_action
        attempt_num = payment.attempt_count + 1

        # Generate idempotency key
        idem_key = generate_idempotency_key(
            payment_id=payment.payment_id,
            action=action,
            attempt_number=attempt_num,
            event_id=event_id,
        )

        # Idempotency check: if key exists and completed, return idempotent skip
        if idem_key in self._idempotency_store:
            existing = self._idempotency_store[idem_key]
            if existing.status == IdempotencyStatus.COMPLETED:
                return ExecutionResult(
                    payment_id=payment.payment_id,
                    action=action,
                    status=ExecutionStatus.SKIPPED_IDEMPOTENT,
                    idempotency_key=idem_key,
                    resulting_state=payment.state,
                    error_message="Duplicate execution prevented by idempotency gate.",
                )

        # Register in-flight record
        record = IdempotencyRecord.create(
            payment_id=payment.payment_id,
            action=action,
            attempt_number=attempt_num,
            event_id=event_id,
        )
        self._idempotency_store[idem_key] = record

        # Determine target state based on action type
        if action == Action.STOP:
            resulting_state = PaymentState.FAILED_TERMINAL
        elif action == Action.ESCALATE:
            resulting_state = PaymentState.ESCALATED
        else:
            resulting_state = PaymentState.RECOVERING

        record.mark_completed({"resulting_state": resulting_state.value})

        return ExecutionResult(
            payment_id=payment.payment_id,
            action=action,
            status=ExecutionStatus.SUCCESS,
            idempotency_key=idem_key,
            resulting_state=resulting_state,
        )
