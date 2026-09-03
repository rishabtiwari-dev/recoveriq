"""Payment lifecycle and recovery event definitions."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum, unique
from typing import Any, Dict, Optional

from recoveriq.domain.actions import Action
from recoveriq.domain.models import CustomerTier, PaymentMethod
from recoveriq.domain.state import PaymentState


@unique
class EventType(str, Enum):
    """Payment event categories across the recovery lifecycle."""

    PAYMENT_FAILED = "PAYMENT_FAILED"
    RECOVERY_PROPOSED = "RECOVERY_PROPOSED"
    POLICY_EVALUATED = "POLICY_EVALUATED"
    ACTION_DISPATCHED = "ACTION_DISPATCHED"
    OUTCOME_RECEIVED = "OUTCOME_RECEIVED"
    STATE_TRANSITIONED = "STATE_TRANSITIONED"


@dataclass(frozen=True)
class PaymentEvent:
    """Base event model for payment lifecycle tracking."""

    event_id: str
    payment_id: str
    event_type: EventType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PaymentFailedEvent(PaymentEvent):
    """Ingested payment failure event (e.g. from gateway webhook or transaction pipeline)."""

    customer_id: str = ""
    amount: Decimal = Decimal("0.00")
    currency: str = "USD"
    customer_tier: CustomerTier = CustomerTier.STANDARD
    payment_method: PaymentMethod = PaymentMethod.CREDIT_CARD
    raw_error_code: str = ""
    raw_error_message: str = ""
    attempt_count: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.amount, Decimal):
            object.__setattr__(self, "amount", Decimal(str(self.amount)))


@dataclass(frozen=True)
class OutcomeReceivedEvent(PaymentEvent):
    """Event carrying the resolution outcome of an executed recovery action."""

    action: Action = Action.STOP
    success: bool = False
    resulting_state: PaymentState = PaymentState.FAILED_TERMINAL
    error_code: Optional[str] = None
    error_message: Optional[str] = None
