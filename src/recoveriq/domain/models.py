"""Core domain models for payment transactions and context representations."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum, unique
from typing import Any, Dict, Optional

from recoveriq.domain.state import PaymentState


@unique
class CustomerTier(str, Enum):
    """Customer loyalty / value segmentation tier."""

    STANDARD = "STANDARD"
    PREMIUM = "PREMIUM"
    VIP = "VIP"
    NEW = "NEW"


@unique
class PaymentMethod(str, Enum):
    """Payment instrument / rail."""

    CREDIT_CARD = "CREDIT_CARD"
    DEBIT_CARD = "DEBIT_CARD"
    UPI = "UPI"
    NET_BANKING = "NET_BANKING"
    WALLET = "WALLET"


@unique
class FailureCategory(str, Enum):
    """Standardized failure taxonomy extracted from raw decline signals."""

    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    NETWORK_TIMEOUT = "NETWORK_TIMEOUT"
    CARD_EXPIRED = "CARD_EXPIRED"
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    AUTHENTICATION_REJECTED = "AUTHENTICATION_REJECTED"
    INVALID_DETAILS = "INVALID_DETAILS"
    HARD_DECLINE = "HARD_DECLINE"
    VELOCITY_EXCEEDED = "VELOCITY_EXCEEDED"
    UNKNOWN = "UNKNOWN"

    @property
    def is_hard_decline(self) -> bool:
        """Indicates whether this failure category represents an irrecoverable hard decline."""
        return self in (FailureCategory.HARD_DECLINE, FailureCategory.INVALID_DETAILS)


@unique
class FailureSeverity(str, Enum):
    """Failure persistence categorization."""

    TRANSIENT = "TRANSIENT"       # e.g., Network timeout, switch reset
    RECOVERABLE = "RECOVERABLE"   # e.g., Balance replenishment needed, link update
    STRUCTURAL = "STRUCTURAL"     # e.g., Expired card, limits exceeded
    FATAL = "FATAL"               # e.g., Stolen card, closed account


@dataclass(frozen=True)
class PaymentContext:
    """Structured payment context extracted from unstructured error payloads and metadata."""

    payment_id: str
    customer_id: str
    customer_tier: CustomerTier
    payment_method: PaymentMethod
    raw_error_code: str
    raw_error_message: str
    failure_category: FailureCategory
    failure_severity: FailureSeverity
    attempt_count: int = 0
    last_attempt_timestamp: Optional[datetime] = None
    diagnostic_explanation: Optional[str] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Payment:
    """Payment domain entity tracking lifecycle and state."""

    payment_id: str
    customer_id: str
    amount: Decimal
    currency: str = "USD"
    state: PaymentState = PaymentState.FAILED_INITIAL
    attempt_count: int = 0
    last_event_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.amount, Decimal):
            self.amount = Decimal(str(self.amount))
        if self.amount <= Decimal("0"):
            raise ValueError(f"Payment amount must be strictly positive, got: {self.amount}")

    @property
    def is_terminal(self) -> bool:
        """Returns True if the payment is in a terminal state."""
        return self.state.is_terminal
