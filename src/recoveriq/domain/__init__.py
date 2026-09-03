"""Domain package for RecoverIQ."""

from recoveriq.domain.actions import Action
from recoveriq.domain.decisions import (
    CandidateActionEV,
    PolicyDecision,
    PolicyRuleResult,
    RecoveryDecision,
)
from recoveriq.domain.events import (
    EventType,
    OutcomeReceivedEvent,
    PaymentEvent,
    PaymentFailedEvent,
)
from recoveriq.domain.idempotency import (
    IdempotencyRecord,
    IdempotencyStatus,
    generate_idempotency_key,
)
from recoveriq.domain.models import (
    CustomerTier,
    FailureCategory,
    FailureSeverity,
    Payment,
    PaymentContext,
    PaymentMethod,
)
from recoveriq.domain.state import (
    InvalidStateTransitionError,
    PaymentState,
    StateTransition,
    is_valid_transition,
)

__all__ = [
    "Action",
    "CandidateActionEV",
    "CustomerTier",
    "EventType",
    "FailureCategory",
    "FailureSeverity",
    "IdempotencyRecord",
    "IdempotencyStatus",
    "InvalidStateTransitionError",
    "OutcomeReceivedEvent",
    "Payment",
    "PaymentContext",
    "PaymentEvent",
    "PaymentFailedEvent",
    "PaymentMethod",
    "PaymentState",
    "PolicyDecision",
    "PolicyRuleResult",
    "RecoveryDecision",
    "StateTransition",
    "generate_idempotency_key",
    "is_valid_transition",
]
