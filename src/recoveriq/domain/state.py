"""Payment lifecycle state machine and transition validation."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, unique
from typing import Dict, FrozenSet, Optional


@unique
class PaymentState(str, Enum):
    """Explicit payment lifecycle states resolving naming ambiguity."""

    FAILED_INITIAL = "FAILED_INITIAL"
    RECOVERING = "RECOVERING"
    RECOVERED = "RECOVERED"
    ESCALATED = "ESCALATED"
    FAILED_TERMINAL = "FAILED_TERMINAL"

    @property
    def is_terminal(self) -> bool:
        """Returns True if the state is terminal (no further transitions allowed)."""
        return self in (
            PaymentState.RECOVERED,
            PaymentState.ESCALATED,
            PaymentState.FAILED_TERMINAL,
        )

    def __repr__(self) -> str:
        return f"PaymentState.{self.name}"


class InvalidStateTransitionError(ValueError):
    """Raised when attempting an illegal payment state transition."""

    def __init__(self, from_state: PaymentState, to_state: PaymentState, reason: str = ""):
        self.from_state = from_state
        self.to_state = to_state
        self.reason = reason
        msg = f"Invalid state transition: cannot transition from {from_state.value} to {to_state.value}."
        if reason:
            msg += f" Reason: {reason}"
        super().__init__(msg)


# Explicit mapping of permitted forward state transitions
VALID_TRANSITIONS: Dict[PaymentState, FrozenSet[PaymentState]] = {
    PaymentState.FAILED_INITIAL: frozenset(
        {
            PaymentState.RECOVERING,
            PaymentState.FAILED_TERMINAL,
            PaymentState.ESCALATED,
        }
    ),
    PaymentState.RECOVERING: frozenset(
        {
            PaymentState.RECOVERING,  # Re-attempt cycle
            PaymentState.RECOVERED,
            PaymentState.ESCALATED,
            PaymentState.FAILED_TERMINAL,
        }
    ),
    PaymentState.RECOVERED: frozenset(),  # Terminal state: no outgoing transitions
    PaymentState.ESCALATED: frozenset(),  # Terminal state: no outgoing transitions
    PaymentState.FAILED_TERMINAL: frozenset(),  # Terminal state: no outgoing transitions
}


def is_valid_transition(from_state: PaymentState, to_state: PaymentState) -> bool:
    """Check if a state transition is permitted by the state machine."""
    allowed = VALID_TRANSITIONS.get(from_state, frozenset())
    return to_state in allowed


@dataclass(frozen=True)
class StateTransition:
    """Immutable record of a payment state transition."""

    from_state: PaymentState
    to_state: PaymentState
    timestamp: datetime
    trigger_event_id: str
    reason: Optional[str] = None

    @classmethod
    def create(
        cls,
        from_state: PaymentState,
        to_state: PaymentState,
        trigger_event_id: str,
        reason: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> "StateTransition":
        if not is_valid_transition(from_state, to_state):
            raise InvalidStateTransitionError(from_state, to_state, reason or "Disallowed by FSM rules")
        return cls(
            from_state=from_state,
            to_state=to_state,
            timestamp=timestamp or datetime.now(timezone.utc),
            trigger_event_id=trigger_event_id,
            reason=reason,
        )
