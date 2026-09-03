"""Tests for the Payment State Machine and transition constraints."""

import pytest
from recoveriq.domain.state import (
    InvalidStateTransitionError,
    PaymentState,
    StateTransition,
    is_valid_transition,
)


def test_payment_states_completeness():
    """Verify explicit non-ambiguous state definitions."""
    expected_states = {
        "FAILED_INITIAL",
        "RECOVERING",
        "RECOVERED",
        "ESCALATED",
        "FAILED_TERMINAL",
    }
    actual_states = {s.value for s in PaymentState}
    assert actual_states == expected_states


def test_terminal_state_classification():
    """Verify terminal state helper flags."""
    assert PaymentState.RECOVERED.is_terminal is True
    assert PaymentState.ESCALATED.is_terminal is True
    assert PaymentState.FAILED_TERMINAL.is_terminal is True
    assert PaymentState.FAILED_INITIAL.is_terminal is False
    assert PaymentState.RECOVERING.is_terminal is False


@pytest.mark.parametrize(
    "from_state,to_state",
    [
        (PaymentState.FAILED_INITIAL, PaymentState.RECOVERING),
        (PaymentState.FAILED_INITIAL, PaymentState.FAILED_TERMINAL),
        (PaymentState.FAILED_INITIAL, PaymentState.ESCALATED),
        (PaymentState.RECOVERING, PaymentState.RECOVERING),
        (PaymentState.RECOVERING, PaymentState.RECOVERED),
        (PaymentState.RECOVERING, PaymentState.ESCALATED),
        (PaymentState.RECOVERING, PaymentState.FAILED_TERMINAL),
    ],
)
def test_valid_state_transitions(from_state: PaymentState, to_state: PaymentState):
    """Verify permitted forward transitions pass validation."""
    assert is_valid_transition(from_state, to_state) is True
    transition = StateTransition.create(
        from_state=from_state,
        to_state=to_state,
        trigger_event_id="evt_test_123",
        reason="Test valid transition",
    )
    assert transition.from_state == from_state
    assert transition.to_state == to_state


@pytest.mark.parametrize(
    "from_state,to_state",
    [
        (PaymentState.FAILED_INITIAL, PaymentState.RECOVERED),  # Cannot jump straight to recovered
        (PaymentState.RECOVERED, PaymentState.RECOVERING),      # Terminal state cannot transition
        (PaymentState.RECOVERED, PaymentState.FAILED_TERMINAL),
        (PaymentState.RECOVERED, PaymentState.FAILED_INITIAL),
        (PaymentState.ESCALATED, PaymentState.RECOVERING),      # Terminal state cannot transition
        (PaymentState.ESCALATED, PaymentState.RECOVERED),
        (PaymentState.FAILED_TERMINAL, PaymentState.RECOVERING),# Terminal state cannot transition
        (PaymentState.FAILED_TERMINAL, PaymentState.RECOVERED),
    ],
)
def test_invalid_state_transitions(from_state: PaymentState, to_state: PaymentState):
    """Verify illegal transitions are rejected by validator and raise InvalidStateTransitionError."""
    assert is_valid_transition(from_state, to_state) is False
    with pytest.raises(InvalidStateTransitionError) as exc_info:
        StateTransition.create(
            from_state=from_state,
            to_state=to_state,
            trigger_event_id="evt_test_invalid",
        )
    assert "Invalid state transition" in str(exc_info.value)
