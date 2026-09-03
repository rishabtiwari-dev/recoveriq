"""Tests for the RecoverIQ Action domain space."""

import pytest
from recoveriq.domain.actions import Action


def test_action_space_completeness():
    """Verify that the action space matches SPEC.md exactly."""
    expected_actions = {
        "RETRY_NOW",
        "RETRY_LATER",
        "SEND_LINK",
        "NUDGE",
        "ESCALATE",
        "STOP",
    }
    actual_actions = {a.value for a in Action}
    assert actual_actions == expected_actions
    assert len(Action) == 6


def test_action_properties():
    """Verify helper classification properties on Action enum."""
    assert Action.RETRY_NOW.is_retry is True
    assert Action.RETRY_LATER.is_retry is True
    assert Action.SEND_LINK.is_retry is False
    assert Action.NUDGE.is_retry is False
    assert Action.ESCALATE.is_retry is False
    assert Action.STOP.is_retry is False

    assert Action.STOP.is_terminal_stop is True
    assert Action.RETRY_NOW.is_terminal_stop is False

    assert Action.ESCALATE.is_escalation is True
    assert Action.NUDGE.is_escalation is False
