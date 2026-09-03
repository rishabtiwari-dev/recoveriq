"""Tests for the Deterministic Policy Gate."""

from decimal import Decimal
from recoveriq.config.settings import PolicyConfig
from recoveriq.domain.actions import Action
from recoveriq.domain.decisions import RecoveryDecision
from recoveriq.domain.models import (
    CustomerTier,
    FailureCategory,
    FailureSeverity,
    Payment,
    PaymentContext,
    PaymentMethod,
)
from recoveriq.domain.state import PaymentState
from recoveriq.policy.gate import InvariantPolicyGate


def _make_context(
    category: FailureCategory = FailureCategory.NETWORK_TIMEOUT,
    tier: CustomerTier = CustomerTier.STANDARD,
) -> PaymentContext:
    return PaymentContext(
        payment_id="pay_pol_1",
        customer_id="cust_pol_1",
        customer_tier=tier,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="TEST_CODE",
        raw_error_message="Test message",
        failure_category=category,
        failure_severity=FailureSeverity.RECOVERABLE,
    )


def test_policy_authorizes_valid_action():
    """Verify standard valid proposed action is approved."""
    gate = InvariantPolicyGate()
    payment = Payment(payment_id="pay_pol_1", customer_id="cust_pol_1", amount=Decimal("100.00"))
    context = _make_context(FailureCategory.NETWORK_TIMEOUT)
    decision = RecoveryDecision("pay_pol_1", Action.RETRY_NOW)

    policy_decision = gate.authorize(payment, context, decision)
    assert policy_decision.is_authorized is True
    assert policy_decision.authorized_action == Action.RETRY_NOW
    assert policy_decision.rejection_reason is None


def test_policy_rejects_terminal_state():
    """Verify gate rejects any action on an already-terminal payment."""
    gate = InvariantPolicyGate()
    payment = Payment(
        payment_id="pay_pol_2",
        customer_id="cust_pol_2",
        amount=Decimal("100.00"),
        state=PaymentState.RECOVERED,
    )
    context = _make_context()
    decision = RecoveryDecision("pay_pol_2", Action.RETRY_NOW)

    policy_decision = gate.authorize(payment, context, decision)
    assert policy_decision.is_authorized is False
    assert policy_decision.authorized_action == Action.STOP
    assert "terminal" in policy_decision.rejection_reason.lower()


def test_policy_rejects_retry_on_hard_decline():
    """Verify gate forbids retry actions when failure is a hard decline."""
    gate = InvariantPolicyGate()
    payment = Payment(payment_id="pay_pol_3", customer_id="cust_pol_3", amount=Decimal("100.00"))
    context = _make_context(FailureCategory.HARD_DECLINE)
    decision = RecoveryDecision("pay_pol_3", Action.RETRY_NOW)

    policy_decision = gate.authorize(payment, context, decision)
    assert policy_decision.is_authorized is False
    assert policy_decision.authorized_action == Action.STOP
    assert "hard decline" in policy_decision.rejection_reason.lower()


def test_policy_rejects_when_retry_budget_exhausted():
    """Verify gate enforces max_attempts limit."""
    config = PolicyConfig(max_attempts=3)
    gate = InvariantPolicyGate(config)
    payment = Payment(
        payment_id="pay_pol_4",
        customer_id="cust_pol_4",
        amount=Decimal("100.00"),
        attempt_count=3,  # Already at limit
    )
    context = _make_context(FailureCategory.NETWORK_TIMEOUT, CustomerTier.STANDARD)
    decision = RecoveryDecision("pay_pol_4", Action.RETRY_NOW)

    policy_decision = gate.authorize(payment, context, decision)
    assert policy_decision.is_authorized is False
    assert policy_decision.authorized_action == Action.STOP
    assert "budget" in policy_decision.rejection_reason.lower()


def test_policy_escalates_vip_on_budget_exhaustion():
    """Verify VIP customer is escalated when retry budget is exhausted."""
    config = PolicyConfig(max_attempts=3, vip_escalation_enabled=True)
    gate = InvariantPolicyGate(config)
    payment = Payment(
        payment_id="pay_pol_5",
        customer_id="cust_pol_5",
        amount=Decimal("500.00"),
        attempt_count=3,
    )
    context = _make_context(FailureCategory.NETWORK_TIMEOUT, CustomerTier.VIP)
    decision = RecoveryDecision("pay_pol_5", Action.RETRY_NOW)

    policy_decision = gate.authorize(payment, context, decision)
    assert policy_decision.is_authorized is False
    assert policy_decision.authorized_action == Action.ESCALATE
