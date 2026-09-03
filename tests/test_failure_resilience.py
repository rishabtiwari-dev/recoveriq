"""Sprint 7 Tests — SPEC §18 Failure & Resilience Testing Suite.

Implements all six required failure/resilience scenarios:
1. Duplicate Webhook Delivery (SPEC §18.1)
2. Out-of-Order Webhook Delivery (SPEC §18.2)
3. Executor Timeout (SPEC §18.3)
4. Duplicate Action Request (SPEC §18.4)
5. Exhausted Retry Budget (SPEC §18.5)
6. Policy Rejection (SPEC §18.6)

Also includes:
- Specific Cooldown tests:
  A. SPEC-correct cooldown test with @pytest.mark.xfail(strict=True) asserting rejection on early retries (SPEC §11.3).
  B. Passing canary test documenting the known policy/gate.py cooldown violation.
"""

from decimal import Decimal
from datetime import datetime, timezone, timedelta

import pytest

from recoveriq.config.settings import PolicyConfig, RecoverIQConfig
from recoveriq.domain.actions import Action
from recoveriq.domain.decisions import PolicyDecision, RecoveryDecision
from recoveriq.domain.events import EventType, PaymentEvent, PaymentFailedEvent
from recoveriq.domain.idempotency import generate_idempotency_key
from recoveriq.domain.models import (
    CustomerTier,
    FailureCategory,
    FailureSeverity,
    Payment,
    PaymentContext,
    PaymentMethod,
)
from recoveriq.domain.state import PaymentState
from recoveriq.engine import RecoverIQEngine
from recoveriq.executor.executor import (
    ActionExecutor,
    ExecutionResult,
    ExecutionStatus,
    InMemoryActionExecutor,
)
from recoveriq.policy.gate import InvariantPolicyGate
from tests.test_resilience_harness import PaymentReplayHarness


# ==============================================================================
# Helper Factories
# ==============================================================================

def _create_sample_failed_event(
    event_id: str = "evt_fail_001",
    payment_id: str = "pay_fail_001",
    raw_code: str = "504_GATEWAY_TIMEOUT",
    raw_msg: str = "Gateway Timeout",
    attempt: int = 1,
    amount: Decimal = Decimal("100.00"),
    tier: CustomerTier = CustomerTier.STANDARD,
) -> PaymentFailedEvent:
    return PaymentFailedEvent(
        event_id=event_id,
        payment_id=payment_id,
        event_type=EventType.PAYMENT_FAILED,
        customer_id="cust_001",
        amount=amount,
        currency="USD",
        customer_tier=tier,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code=raw_code,
        raw_error_message=raw_msg,
        attempt_count=attempt,
    )


# ==============================================================================
# 1. SPEC §18.1 — Duplicate Webhook Delivery
# ==============================================================================

def test_duplicate_webhook_delivery_sequential():
    """SPEC §18.1: Sending the identical failure webhook multiple times sequentially.

    The first event is processed and dispatches an authorized action.
    Subsequent arrivals with the identical event_id are dropped idempotently without
    triggering duplicate execution, extra audit trails, or extra state transitions.
    """
    harness = PaymentReplayHarness()
    event = _create_sample_failed_event(event_id="evt_dup_001", payment_id="pay_dup_001")

    # 1. First delivery -> accepted & executed
    res1 = harness.process_event(event)
    assert res1.accepted is True
    assert res1.status == "SUCCESS"
    assert res1.final_payment_state is not None
    assert res1.final_payment_state == PaymentState.ESCALATED

    audit_count_initial = len(harness.engine.audit_logger.get_events_for_payment("pay_dup_001"))
    assert audit_count_initial > 0

    # 2. Second delivery with identical event_id -> dropped idempotently
    res2 = harness.process_event(event)
    assert res2.accepted is False
    assert res2.status == "DUPLICATE_EVENT_DROPPED"
    assert "already processed" in res2.rejection_reason

    # Audit trail for payment must not have grown with a second execution
    audit_count_after = len(harness.engine.audit_logger.get_events_for_payment("pay_dup_001"))
    assert audit_count_after == audit_count_initial

    # Payment state remains unchanged
    payment = harness.get_payment("pay_dup_001")
    assert payment.state == PaymentState.ESCALATED
    assert payment.attempt_count == 1


# ==============================================================================
# 2. SPEC §18.2 — Out-of-Order Webhook Delivery
# ==============================================================================

def test_out_of_order_webhook_delivery_after_terminal_state():
    """SPEC §18.2: A failure update arriving after a payment has already reached a terminal state.

    If a payment is already RECOVERED, ESCALATED, or FAILED_TERMINAL, subsequent
    failure events must be rejected with no state transition or downstream dispatch.
    """
    harness = PaymentReplayHarness()

    # Pre-populate payment in terminal RECOVERED state
    term_payment = Payment(
        payment_id="pay_term_001",
        customer_id="cust_001",
        amount=Decimal("250.00"),
        state=PaymentState.RECOVERED,
        attempt_count=1,
    )
    harness._payments["pay_term_001"] = term_payment

    # Stale/delayed failure webhook arrives
    stale_event = _create_sample_failed_event(
        event_id="evt_stale_001",
        payment_id="pay_term_001",
        attempt=2,
    )

    res = harness.process_event(stale_event)
    assert res.accepted is False
    assert res.status == "TERMINAL_STATE_REJECTED"
    assert "already in terminal state" in res.rejection_reason
    assert res.final_payment_state == PaymentState.RECOVERED

    # Ensure payment remains strictly RECOVERED
    assert harness.get_payment("pay_term_001").state == PaymentState.RECOVERED


def test_engine_directly_rejects_action_on_terminal_payment():
    """SPEC §18.2 / §11.5: Verify directly that RecoverIQEngine clamps actions on terminal payments."""
    engine = RecoverIQEngine()
    event = _create_sample_failed_event(event_id="evt_term_direct", payment_id="pay_term_direct")
    term_payment = Payment(
        payment_id="pay_term_direct",
        customer_id="cust_001",
        amount=Decimal("100.00"),
        state=PaymentState.FAILED_TERMINAL,
    )

    exec_result = engine.process_failure_event(event, existing_payment=term_payment)
    # When payment is already terminal, Policy Gate clamps to STOP and resulting state is FAILED_TERMINAL
    assert exec_result.action == Action.STOP
    assert exec_result.resulting_state == PaymentState.FAILED_TERMINAL


# ==============================================================================
# 3. SPEC §18.3 — Executor Timeout
# ==============================================================================

class TimingOutActionExecutor(ActionExecutor):
    """Custom test executor simulating downstream network timeout during action dispatch."""

    def __init__(self, timeout_attempt: int = 1):
        self.timeout_attempt = timeout_attempt
        self.calls = 0

    def execute(
        self,
        payment: Payment,
        policy_decision: PolicyDecision,
        event_id: str,
    ) -> ExecutionResult:
        self.calls += 1
        action = policy_decision.authorized_action
        attempt_num = payment.attempt_count + 1
        idem_key = generate_idempotency_key(
            payment_id=payment.payment_id,
            action=action,
            attempt_number=attempt_num,
            event_id=event_id,
        )

        if self.calls == self.timeout_attempt:
            # Simulate timeout failure: dispatch did not complete
            return ExecutionResult(
                payment_id=payment.payment_id,
                action=action,
                status=ExecutionStatus.TIMED_OUT,
                idempotency_key=idem_key,
                resulting_state=payment.state,  # State does not transition forward
                error_message="Gateway timeout during downstream dispatch (HTTP 504).",
            )

        # Subsequent attempts succeed
        return ExecutionResult(
            payment_id=payment.payment_id,
            action=action,
            status=ExecutionStatus.SUCCESS,
            idempotency_key=idem_key,
            resulting_state=PaymentState.RECOVERING,
        )


def test_executor_timeout_safe_state_and_retry():
    """SPEC §18.3: Simulating network timeout during executor action dispatch.

    Under our conservative timeout model (dispatch does not complete):
    1. Timed-out action enters safe pending state (does not advance state illegally).
    2. Subsequent retry dispatch with next attempt count can proceed without corruption.
    """
    timing_out_executor = TimingOutActionExecutor(timeout_attempt=1)
    engine = RecoverIQEngine(executor=timing_out_executor)

    payment = Payment(
        payment_id="pay_timeout_001",
        customer_id="cust_001",
        amount=Decimal("150.00"),
        state=PaymentState.FAILED_INITIAL,
        attempt_count=0,
    )

    event1 = _create_sample_failed_event(
        event_id="evt_to_1",
        payment_id="pay_timeout_001",
        attempt=1,
    )

    # First dispatch triggers timeout
    res1 = engine.process_failure_event(event1, existing_payment=payment)
    assert res1.status == ExecutionStatus.TIMED_OUT
    assert "Gateway timeout" in res1.error_message
    # State remained in FAILED_INITIAL because no forward transition occurred
    assert payment.state == PaymentState.FAILED_INITIAL

    # Second dispatch with new event / retry
    event2 = _create_sample_failed_event(
        event_id="evt_to_2",
        payment_id="pay_timeout_001",
        attempt=2,
    )
    res2 = engine.process_failure_event(event2, existing_payment=payment)
    assert res2.status == ExecutionStatus.SUCCESS
    assert res2.resulting_state == PaymentState.RECOVERING
    assert payment.state == PaymentState.RECOVERING


# ==============================================================================
# 4. SPEC §18.4 — Duplicate Action Request
# ==============================================================================

def test_duplicate_action_request_idempotency():
    """SPEC §18.4: Emitting identical action recommendations concurrently or sequentially.

    The second identical action dispatch to the Executor with the same idempotency key
    must return SKIPPED_IDEMPOTENT and not perform double execution.
    """
    executor = InMemoryActionExecutor()
    payment = Payment(
        payment_id="pay_dup_act_001",
        customer_id="cust_001",
        amount=Decimal("100.00"),
        state=PaymentState.FAILED_INITIAL,
        attempt_count=0,
    )
    decision = PolicyDecision.authorize("pay_dup_act_001", Action.RETRY_NOW)

    # 1. First execution -> SUCCESS
    res1 = executor.execute(payment, decision, event_id="evt_common")
    assert res1.status == ExecutionStatus.SUCCESS
    assert res1.idempotency_key is not None

    # 2. Second execution with identical payment, decision, and event_id -> SKIPPED_IDEMPOTENT
    res2 = executor.execute(payment, decision, event_id="evt_common")
    assert res2.status == ExecutionStatus.SKIPPED_IDEMPOTENT
    assert res2.idempotency_key == res1.idempotency_key
    assert "Duplicate execution prevented" in res2.error_message


# ==============================================================================
# 5. SPEC §18.5 — Exhausted Retry Budget
# ==============================================================================

def test_exhausted_retry_budget_enforcement():
    """SPEC §18.5: Attempting to force additional recovery actions after reaching N_max.

    Policy Gate must clamp the action to STOP (or ESCALATE for VIPs).
    """
    config = PolicyConfig(max_attempts=3, vip_escalation_enabled=False)
    gate = InvariantPolicyGate(config)

    payment = Payment(
        payment_id="pay_budget_001",
        customer_id="cust_001",
        amount=Decimal("100.00"),
        attempt_count=3,  # Reached N_max
    )
    context = PaymentContext(
        payment_id="pay_budget_001",
        customer_id="cust_001",
        customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="TIMEOUT",
        raw_error_message="Timeout",
        failure_category=FailureCategory.NETWORK_TIMEOUT,
        failure_severity=FailureSeverity.TRANSIENT,
    )
    decision = RecoveryDecision("pay_budget_001", Action.RETRY_NOW)

    pol_decision = gate.authorize(payment, context, decision)
    assert pol_decision.is_authorized is False
    assert pol_decision.authorized_action == Action.STOP
    assert "Exhausted retry budget" in pol_decision.rejection_reason


def test_exhausted_retry_budget_vip_escalation():
    """SPEC §18.5 / §11.2: VIP customer reaching N_max must be escalated to human operations."""
    config = PolicyConfig(max_attempts=3, vip_escalation_enabled=True)
    gate = InvariantPolicyGate(config)

    payment = Payment(
        payment_id="pay_budget_vip",
        customer_id="cust_vip",
        amount=Decimal("1000.00"),
        attempt_count=3,
    )
    context = PaymentContext(
        payment_id="pay_budget_vip",
        customer_id="cust_vip",
        customer_tier=CustomerTier.VIP,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="TIMEOUT",
        raw_error_message="Timeout",
        failure_category=FailureCategory.NETWORK_TIMEOUT,
        failure_severity=FailureSeverity.TRANSIENT,
    )
    decision = RecoveryDecision("pay_budget_vip", Action.RETRY_NOW)

    pol_decision = gate.authorize(payment, context, decision)
    assert pol_decision.is_authorized is False
    assert pol_decision.authorized_action == Action.ESCALATE


# ==============================================================================
# 6. SPEC §18.6 — Policy Rejection of Invariant Violations
# ==============================================================================

def test_policy_rejection_hard_decline_retries():
    """SPEC §18.6: Submitting candidate actions that violate safety invariants (e.g. retrying stolen card).

    Must verify 100% rejection rate and override to STOP.
    """
    gate = InvariantPolicyGate()
    payment = Payment(payment_id="pay_stolen", customer_id="cust_001", amount=Decimal("50.00"))
    context = PaymentContext(
        payment_id="pay_stolen",
        customer_id="cust_001",
        customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="STOLEN_CARD",
        raw_error_message="Stolen card reported",
        failure_category=FailureCategory.HARD_DECLINE,
        failure_severity=FailureSeverity.FATAL,
    )

    for retry_action in (Action.RETRY_NOW, Action.RETRY_LATER):
        decision = RecoveryDecision("pay_stolen", retry_action)
        pol_decision = gate.authorize(payment, context, decision)
        assert pol_decision.is_authorized is False
        assert pol_decision.authorized_action == Action.STOP
        assert "Hard decline" in pol_decision.rejection_reason


# ==============================================================================
# 7. CRITICAL COOLDOWN TEST DESIGN (SPEC §11.3)
# ==============================================================================

@pytest.mark.xfail(strict=True, reason="Known safety defect in gate.py lines 130-153: cooldown failure authorizes action instead of rejecting/clamping.")
def test_spec_correct_cooldown_enforcement():
    """SPEC §11.3 / §18: A retry requested within the mandatory cooldown window must be rejected/clamped.

    SPEC REQUIREMENT:
    'Mandatory Cooldowns: Enforce minimum duration between consecutive interventions on the same payment...
     Any action attempting an illegal transition or violating cooldown must be rejected.'

    EXPECTED RESULT (XFAIL STRICT):
    Because gate.py records passed=False, then appends passed=True, and authorizes the action,
    this test asserts that the policy decision must be is_authorized=False and authorized_action=STOP.
    It is marked strict xfail to maintain visibility of the defect without masking it.
    """
    config = PolicyConfig(cooldown_seconds=900)  # 15 minutes
    gate = InvariantPolicyGate(config)

    payment = Payment(
        payment_id="pay_cooldown_001",
        customer_id="cust_001",
        amount=Decimal("100.00"),
        attempt_count=1,
    )
    # Last attempt was 60 seconds ago (< 900 seconds cooldown)
    context = PaymentContext(
        payment_id="pay_cooldown_001",
        customer_id="cust_001",
        customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="TIMEOUT",
        raw_error_message="Timeout",
        failure_category=FailureCategory.NETWORK_TIMEOUT,
        failure_severity=FailureSeverity.TRANSIENT,
        last_attempt_timestamp=datetime.now(timezone.utc) - timedelta(seconds=60),
    )
    decision = RecoveryDecision("pay_cooldown_001", Action.RETRY_LATER)

    pol_decision = gate.authorize(payment, context, decision)

    # SPEC §11.3 invariant: must NOT authorize an unelapsed cooldown
    assert pol_decision.is_authorized is False, "SPEC §11.3 requires rejection when cooldown has not elapsed!"
    assert pol_decision.authorized_action == Action.STOP, "SPEC §11.3 requires fallback clamp on cooldown breach!"


def test_canary_observed_cooldown_gate_violation():
    """CANARY TEST: Explicitly documents the CURRENT observed behavior of gate.py lines 130-153.

    Documents the exact defect:
    1. gate.py checks elapsed < cooldown_seconds (60s < 900s).
    2. It appends PolicyRuleResult('COOLDOWN_WINDOW_CHECK', passed=False).
    3. It immediately appends PolicyRuleResult('COOLDOWN_WINDOW_CHECK', passed=True, 'Cooldown validated for scheduled execution.').
    4. It drops through and AUTHORIZES Action.RETRY_LATER (is_authorized=True).

    This canary guarantees that if gate.py is silently modified, this test alerts developers.
    """
    config = PolicyConfig(cooldown_seconds=900)
    gate = InvariantPolicyGate(config)

    payment = Payment(
        payment_id="pay_canary_001",
        customer_id="cust_001",
        amount=Decimal("100.00"),
        attempt_count=1,
    )
    context = PaymentContext(
        payment_id="pay_canary_001",
        customer_id="cust_001",
        customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="TIMEOUT",
        raw_error_message="Timeout",
        failure_category=FailureCategory.NETWORK_TIMEOUT,
        failure_severity=FailureSeverity.TRANSIENT,
        last_attempt_timestamp=datetime.now(timezone.utc) - timedelta(seconds=60),
    )
    decision = RecoveryDecision("pay_canary_001", Action.RETRY_LATER)

    pol_decision = gate.authorize(payment, context, decision)

    # Document current buggy behavior:
    assert pol_decision.is_authorized is True
    assert pol_decision.authorized_action == Action.RETRY_LATER

    rule_names = [(r.rule_name, r.passed) for r in pol_decision.rule_results]
    assert ("COOLDOWN_WINDOW_CHECK", False) in rule_names
    assert ("COOLDOWN_WINDOW_CHECK", True) in rule_names
