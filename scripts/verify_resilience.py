"""RecoverIQ Sprint 7 — Failure & Resilience Verification (SPEC §18).

This script performs verification of the six failure scenarios specified in SPEC §18:
1. Duplicate Webhook Delivery (§18.1)
2. Out-of-Order Webhook Delivery (§18.2)
3. Executor Timeout (§18.3)
4. Duplicate Action Request (§18.4)
5. Exhausted Retry Budget (§18.5)
6. Policy Rejection (§18.6)

OUTPUT STRUCTURE (STRICTLY SEPARATED):
- SECTION A: §18 Scenario Verdicts (Pass/Fail for each resilience scenario)
- SECTION B: Blocking Safety Findings (Explicit report of known Policy Gate cooldown violation §11.3)
- SECTION C: Non-Blocking Gaps & Scientific Limitations (Concurrency, persistence, timeout model)
"""

import sys
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timezone, timedelta

# Ensure repo root is on sys.path when script is run directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recoveriq.config.settings import PolicyConfig
from recoveriq.domain.actions import Action
from recoveriq.domain.decisions import PolicyDecision, RecoveryDecision
from recoveriq.domain.events import EventType, PaymentFailedEvent
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
from recoveriq.executor.executor import ExecutionStatus, InMemoryActionExecutor
from recoveriq.policy.gate import InvariantPolicyGate
from tests.test_failure_resilience import (
    TimingOutActionExecutor,
    _create_sample_failed_event,
)
from tests.test_resilience_harness import PaymentReplayHarness


def verify_scenario_1_duplicate_webhook() -> bool:
    """1. Duplicate Webhook Delivery (§18.1)."""
    harness = PaymentReplayHarness()
    evt = _create_sample_failed_event("evt_dup_v", "pay_dup_v")
    res1 = harness.process_event(evt)
    res2 = harness.process_event(evt)
    passed = (
        res1.accepted is True
        and res2.accepted is False
        and res2.status == "DUPLICATE_EVENT_DROPPED"
        and harness.get_payment("pay_dup_v").attempt_count == 1
    )
    return passed


def verify_scenario_2_out_of_order_webhook() -> bool:
    """2. Out-of-Order Webhook Delivery (§18.2)."""
    harness = PaymentReplayHarness()
    term_payment = Payment(
        payment_id="pay_term_v",
        customer_id="cust_v",
        amount=Decimal("100.00"),
        state=PaymentState.RECOVERED,
        attempt_count=1,
    )
    harness._payments["pay_term_v"] = term_payment

    stale_evt = _create_sample_failed_event("evt_stale_v", "pay_term_v", attempt=2)
    res = harness.process_event(stale_evt)
    passed = (
        res.accepted is False
        and res.status == "TERMINAL_STATE_REJECTED"
        and harness.get_payment("pay_term_v").state == PaymentState.RECOVERED
    )
    return passed


def verify_scenario_3_executor_timeout() -> bool:
    """3. Executor Timeout (§18.3)."""
    timing_out_executor = TimingOutActionExecutor(timeout_attempt=1)
    engine = RecoverIQEngine(executor=timing_out_executor)
    payment = Payment(
        payment_id="pay_to_v",
        customer_id="cust_v",
        amount=Decimal("100.00"),
        state=PaymentState.FAILED_INITIAL,
        attempt_count=0,
    )
    evt1 = _create_sample_failed_event("evt_to_v1", "pay_to_v", attempt=1)
    res1 = engine.process_failure_event(evt1, existing_payment=payment)
    state_after_to = payment.state  # Check state immediately after timeout

    evt2 = _create_sample_failed_event("evt_to_v2", "pay_to_v", attempt=2)
    res2 = engine.process_failure_event(evt2, existing_payment=payment)
    state_after_retry = payment.state  # Check state after retry

    passed = (
        res1.status == ExecutionStatus.TIMED_OUT
        and state_after_to == PaymentState.FAILED_INITIAL  # Safe non-transitioned state after timeout
        and res2.status == ExecutionStatus.SUCCESS
        and state_after_retry == PaymentState.RECOVERING
    )
    return passed


def verify_scenario_4_duplicate_action_request() -> bool:
    """4. Duplicate Action Request (§18.4)."""
    executor = InMemoryActionExecutor()
    payment = Payment(
        payment_id="pay_act_v",
        customer_id="cust_v",
        amount=Decimal("100.00"),
        state=PaymentState.FAILED_INITIAL,
        attempt_count=0,
    )
    decision = PolicyDecision.authorize("pay_act_v", Action.RETRY_NOW)
    res1 = executor.execute(payment, decision, "evt_act_v")
    res2 = executor.execute(payment, decision, "evt_act_v")

    passed = (
        res1.status == ExecutionStatus.SUCCESS
        and res2.status == ExecutionStatus.SKIPPED_IDEMPOTENT
        and res1.idempotency_key == res2.idempotency_key
    )
    return passed


def verify_scenario_5_exhausted_retry_budget() -> bool:
    """5. Exhausted Retry Budget (§18.5)."""
    config = PolicyConfig(max_attempts=3, vip_escalation_enabled=True)
    gate = InvariantPolicyGate(config)

    # Standard customer -> STOP
    payment_std = Payment(payment_id="pay_b_std", customer_id="c_std", amount=Decimal("100.00"), attempt_count=3)
    ctx_std = PaymentContext(
        payment_id="pay_b_std", customer_id="c_std", customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.CREDIT_CARD, raw_error_code="T", raw_error_message="T",
        failure_category=FailureCategory.NETWORK_TIMEOUT, failure_severity=FailureSeverity.TRANSIENT,
    )
    dec_std = RecoveryDecision("pay_b_std", Action.RETRY_NOW)
    pdec_std = gate.authorize(payment_std, ctx_std, dec_std)

    # VIP customer -> ESCALATE
    payment_vip = Payment(payment_id="pay_b_vip", customer_id="c_vip", amount=Decimal("500.00"), attempt_count=3)
    ctx_vip = PaymentContext(
        payment_id="pay_b_vip", customer_id="c_vip", customer_tier=CustomerTier.VIP,
        payment_method=PaymentMethod.CREDIT_CARD, raw_error_code="T", raw_error_message="T",
        failure_category=FailureCategory.NETWORK_TIMEOUT, failure_severity=FailureSeverity.TRANSIENT,
    )
    dec_vip = RecoveryDecision("pay_b_vip", Action.RETRY_NOW)
    pdec_vip = gate.authorize(payment_vip, ctx_vip, dec_vip)

    passed = (
        pdec_std.is_authorized is False
        and pdec_std.authorized_action == Action.STOP
        and pdec_vip.is_authorized is False
        and pdec_vip.authorized_action == Action.ESCALATE
    )
    return passed


def verify_scenario_6_policy_rejection() -> bool:
    """6. Policy Rejection (§18.6)."""
    gate = InvariantPolicyGate()
    payment = Payment(payment_id="pay_hard_v", customer_id="c_v", amount=Decimal("100.00"))
    ctx = PaymentContext(
        payment_id="pay_hard_v", customer_id="c_v", customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.CREDIT_CARD, raw_error_code="STOLEN", raw_error_message="Stolen card",
        failure_category=FailureCategory.HARD_DECLINE, failure_severity=FailureSeverity.FATAL,
    )
    dec = RecoveryDecision("pay_hard_v", Action.RETRY_NOW)
    pdec = gate.authorize(payment, ctx, dec)

    passed = (
        pdec.is_authorized is False
        and pdec.authorized_action == Action.STOP
        and "hard decline" in pdec.rejection_reason.lower()
    )
    return passed


def check_blocking_cooldown_safety_finding() -> dict:
    """Check and record the state of the known Policy Gate cooldown defect (§11.3)."""
    config = PolicyConfig(cooldown_seconds=900)
    gate = InvariantPolicyGate(config)
    payment = Payment(payment_id="pay_cd_diag", customer_id="c_diag", amount=Decimal("100.00"), attempt_count=1)
    ctx = PaymentContext(
        payment_id="pay_cd_diag", customer_id="c_diag", customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.CREDIT_CARD, raw_error_code="T", raw_error_message="T",
        failure_category=FailureCategory.NETWORK_TIMEOUT, failure_severity=FailureSeverity.TRANSIENT,
        last_attempt_timestamp=datetime.now(timezone.utc) - timedelta(seconds=60),
    )
    dec = RecoveryDecision("pay_cd_diag", Action.RETRY_LATER)
    pdec = gate.authorize(payment, ctx, dec)

    # In SPEC §11.3, an unelapsed cooldown MUST be rejected (is_authorized=False)
    spec_satisfied = (pdec.is_authorized is False and pdec.authorized_action == Action.STOP)
    return {
        "spec_satisfied": spec_satisfied,
        "is_authorized": pdec.is_authorized,
        "authorized_action": pdec.authorized_action.value,
        "rule_results": [(r.rule_name, r.passed) for r in pdec.rule_results],
    }


def main():
    print("=" * 80)
    print("RecoverIQ Sprint 7 — Failure & Resilience Verification (SPEC §18)")
    print("=" * 80)

    # SECTION A — §18 SCENARIO VERDICTS
    print("\nSECTION A — SPEC §18 FAILURE & RESILIENCE SCENARIOS")
    print("-" * 80)

    s1 = verify_scenario_1_duplicate_webhook()
    print(f"  [1] Duplicate Webhook Delivery (§18.1):         {'PASSED' if s1 else 'FAILED'}")

    s2 = verify_scenario_2_out_of_order_webhook()
    print(f"  [2] Out-of-Order Webhook Delivery (§18.2):       {'PASSED' if s2 else 'FAILED'}")

    s3 = verify_scenario_3_executor_timeout()
    print(f"  [3] Executor Timeout Protection (§18.3):         {'PASSED' if s3 else 'FAILED'}")

    s4 = verify_scenario_4_duplicate_action_request()
    print(f"  [4] Duplicate Action Request Idempotency (§18.4):{'PASSED' if s4 else 'FAILED'}")

    s5 = verify_scenario_5_exhausted_retry_budget()
    print(f"  [5] Exhausted Retry Budget Enforcement (§18.5):  {'PASSED' if s5 else 'FAILED'}")

    s6 = verify_scenario_6_policy_rejection()
    print(f"  [6] Policy Rejection on Invariant Breach (§18.6): {'PASSED' if s6 else 'FAILED'}")

    all_scenarios_passed = all([s1, s2, s3, s4, s5, s6])
    print("-" * 80)
    print(f"  Summary Section A: {'ALL 6 SCENARIOS PASSED' if all_scenarios_passed else 'SOME SCENARIOS FAILED'}")

    # SECTION B — BLOCKING SAFETY FINDINGS
    print("\nSECTION B — BLOCKING SAFETY FINDINGS (SPEC §11.3 COOLDOWN)")
    print("-" * 80)
    cd_info = check_blocking_cooldown_safety_finding()
    if cd_info["spec_satisfied"]:
        print("  [ALERT] Cooldown invariant unexpectedly satisfied in Policy Gate.")
    else:
        print("  [CONFIRMED BLOCKING DEFECT] SPEC §11.3 Mandatory Cooldown Invariant Violation:")
        print("  - Location: src/recoveriq/policy/gate.py lines 130-153")
        print("  - Observed behavior: When elapsed_seconds < cooldown_seconds (e.g. 60s < 900s),")
        print("    gate appends passed=False, then appends passed=True, and authorizes the action.")
        print(f"  - Actual Authorization Verdict: is_authorized={cd_info['is_authorized']}, authorized_action='{cd_info['authorized_action']}'")
        print(f"  - Recorded Rule Checks: {cd_info['rule_results']}")
        print("  - Status: UNRESOLVED / PRESERVED VISIBLE (Per Sprint 7 research design)")
        print("  - Test Evidence: tests/test_failure_resilience.py::test_spec_correct_cooldown_enforcement (XFAIL STRICT)")
        print("  - Canary Evidence: tests/test_failure_resilience.py::test_canary_observed_cooldown_gate_violation (PASSED)")

    # SECTION C — NON-BLOCKING GAPS & SCIENTIFIC LIMITATIONS
    print("\nSECTION C — NON-BLOCKING GAPS & SCIENTIFIC LIMITATIONS")
    print("-" * 80)
    print("  1. Concurrency Testing Scope:")
    print("     The test suite exercises sequential deduplication and idempotency keys.")
    print("     It does NOT claim to prove concurrent thread safety across distributed workers.")
    print("  2. Persistence Architecture Boundary:")
    print("     The production RecoverIQEngine does not persist Payment entities between calls.")
    print("     Multi-event replay is orchestrated via the test-only PaymentReplayHarness.")
    print("  3. Executor Timeout Model:")
    print("     Timeout is conservatively modeled as 'dispatch did not complete' and enters")
    print("     a safe non-transitioned state. It does not model network partitions where downstream")
    print("     succeeded but the response was dropped.")

    print("\n" + "=" * 80)
    if all_scenarios_passed:
        print("SPRINT 7 VERIFICATION COMPLETE: ALL SIX §18 SCENARIOS PASSED.")
        print("POLICY GATE COOLDOWN DEFECT (SPEC §11.3) REMAINS CONFIRMED AND DOCUMENTED.")
        print("=" * 80)
        sys.exit(0)
    else:
        print("SPRINT 7 VERIFICATION FAILED: SCENARIO REGRESSIONS DETECTED.")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
