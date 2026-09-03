"""Verification script for Sprint 1 foundation contracts."""

import sys
from decimal import Decimal

def main():
    print("=" * 60)
    print("RecoverIQ - Sprint 1 Foundation Verification")
    print("=" * 60)

    try:
        from recoveriq.domain.actions import Action
        from recoveriq.domain.state import PaymentState, is_valid_transition
        from recoveriq.domain.models import Payment, PaymentContext, CustomerTier, PaymentMethod, FailureCategory, FailureSeverity
        from recoveriq.domain.decisions import CandidateActionEV, RecoveryDecision, PolicyDecision
        from recoveriq.domain.events import PaymentFailedEvent, EventType
        from recoveriq.domain.idempotency import generate_idempotency_key, IdempotencyRecord
        from recoveriq.config.settings import RecoverIQConfig
        from recoveriq.context.extractor import RuleBasedContextExtractor
        from recoveriq.ai.context_layer import StubAIContextLayer
        from recoveriq.model.probability import StubProbabilityModel
        from recoveriq.economics.engine import DefaultEconomicEngine
        from recoveriq.policy.gate import InvariantPolicyGate
        from recoveriq.executor.executor import InMemoryActionExecutor
        from recoveriq.audit.logger import InMemoryAuditLogger
        from recoveriq.engine import RecoverIQEngine

        print("[OK] All modules imported successfully without circular dependencies.")

        # Verify Action Space
        assert len(Action) == 6
        print(f"[OK] Action space verified: {[a.value for a in Action]}")

        # Verify State Machine
        assert is_valid_transition(PaymentState.FAILED_INITIAL, PaymentState.RECOVERING)
        assert not is_valid_transition(PaymentState.RECOVERED, PaymentState.RECOVERING)
        print("[OK] State machine transitions verified.")

        # Verify Idempotency key
        key = generate_idempotency_key("pay_1", Action.RETRY_NOW, 1, "evt_1")
        assert len(key) == 64
        print(f"[OK] Idempotency key generation verified: {key[:12]}...")

        # Verify End-to-end pipeline run
        engine = RecoverIQEngine()
        event = PaymentFailedEvent(
            event_id="evt_verify_1",
            payment_id="pay_verify_1",
            event_type=EventType.PAYMENT_FAILED,
            customer_id="cust_1",
            amount=Decimal("100.00"),
            customer_tier=CustomerTier.STANDARD,
            payment_method=PaymentMethod.CREDIT_CARD,
            raw_error_code="TIMEOUT",
            raw_error_message="Gateway timeout",
        )
        res = engine.process_failure_event(event)
        assert res.payment_id == "pay_verify_1"
        print(f"[OK] Pipeline execution verified: Action={res.action.value}, Status={res.status.value}")

        print("=" * 60)
        print("ALL SPRINT 1 FOUNDATION CHECKS PASSED.")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"[FAIL] Verification failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
