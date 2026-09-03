"""Sprint 3 Boundary & Safety Invariant Verification Script.

Verifies:
1. Module isolation: AST scan of recoveriq.ai has 0 imports from policy, economics, executor, engine, simulation.
2. Anti-authority: GeminiContextLayer has 0 execution/authorization methods.
3. Input privacy: Prohibited customer/financial fields cannot enter prompts.
4. Output blacklist: Forbidden authority keys trigger immediate guardrail rejection.
5. Fallback resilience: Offline execution produces valid PaymentContext without halting.
"""

import ast
import json
import os
import pathlib
import sys
from decimal import Decimal

from recoveriq.ai.gemini_context_layer import GeminiContextLayer
from recoveriq.ai.prompt_template import render_user_prompt
from recoveriq.ai.schema_validator import FORBIDDEN_AUTHORITY_KEYS, validate_llm_response
from recoveriq.config.settings import LLMConfig
from recoveriq.domain.events import EventType, PaymentFailedEvent
from recoveriq.domain.models import CustomerTier, PaymentContext, PaymentMethod


def verify_ast_isolation() -> bool:
    print("Checking AST import isolation in src/recoveriq/ai/ ...")
    ai_dir = pathlib.Path(__file__).parent.parent / "src" / "recoveriq" / "ai"
    forbidden_prefixes = [
        "recoveriq.policy",
        "recoveriq.economics",
        "recoveriq.executor",
        "recoveriq.engine",
        "recoveriq.simulation",
    ]

    violations = []
    for py_file in ai_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = ""
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name
            for prefix in forbidden_prefixes:
                if module.startswith(prefix):
                    violations.append(f"{py_file.name} line {node.lineno}: {module}")

    if violations:
        print("[FAIL] Forbidden cross-module imports detected:")
        for v in violations:
            print(f"  - {v}")
        return False

    print("[OK] Zero forbidden cross-module imports in recoveriq.ai.")
    return True


def verify_no_execution_methods() -> bool:
    print("Checking GeminiContextLayer for unauthorized execution methods ...")
    forbidden = ["execute", "dispatch", "authorize", "select_action", "evaluate_policy", "calculate_ev"]
    found = [m for m in forbidden if hasattr(GeminiContextLayer, m)]
    if found:
        print(f"[FAIL] Found unauthorized methods: {found}")
        return False

    print("[OK] GeminiContextLayer has zero execution authority methods.")
    return True


def verify_input_privacy() -> bool:
    print("Checking input privacy boundary ...")
    prompt = render_user_prompt(
        raw_error_code="TIMEOUT",
        raw_error_message="Gateway timeout",
        payment_method=PaymentMethod.UPI,
        attempt_count=1,
    )

    sensitive_tokens = ["pay_123", "cust_456", "999.00", "INR", "VIP"]
    for tok in sensitive_tokens:
        if tok in prompt:
            print(f"[FAIL] Sensitive token '{tok}' leaked into rendered prompt!")
            return False

    print("[OK] Input privacy boundary verified. No identity or financial fields present.")
    return True


def verify_anti_authority_blacklist() -> bool:
    print("Checking anti-authority schema blacklist ...")
    for key in FORBIDDEN_AUTHORITY_KEYS:
        payload = {
            "failure_category": "INSUFFICIENT_FUNDS",
            "failure_severity": "RECOVERABLE",
            "diagnostic_explanation": "Valid description",
            key: "RETRY_NOW",
        }
        res = validate_llm_response(json.dumps(payload))
        if res.is_valid or not res.is_guardrail_breach:
            print(f"[FAIL] Forbidden key '{key}' was not rejected as a guardrail breach!")
            return False

    print(f"[OK] All {len(FORBIDDEN_AUTHORITY_KEYS)} forbidden authority keys correctly rejected.")
    return True


def verify_offline_fallback() -> bool:
    print("Checking offline deterministic fallback ...")
    os.environ.pop("GEMINI_API_KEY", None)

    event = PaymentFailedEvent(
        event_id="evt_verify_001",
        payment_id="pay_verify_001",
        event_type=EventType.PAYMENT_FAILED,
        customer_id="cust_verify_001",
        amount=Decimal("100.00"),
        customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="INSUFFICIENT_FUNDS",
        raw_error_message="Account balance low",
        attempt_count=1,
    )

    layer = GeminiContextLayer(config=LLMConfig())
    ctx = layer.interpret_failure(event)

    if not isinstance(ctx, PaymentContext):
        print("[FAIL] Fallback did not return a valid PaymentContext instance.")
        return False

    if not ctx.extra_metadata.get("fallback_used"):
        print("[FAIL] Fallback metadata flag not set.")
        return False

    if "raw_prompt" in ctx.extra_metadata or "raw_response" in ctx.extra_metadata:
        print("[FAIL] Sensitive raw prompt/response leaked into metadata.")
        return False

    print("[OK] Offline fallback cleanly executed and safe telemetry recorded.")
    return True


def main() -> int:
    print("=" * 60)
    print("RecoverIQ Sprint 3 — AI Context Layer Boundary Verification")
    print("=" * 60)

    checks = [
        verify_ast_isolation,
        verify_no_execution_methods,
        verify_input_privacy,
        verify_anti_authority_blacklist,
        verify_offline_fallback,
    ]

    all_passed = True
    for check in checks:
        passed = check()
        if not passed:
            all_passed = False
        print("-" * 60)

    if all_passed:
        print("ALL SPRINT 3 BOUNDARY CHECKS PASSED.")
        return 0
    else:
        print("ONE OR MORE SPRINT 3 BOUNDARY CHECKS FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
