"""Sprint 3 Tests — Static AST and runtime boundary contract verification."""

import ast
import json
import pathlib
from decimal import Decimal
import pytest

from recoveriq.ai.gemini_context_layer import GeminiContextLayer
from recoveriq.config.settings import LLMConfig
from recoveriq.domain.events import EventType, PaymentFailedEvent
from recoveriq.domain.models import CustomerTier, PaymentContext, PaymentMethod


def test_input_privacy_boundary_strictly_enforced():
    """Prove that forbidden customer and payment fields NEVER enter the prompt."""
    captured_prompts = []

    def mock_caller(prompt, system_instruction, timeout):
        captured_prompts.append(prompt)
        return json.dumps({
            "failure_category": "NETWORK_TIMEOUT",
            "failure_severity": "TRANSIENT",
            "diagnostic_explanation": "Gateway timeout.",
        })

    layer = GeminiContextLayer(
        config=LLMConfig(),
        api_caller=mock_caller,
    )

    event = PaymentFailedEvent(
        event_id="evt_leak_check_999",
        payment_id="pay_confidential_12345",
        event_type=EventType.PAYMENT_FAILED,
        customer_id="cust_ultra_secret_67890",
        amount=Decimal("884422.50"),
        currency="USD",
        customer_tier=CustomerTier.VIP,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="GATEWAY_504_TIMEOUT",
        raw_error_message="Upstream switch unreachable",
        attempt_count=2,
    )

    ctx = layer.interpret_failure(event)
    assert len(captured_prompts) == 1
    rendered = captured_prompts[0]

    # Prohibited fields must be strictly absent
    assert "pay_confidential_12345" not in rendered
    assert "cust_ultra_secret_67890" not in rendered
    assert "884422.50" not in rendered
    assert "USD" not in rendered
    assert "VIP" not in rendered
    assert "evt_leak_check_999" not in rendered

    # Permitted fields must be present
    assert "GATEWAY_504_TIMEOUT" in rendered
    assert "Upstream switch unreachable" in rendered
    assert "CREDIT_CARD" in rendered
    assert "2" in rendered


def test_ast_scan_no_forbidden_module_imports_in_ai_package():
    """Verify via AST that recoveriq.ai package has zero imports from policy, economics, executor, engine, or simulation."""
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
                    violations.append(f"{py_file.name} line {node.lineno}: imports {module}")

    assert not violations, f"Forbidden cross-module imports detected in AI package:\n" + "\n".join(violations)


def test_gemini_context_layer_has_zero_execution_authority_methods():
    """Verify that GeminiContextLayer possesses zero execution or authorization methods."""
    forbidden_methods = [
        "execute",
        "dispatch",
        "authorize",
        "select_action",
        "evaluate_policy",
        "calculate_ev",
    ]

    for m in forbidden_methods:
        assert not hasattr(GeminiContextLayer, m), f"GeminiContextLayer must not have method '{m}'"


def test_payment_context_cannot_carry_an_action():
    """Verify that PaymentContext domain model has no action attribute."""
    event = PaymentFailedEvent(
        event_id="evt_0",
        payment_id="pay_0",
        event_type=EventType.PAYMENT_FAILED,
        raw_error_code="TIMEOUT",
        raw_error_message="timeout",
    )
    layer = GeminiContextLayer(
        config=LLMConfig(),
        api_caller=lambda p, s, t: json.dumps({
            "failure_category": "NETWORK_TIMEOUT",
            "failure_severity": "TRANSIENT",
            "diagnostic_explanation": "timeout",
        }),
    )
    ctx = layer.interpret_failure(event)

    assert not hasattr(ctx, "action")
    assert not hasattr(ctx, "authorized_action")
    assert not hasattr(ctx, "proposed_action")
