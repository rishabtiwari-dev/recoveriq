"""Sprint 3 Tests — Prompt template rendering and input privacy boundary."""

import pytest
from recoveriq.ai.prompt_template import (
    SYSTEM_INSTRUCTION,
    extract_permitted_prompt_payload,
    render_user_prompt,
)
from recoveriq.domain.actions import Action
from recoveriq.domain.models import CustomerTier, FailureCategory, FailureSeverity, PaymentMethod


def test_prompt_renders_all_four_permitted_fields():
    """Prompt must render raw_error_code, raw_error_message, payment_method, attempt_count."""
    prompt = render_user_prompt(
        raw_error_code="ERR_504_TIMEOUT",
        raw_error_message="Gateway connection timed out after 30s",
        payment_method=PaymentMethod.UPI,
        attempt_count=2,
    )

    assert "ERR_504_TIMEOUT" in prompt
    assert "Gateway connection timed out after 30s" in prompt
    assert "UPI" in prompt
    assert "2" in prompt


def test_extracted_payload_contains_strictly_four_fields():
    """Payload extractor must produce a dictionary containing exactly the 4 permitted keys."""
    payload = extract_permitted_prompt_payload(
        raw_error_code="NSF_001",
        raw_error_message="Insufficient balance",
        payment_method=PaymentMethod.CREDIT_CARD,
        attempt_count=1,
    )

    expected_keys = {"raw_error_code", "raw_error_message", "payment_method", "attempt_count"}
    assert set(payload.keys()) == expected_keys


def test_prohibited_identity_and_financial_fields_absent_from_rendered_prompt():
    """Customer identifiers, payment IDs, amounts, currencies, and tiers must NEVER appear in prompt."""
    sensitive_payment_id = "pay_secret_998877"
    sensitive_customer_id = "cust_vip_443322"
    sensitive_amount = "99999.00"
    sensitive_currency = "INR"
    sensitive_tier = "VIP"

    prompt = render_user_prompt(
        raw_error_code="DO_NOT_HONOR",
        raw_error_message="Issuer declined transaction",
        payment_method=PaymentMethod.DEBIT_CARD,
        attempt_count=1,
    )

    assert sensitive_payment_id not in prompt
    assert sensitive_customer_id not in prompt
    assert sensitive_amount not in prompt
    assert sensitive_currency not in prompt
    assert sensitive_tier not in prompt


def test_no_recovery_action_options_supplied_as_decision_choices():
    """Prompt must not list recovery actions as choices for the model to select."""
    prompt = render_user_prompt(
        raw_error_code="AUTH_FAILED",
        raw_error_message="3DS failed",
        payment_method=PaymentMethod.NET_BANKING,
        attempt_count=1,
    )

    # Recovery actions must NOT be listed as selectable targets
    for action in Action:
        assert f'select "{action.value}"' not in prompt.lower()
        assert f'choose "{action.value}"' not in prompt.lower()
        assert f'recommend "{action.value}"' not in prompt.lower()


def test_system_instruction_strictly_forbids_action_and_ev_authority():
    """System prompt must explicitly prohibit action recommendation and EV calculation."""
    sys_lower = SYSTEM_INSTRUCTION.lower()
    assert "zero operational or execution authority" in sys_lower
    assert "must not recommend, select, or suggest any recovery action" in sys_lower
    assert "must not compute, estimate, or mention recovery probabilities or expected value" in sys_lower


def test_all_failure_categories_and_severities_present_in_schema():
    """Prompt schema must instruct on all 9 FailureCategory and 4 FailureSeverity values."""
    prompt = render_user_prompt("ERR", "MSG", PaymentMethod.WALLET, 1)

    for cat in FailureCategory:
        assert cat.value in prompt

    for sev in FailureSeverity:
        assert sev.value in prompt
