"""Sprint 3 Tests — Schema validation and anti-authority blacklist enforcement."""

import json
import pytest
from recoveriq.ai.schema_validator import (
    FORBIDDEN_AUTHORITY_KEYS,
    MAX_DIAGNOSTIC_LENGTH,
    validate_llm_response,
)
from recoveriq.domain.models import FailureCategory, FailureSeverity


def test_valid_json_response_parses_cleanly():
    """Valid JSON containing permitted fields parses successfully into domain enums."""
    payload = {
        "failure_category": "INSUFFICIENT_FUNDS",
        "failure_severity": "RECOVERABLE",
        "diagnostic_explanation": "Customer account balance too low to authorize transaction.",
    }
    result = validate_llm_response(json.dumps(payload))

    assert result.is_valid is True
    assert result.failure_category == FailureCategory.INSUFFICIENT_FUNDS
    assert result.failure_severity == FailureSeverity.RECOVERABLE
    assert result.diagnostic_explanation == "Customer account balance too low to authorize transaction."
    assert result.is_guardrail_breach is False


def test_markdown_wrapped_json_parsed_successfully():
    """Responses wrapped in markdown code blocks must be cleanly stripped and parsed."""
    raw = """```json
    {
      "failure_category": "NETWORK_TIMEOUT",
      "failure_severity": "TRANSIENT",
      "diagnostic_explanation": "Issuer connection timeout after 30s."
    }
    ```"""
    result = validate_llm_response(raw)

    assert result.is_valid is True
    assert result.failure_category == FailureCategory.NETWORK_TIMEOUT
    assert result.failure_severity == FailureSeverity.TRANSIENT


def test_malformed_json_triggers_failure():
    """Malformed non-JSON text must fail with MALFORMED_JSON error category."""
    raw = "{failure_category: 'INSUFFICIENT_FUNDS', severity: bad json"
    result = validate_llm_response(raw)

    assert result.is_valid is False
    assert result.error_category == "MALFORMED_JSON"


def test_empty_or_whitespace_response_triggers_failure():
    """Empty or whitespace response must fail cleanly."""
    for empty in ("", "   ", "\n\t"):
        result = validate_llm_response(empty)
        assert result.is_valid is False
        assert result.error_category == "EMPTY_RESPONSE"


def test_missing_required_fields_rejected():
    """Omitting any of the required schema fields must result in validation failure."""
    complete = {
        "failure_category": "CARD_EXPIRED",
        "failure_severity": "STRUCTURAL",
        "diagnostic_explanation": "Card expiration date is in the past.",
    }

    for req in ("failure_category", "failure_severity", "diagnostic_explanation"):
        truncated = dict(complete)
        del truncated[req]
        result = validate_llm_response(json.dumps(truncated))
        assert result.is_valid is False
        assert result.error_category == "MISSING_FIELD"
        assert req in result.error_message


def test_invalid_enum_values_rejected():
    """Unknown category or severity values must fail with INVALID_ENUM."""
    payload_bad_cat = {
        "failure_category": "COSMIC_RAY_INTERFERENCE",
        "failure_severity": "RECOVERABLE",
        "diagnostic_explanation": "Unknown cosmic error.",
    }
    res_cat = validate_llm_response(json.dumps(payload_bad_cat))
    assert res_cat.is_valid is False
    assert res_cat.error_category == "INVALID_ENUM"

    payload_bad_sev = {
        "failure_category": "HARD_DECLINE",
        "failure_severity": "APOCALYPTIC",
        "diagnostic_explanation": "Card stolen.",
    }
    res_sev = validate_llm_response(json.dumps(payload_bad_sev))
    assert res_sev.is_valid is False
    assert res_sev.error_category == "INVALID_ENUM"


@pytest.mark.parametrize("forbidden_key", list(FORBIDDEN_AUTHORITY_KEYS))
def test_anti_authority_blacklist_rejects_forbidden_keys(forbidden_key):
    """Presence of ANY forbidden decision-making key must trigger GUARDRAIL_BREACH."""
    payload = {
        "failure_category": "INSUFFICIENT_FUNDS",
        "failure_severity": "RECOVERABLE",
        "diagnostic_explanation": "Card balance insufficient.",
        forbidden_key: "RETRY_LATER",
    }
    result = validate_llm_response(json.dumps(payload))

    assert result.is_valid is False
    assert result.error_category == "GUARDRAIL_BREACH"
    assert result.is_guardrail_breach is True
    assert forbidden_key in result.error_message


def test_nested_anti_authority_key_triggers_rejection():
    """Forbidden keys nested inside sub-dictionaries or lists must also be caught."""
    nested_payload = {
        "failure_category": "AUTHENTICATION_FAILED",
        "failure_severity": "RECOVERABLE",
        "diagnostic_explanation": "3DS authentication dropped.",
        "metadata": {
            "suggested": {
                "action": "SEND_LINK"
            }
        }
    }
    result = validate_llm_response(json.dumps(nested_payload))

    assert result.is_valid is False
    assert result.error_category == "GUARDRAIL_BREACH"
    assert result.is_guardrail_breach is True


def test_oversized_diagnostic_explanation_rejected():
    """Diagnostic explanation exceeding MAX_DIAGNOSTIC_LENGTH must fail validation."""
    oversized = "A" * (MAX_DIAGNOSTIC_LENGTH + 1)
    payload = {
        "failure_category": "UNKNOWN",
        "failure_severity": "RECOVERABLE",
        "diagnostic_explanation": oversized,
    }
    result = validate_llm_response(json.dumps(payload))

    assert result.is_valid is False
    assert result.error_category == "FIELD_TOO_LONG"
