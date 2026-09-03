"""Strict schema validation and anti-authority enforcement for LLM responses.

ANTI-AUTHORITY BLACKLIST:
Any occurrence of decision-making or authority-bearing keys in the response
triggers an immediate guardrail safety rejection and forces deterministic fallback.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional

from recoveriq.domain.models import FailureCategory, FailureSeverity

# Prohibited keys that indicate decision-making, policy gating, or execution authority
FORBIDDEN_AUTHORITY_KEYS: FrozenSet[str] = frozenset(
    {
        "action",
        "authorized_action",
        "policy_decision",
        "dispatch",
        "ev",
        "expected_value",
        "probability",
        "retry",
        "escalate",
        "stop",
    }
)

MAX_DIAGNOSTIC_LENGTH = 300


@dataclass(frozen=True)
class LLMValidationResult:
    """Result of validating an LLM output against RecoverIQ schema and safety invariants."""

    is_valid: bool
    failure_category: Optional[FailureCategory] = None
    failure_severity: Optional[FailureSeverity] = None
    diagnostic_explanation: Optional[str] = None
    error_category: Optional[str] = None
    error_message: Optional[str] = None
    is_guardrail_breach: bool = False

    @classmethod
    def success(
        cls,
        category: FailureCategory,
        severity: FailureSeverity,
        diagnostic: str,
    ) -> "LLMValidationResult":
        return cls(
            is_valid=True,
            failure_category=category,
            failure_severity=severity,
            diagnostic_explanation=diagnostic,
        )

    @classmethod
    def failure(
        cls,
        error_category: str,
        error_message: str,
        is_guardrail_breach: bool = False,
    ) -> "LLMValidationResult":
        return cls(
            is_valid=False,
            error_category=error_category,
            error_message=error_message,
            is_guardrail_breach=is_guardrail_breach,
        )


def _check_forbidden_keys(data: Any) -> Optional[str]:
    """Recursively check for the presence of any forbidden authority-bearing keys."""
    if isinstance(data, dict):
        for key in data:
            key_lower = str(key).strip().lower()
            if key_lower in FORBIDDEN_AUTHORITY_KEYS:
                return key_lower
            res = _check_forbidden_keys(data[key])
            if res:
                return res
    elif isinstance(data, list):
        for item in data:
            res = _check_forbidden_keys(item)
            if res:
                return res
    return None


def validate_llm_response(raw_text: str) -> LLMValidationResult:
    """Parse and rigorously validate raw response text from Gemini.

    Enforces:
    1. Parseable JSON syntax.
    2. Zero presence of forbidden authority keys.
    3. Exact conformance to FailureCategory and FailureSeverity enums.
    4. Diagnostic explanation length constraint.
    """
    if not raw_text or not raw_text.strip():
        return LLMValidationResult.failure(
            error_category="EMPTY_RESPONSE",
            error_message="LLM returned empty or whitespace response.",
        )

    text = raw_text.strip()
    # Strip markdown code fencing if returned by model
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        data = json.loads(text)
    except Exception as e:
        return LLMValidationResult.failure(
            error_category="MALFORMED_JSON",
            error_message=f"JSON decoding failed: {str(e)}",
        )

    if not isinstance(data, dict):
        return LLMValidationResult.failure(
            error_category="INVALID_SCHEMA",
            error_message="Expected JSON object at top level.",
        )

    # 1. Anti-authority blacklist check
    forbidden_key = _check_forbidden_keys(data)
    if forbidden_key:
        return LLMValidationResult.failure(
            error_category="GUARDRAIL_BREACH",
            error_message=f"Forbidden decision/authority key detected: '{forbidden_key}'",
            is_guardrail_breach=True,
        )

    # 2. Required fields check
    for req in ("failure_category", "failure_severity", "diagnostic_explanation"):
        if req not in data:
            return LLMValidationResult.failure(
                error_category="MISSING_FIELD",
                error_message=f"Required field '{req}' is missing from JSON response.",
            )

    # 3. Enum conformance check
    cat_str = str(data["failure_category"]).strip().upper()
    try:
        category = FailureCategory(cat_str)
    except ValueError:
        return LLMValidationResult.failure(
            error_category="INVALID_ENUM",
            error_message=f"Value '{cat_str}' is not a valid FailureCategory.",
        )

    sev_str = str(data["failure_severity"]).strip().upper()
    try:
        severity = FailureSeverity(sev_str)
    except ValueError:
        return LLMValidationResult.failure(
            error_category="INVALID_ENUM",
            error_message=f"Value '{sev_str}' is not a valid FailureSeverity.",
        )

    # 4. Diagnostic length check
    diagnostic = str(data["diagnostic_explanation"]).strip()
    if not diagnostic:
        return LLMValidationResult.failure(
            error_category="INVALID_DIAGNOSTIC",
            error_message="Diagnostic explanation cannot be empty.",
        )
    if len(diagnostic) > MAX_DIAGNOSTIC_LENGTH:
        return LLMValidationResult.failure(
            error_category="FIELD_TOO_LONG",
            error_message=f"Diagnostic length ({len(diagnostic)}) exceeds max allowed {MAX_DIAGNOSTIC_LENGTH}.",
        )

    return LLMValidationResult.success(
        category=category,
        severity=severity,
        diagnostic=diagnostic,
    )
