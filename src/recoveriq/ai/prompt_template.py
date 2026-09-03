"""Prompt construction and templating for LLM payment failure interpretation.

CRITICAL PRIVACY BOUNDARY:
Only the following four technical fields are permitted into prompt rendering:
1. raw_error_code
2. raw_error_message
3. payment_method
4. attempt_count

The following fields must NEVER be passed or interpolated:
- payment_id
- customer_id
- amount
- currency
- customer_tier
- event_id
- timestamp
- latent_recoverability_profile
- action_base_probabilities

ANTI-AUTHORITY BOUNDARY:
- No recovery action options are presented as choices to the model.
- The model is instructed strictly to classify the failure taxonomy and provide
  a factual explanation.
- The model is explicitly barred from recommending, authorizing, or evaluating actions.
"""

from typing import Any, Dict

from recoveriq.domain.models import FailureCategory, FailureSeverity, PaymentMethod

SYSTEM_INSTRUCTION = (
    "You are a technical payment failure diagnostics interpreter. "
    "Your sole task is to analyze raw gateway error signals and classify them into a standardized taxonomy.\n\n"
    "STRICT SAFETY & ROLE CONSTRAINTS:\n"
    "- You are strictly an interpreter. You have zero operational or execution authority.\n"
    "- You MUST NOT recommend, select, or suggest any recovery action.\n"
    "- You MUST NOT compute, estimate, or mention recovery probabilities or expected value.\n"
    "- You MUST NOT evaluate business policies, customer status, or financial priority.\n"
    "- Output pure JSON matching the requested schema. Do not include markdown code blocks or additional text."
)

_VALID_CATEGORIES = ", ".join(c.value for c in FailureCategory)
_VALID_SEVERITIES = ", ".join(s.value for s in FailureSeverity)

USER_PROMPT_TEMPLATE = """Analyze the following technical payment failure signals:

Technical Error Code: {raw_error_code}
Technical Error Message: {raw_error_message}
Payment Instrument Rail: {payment_method}
Failure Attempt Count: {attempt_count}

Output a single JSON object strictly matching this schema:
{{
  "failure_category": "<must be exactly one of: {valid_categories}>",
  "failure_severity": "<must be exactly one of: {valid_severities}>",
  "diagnostic_explanation": "<factual, concise diagnostic summary of the error, max 300 chars>"
}}
"""


def render_user_prompt(
    raw_error_code: str,
    raw_error_message: str,
    payment_method: PaymentMethod | str,
    attempt_count: int,
) -> str:
    """Render the user prompt using strictly the four permitted technical fields."""
    method_str = payment_method.value if isinstance(payment_method, PaymentMethod) else str(payment_method)
    return USER_PROMPT_TEMPLATE.format(
        raw_error_code=raw_error_code or "N/A",
        raw_error_message=raw_error_message or "N/A",
        payment_method=method_str or "UNKNOWN",
        attempt_count=attempt_count,
        valid_categories=_VALID_CATEGORIES,
        valid_severities=_VALID_SEVERITIES,
    )


def extract_permitted_prompt_payload(
    raw_error_code: str,
    raw_error_message: str,
    payment_method: PaymentMethod | str,
    attempt_count: int,
) -> Dict[str, Any]:
    """Package strictly the 4 permitted fields for external API transmission."""
    method_str = payment_method.value if isinstance(payment_method, PaymentMethod) else str(payment_method)
    return {
        "raw_error_code": raw_error_code or "",
        "raw_error_message": raw_error_message or "",
        "payment_method": method_str or "",
        "attempt_count": int(attempt_count),
    }
