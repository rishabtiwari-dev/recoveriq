"""Context extraction interfaces and rule-based fallback implementation."""

from typing import Protocol, runtime_checkable

from recoveriq.domain.events import PaymentFailedEvent
from recoveriq.domain.models import (
    FailureCategory,
    FailureSeverity,
    PaymentContext,
)


@runtime_checkable
class ContextExtractor(Protocol):
    """Protocol for extracting structured context from raw payment failure events."""

    def extract_context(self, event: PaymentFailedEvent) -> PaymentContext:
        """Parse raw event payload into a validated PaymentContext domain object."""
        ...


class RuleBasedContextExtractor:
    """Deterministic, rule-based baseline context extractor.

    Acts as the standard rule baseline and the zero-downtime fallback if AI extraction fails.
    """

    def extract_context(self, event: PaymentFailedEvent) -> PaymentContext:
        code = (event.raw_error_code or "").upper().strip()
        msg = (event.raw_error_message or "").upper().strip()

        # Categorization heuristics
        if any(term in code or term in msg for term in ("INSUFFICIENT_FUNDS", "LOW_BALANCE", "NSF")):
            category = FailureCategory.INSUFFICIENT_FUNDS
            severity = FailureSeverity.RECOVERABLE
        elif any(term in code or term in msg for term in ("TIMEOUT", "GATEWAY_ERROR", "504", "502", "NETWORK")):
            category = FailureCategory.NETWORK_TIMEOUT
            severity = FailureSeverity.TRANSIENT
        elif any(term in code or term in msg for term in ("EXPIRED", "CARD_EXPIRED")):
            category = FailureCategory.CARD_EXPIRED
            severity = FailureSeverity.STRUCTURAL
        elif any(term in code or term in msg for term in ("OTP", "3DS_FAILED", "AUTH_FAILED")):
            category = FailureCategory.AUTHENTICATION_FAILED
            severity = FailureSeverity.RECOVERABLE
        elif any(term in code or term in msg for term in ("STOLEN", "FRAUD", "RESTRICTED", "DO_NOT_HONOR")):
            category = FailureCategory.HARD_DECLINE
            severity = FailureSeverity.FATAL
        elif any(term in code or term in msg for term in ("INVALID_CVV", "INVALID_CARD_NUMBER")):
            category = FailureCategory.INVALID_DETAILS
            severity = FailureSeverity.FATAL
        elif any(term in code or term in msg for term in ("LIMIT_EXCEEDED", "VELOCITY")):
            category = FailureCategory.VELOCITY_EXCEEDED
            severity = FailureSeverity.STRUCTURAL
        else:
            category = FailureCategory.UNKNOWN
            severity = FailureSeverity.RECOVERABLE

        return PaymentContext(
            payment_id=event.payment_id,
            customer_id=event.customer_id,
            customer_tier=event.customer_tier,
            payment_method=event.payment_method,
            raw_error_code=event.raw_error_code,
            raw_error_message=event.raw_error_message,
            failure_category=category,
            failure_severity=severity,
            attempt_count=event.attempt_count,
            last_attempt_timestamp=event.timestamp,
            diagnostic_explanation=f"Rule-parsed category: {category.value}",
        )
