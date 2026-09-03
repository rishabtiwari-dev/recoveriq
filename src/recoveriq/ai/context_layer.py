"""AI Context Layer interfaces and contracts.

CRITICAL ARCHITECTURE INVARIANT:
The AI / LLM Context Layer is strictly an interpretive module.
It extracts structured context from unstructured logs and produces explanations.
It has ZERO execution authority and NO capability to execute payment actions.
"""

from typing import Protocol, runtime_checkable

from recoveriq.domain.events import PaymentFailedEvent
from recoveriq.domain.models import PaymentContext


@runtime_checkable
class AIContextLayer(Protocol):
    """Protocol for AI-driven contextual interpretation of failure payloads."""

    def interpret_failure(self, event: PaymentFailedEvent) -> PaymentContext:
        """Parse raw decline strings and metadata into structured PaymentContext.

        NOTE: This method returns context only. It cannot authorize or dispatch actions.
        """
        ...


class StubAIContextLayer:
    """Stub AI context layer for Sprint 1 foundation contracts.

    Sprint 3 will implement LLM structured prompt extraction with strict schema validation.
    """

    def __init__(self, fallback_extractor=None):
        from recoveriq.context.extractor import RuleBasedContextExtractor

        self._fallback = fallback_extractor or RuleBasedContextExtractor()

    def interpret_failure(self, event: PaymentFailedEvent) -> PaymentContext:
        """Interpret failure context with simulated diagnostic metadata."""
        base_context = self._fallback.extract_context(event)

        diagnostic = (
            f"[AI Interpreted] Category: {base_context.failure_category.value}, "
            f"Severity: {base_context.failure_severity.value}. "
            f"Observed error: '{event.raw_error_message or event.raw_error_code}'."
        )

        return PaymentContext(
            payment_id=base_context.payment_id,
            customer_id=base_context.customer_id,
            customer_tier=base_context.customer_tier,
            payment_method=base_context.payment_method,
            raw_error_code=base_context.raw_error_code,
            raw_error_message=base_context.raw_error_message,
            failure_category=base_context.failure_category,
            failure_severity=base_context.failure_severity,
            attempt_count=base_context.attempt_count,
            last_attempt_timestamp=base_context.last_attempt_timestamp,
            diagnostic_explanation=diagnostic,
            extra_metadata=dict(base_context.extra_metadata),
        )
