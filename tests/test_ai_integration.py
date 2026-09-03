"""Sprint 3 Tests — Live Gemini 3.8 Flash integration test (optional, skippable).

NOTE: This test suite is skipped by default unless GEMINI_API_KEY is exported.
It is NOT part of the mandatory offline unit test suite.
"""

import os
from decimal import Decimal
import pytest

from recoveriq.ai.gemini_context_layer import GeminiContextLayer
from recoveriq.config.settings import LLMConfig
from recoveriq.domain.events import EventType, PaymentFailedEvent
from recoveriq.domain.models import CustomerTier, FailureCategory, PaymentMethod


@pytest.mark.llm_integration
@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="Live GEMINI_API_KEY required for integration test",
)
def test_live_gemini_classifies_timeout():
    """Live call to Gemini 3.8 Flash verifying successful classification and schema conformance."""
    event = PaymentFailedEvent(
        event_id="evt_live_001",
        payment_id="pay_live_001",
        event_type=EventType.PAYMENT_FAILED,
        customer_id="cust_live_001",
        amount=Decimal("500.00"),
        currency="INR",
        customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.UPI,
        raw_error_code="504_GATEWAY_TIMEOUT",
        raw_error_message="Gateway timeout contacting upstream switch after 30s",
        attempt_count=1,
    )

    layer = GeminiContextLayer(config=LLMConfig(model_name="gemini-3.8-flash"))
    ctx = layer.interpret_failure(event)

    assert ctx.failure_category == FailureCategory.NETWORK_TIMEOUT
    assert ctx.extra_metadata["fallback_used"] is False
    assert ctx.extra_metadata["llm_model"] == "gemini-3.8-flash"
    assert "raw_prompt" not in ctx.extra_metadata
    assert "raw_response" not in ctx.extra_metadata
