"""Sprint 3 Tests — Gemini Context Layer offline unit tests with mocked API."""

import json
import os
import time
from decimal import Decimal
import pytest

from recoveriq.ai.gemini_context_layer import GeminiContextLayer
from recoveriq.config.settings import LLMConfig
from recoveriq.domain.events import EventType, PaymentFailedEvent
from recoveriq.domain.models import (
    CustomerTier,
    FailureCategory,
    FailureSeverity,
    PaymentContext,
    PaymentMethod,
)


@pytest.fixture
def sample_event():
    return PaymentFailedEvent(
        event_id="evt_test_001",
        payment_id="pay_test_001",
        event_type=EventType.PAYMENT_FAILED,
        customer_id="cust_test_001",
        amount=Decimal("1500.00"),
        currency="INR",
        customer_tier=CustomerTier.PREMIUM,
        payment_method=PaymentMethod.UPI,
        raw_error_code="INSUFFICIENT_FUNDS",
        raw_error_message="Account balance is too low",
        attempt_count=1,
    )


def test_successful_gemini_extraction(sample_event):
    """Successful mocked response parses into PaymentContext with LLM fields and safe telemetry."""
    mock_response = json.dumps({
        "failure_category": "INSUFFICIENT_FUNDS",
        "failure_severity": "RECOVERABLE",
        "diagnostic_explanation": "Customer account balance insufficient for UPI collection.",
    })

    layer = GeminiContextLayer(
        config=LLMConfig(model_name="gemini-3.8-flash"),
        api_caller=lambda p, s, t: mock_response,
    )

    ctx = layer.interpret_failure(sample_event)

    assert isinstance(ctx, PaymentContext)
    assert ctx.payment_id == sample_event.payment_id
    assert ctx.customer_id == sample_event.customer_id
    assert ctx.customer_tier == sample_event.customer_tier
    assert ctx.failure_category == FailureCategory.INSUFFICIENT_FUNDS
    assert ctx.failure_severity == FailureSeverity.RECOVERABLE
    assert ctx.diagnostic_explanation == "Customer account balance insufficient for UPI collection."

    # Verify safe operational telemetry
    meta = ctx.extra_metadata
    assert meta["llm_provider"] == "gemini"
    assert meta["llm_model"] == "gemini-3.8-flash"
    assert meta["fallback_used"] is False
    assert "latency_ms" in meta
    # Privacy verification: raw prompt and raw completion must NOT be in metadata
    assert "raw_prompt" not in meta
    assert "raw_response" not in meta


def test_fallback_on_malformed_json(sample_event):
    """Malformed response triggers deterministic rule-based fallback."""
    layer = GeminiContextLayer(
        config=LLMConfig(),
        api_caller=lambda p, s, t: "INVALID NON-JSON RESPONSE",
    )

    ctx = layer.interpret_failure(sample_event)

    assert isinstance(ctx, PaymentContext)
    assert ctx.failure_category == FailureCategory.INSUFFICIENT_FUNDS  # from rule-based fallback
    assert ctx.extra_metadata["fallback_used"] is True
    assert ctx.extra_metadata["fallback_category"] == "MALFORMED_JSON"


def test_fallback_on_guardrail_authority_breach(sample_event):
    """Model attempting to return an action triggers immediate guardrail fallback."""
    breach_response = json.dumps({
        "failure_category": "INSUFFICIENT_FUNDS",
        "failure_severity": "RECOVERABLE",
        "diagnostic_explanation": "Insufficient balance.",
        "action": "RETRY_LATER",  # FORBIDDEN DECISION KEY
    })

    layer = GeminiContextLayer(
        config=LLMConfig(),
        api_caller=lambda p, s, t: breach_response,
    )

    ctx = layer.interpret_failure(sample_event)

    assert isinstance(ctx, PaymentContext)
    assert ctx.extra_metadata["fallback_used"] is True
    assert ctx.extra_metadata["fallback_category"] == "GUARDRAIL_BREACH"
    # Ensure PaymentContext does not carry the forbidden key
    assert not hasattr(ctx, "action")


def test_fallback_on_api_timeout(sample_event):
    """Timeout during API call triggers deterministic fallback."""
    def _timeout_caller(p, s, t):
        time.sleep(0.05)
        raise TimeoutError("Client timeout")

    layer = GeminiContextLayer(
        config=LLMConfig(timeout_seconds=0.01),
        api_caller=_timeout_caller,
    )

    ctx = layer.interpret_failure(sample_event)

    assert isinstance(ctx, PaymentContext)
    assert ctx.extra_metadata["fallback_used"] is True
    assert ctx.extra_metadata["fallback_category"] in ("TIMEOUT", "API_ERROR")


def test_fallback_on_api_network_exception(sample_event):
    """Network connection error triggers fallback without crashing pipeline."""
    def _failing_caller(p, s, t):
        raise ConnectionResetError("Remote server disconnected")

    layer = GeminiContextLayer(
        config=LLMConfig(),
        api_caller=_failing_caller,
    )

    ctx = layer.interpret_failure(sample_event)

    assert isinstance(ctx, PaymentContext)
    assert ctx.extra_metadata["fallback_used"] is True
    assert ctx.extra_metadata["fallback_category"] == "API_ERROR"


def test_fallback_when_credentials_missing(sample_event, monkeypatch):
    """Absence of GEMINI_API_KEY environment variable cleanly triggers fallback."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    layer = GeminiContextLayer(config=LLMConfig())
    ctx = layer.interpret_failure(sample_event)

    assert isinstance(ctx, PaymentContext)
    assert ctx.extra_metadata["fallback_used"] is True
    assert ctx.extra_metadata["fallback_category"] == "MISSING_CREDENTIALS"


def test_model_name_configuration_and_override(monkeypatch):
    """Model name defaults to gemini-3.8-flash and can be overridden by LLM_MODEL."""
    monkeypatch.delenv("LLM_MODEL", raising=False)
    cfg_default = LLMConfig()
    assert cfg_default.model_name == "gemini-3.8-flash"

    monkeypatch.setenv("LLM_MODEL", "gemini-3.8-pro-experimental")
    cfg_override = LLMConfig()
    assert cfg_override.model_name == "gemini-3.8-pro-experimental"
