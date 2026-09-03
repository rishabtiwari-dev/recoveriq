"""Gemini 3.8 Flash Context Layer implementation of AIContextLayer.

ARCHITECTURAL INVARIANTS:
1. Zero execution authority: This module only interprets failure context.
2. Privacy boundary: Only 4 technical fields (raw_error_code, raw_error_message,
   payment_method, attempt_count) are transmitted to the external API.
   Financial/customer identity fields are NEVER transmitted.
3. Anti-authority boundary: The response is strictly validated against a blacklist
   of decision-making keys (action, dispatch, policy_decision, ev, etc.).
4. Resilient fallback: Any error, timeout, or schema failure triggers deterministic
   fallback to RuleBasedContextExtractor without halting execution.
5. Privacy-conscious telemetry: Raw prompts and raw completions are NEVER persisted.
   Only safe operational metadata is recorded.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Callable, Dict, Optional

from recoveriq.ai.prompt_template import (
    SYSTEM_INSTRUCTION,
    extract_permitted_prompt_payload,
    render_user_prompt,
)
from recoveriq.ai.schema_validator import validate_llm_response
from recoveriq.config.settings import LLMConfig
from recoveriq.context.extractor import ContextExtractor, RuleBasedContextExtractor
from recoveriq.domain.events import PaymentFailedEvent
from recoveriq.domain.models import PaymentContext

logger = logging.getLogger(__name__)


class GeminiContextLayer:
    """Production implementation of AIContextLayer connecting to Gemini 3.8 Flash."""

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        client: Optional[Any] = None,
        fallback_extractor: Optional[ContextExtractor] = None,
        api_caller: Optional[Callable[[str, str, float], str]] = None,
    ):
        self.config = config or LLMConfig()
        self._fallback_extractor = fallback_extractor or RuleBasedContextExtractor()
        self._client = client
        self._custom_api_caller = api_caller

    def _get_client(self) -> Optional[Any]:
        """Lazy-initialize google-genai client."""
        if self._client is not None:
            return self._client

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None

        try:
            from google import genai  # Modern SDK import
            self._client = genai.Client(api_key=api_key)
            return self._client
        except Exception as e:
            logger.warning(f"Failed to initialize google-genai client: {e}")
            return None

    def _call_gemini_api(self, prompt: str, system_instruction: str, timeout_seconds: float) -> str:
        """Execute Gemini API call with modern google-genai SDK or custom caller."""
        if self._custom_api_caller is not None:
            return self._custom_api_caller(prompt, system_instruction, timeout_seconds)

        client = self._get_client()
        if client is None:
            raise RuntimeError("GEMINI_CLIENT_UNAVAILABLE")

        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self.config.temperature,
            response_mime_type="application/json",
        )

        def _invoke() -> str:
            response = client.models.generate_content(
                model=self.config.model_name,
                contents=prompt,
                config=config,
            )
            return response.text or ""

        # Enforce deterministic client timeout via thread pool
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_invoke)
            return future.result(timeout=timeout_seconds)

    def interpret_failure(self, event: PaymentFailedEvent) -> PaymentContext:
        """Parse raw decline signals and metadata into structured PaymentContext."""
        start_time = time.perf_counter()

        # 1. Enforce strict 4-field input privacy boundary
        permitted_payload = extract_permitted_prompt_payload(
            raw_error_code=event.raw_error_code,
            raw_error_message=event.raw_error_message,
            payment_method=event.payment_method,
            attempt_count=event.attempt_count,
        )

        user_prompt = render_user_prompt(
            raw_error_code=permitted_payload["raw_error_code"],
            raw_error_message=permitted_payload["raw_error_message"],
            payment_method=permitted_payload["payment_method"],
            attempt_count=permitted_payload["attempt_count"],
        )

        # 2. Check credentials & client availability
        api_key = os.environ.get("GEMINI_API_KEY")
        if self._custom_api_caller is None and (not api_key or self._get_client() is None):
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return self._execute_fallback(
                event=event,
                fallback_category="MISSING_CREDENTIALS" if not api_key else "SDK_UNAVAILABLE",
                latency_ms=elapsed_ms,
            )

        # 3. Invoke external LLM with strict timeout
        try:
            raw_response_text = self._call_gemini_api(
                prompt=user_prompt,
                system_instruction=SYSTEM_INSTRUCTION,
                timeout_seconds=self.config.timeout_seconds,
            )
        except FuturesTimeoutError:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return self._execute_fallback(
                event=event,
                fallback_category="TIMEOUT",
                latency_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return self._execute_fallback(
                event=event,
                fallback_category="API_ERROR",
                latency_ms=elapsed_ms,
            )

        # 4. Strict Schema & Anti-Authority Validation
        validation = validate_llm_response(raw_response_text)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        if not validation.is_valid:
            fallback_category = validation.error_category or "SCHEMA_VALIDATION_ERROR"
            return self._execute_fallback(
                event=event,
                fallback_category=fallback_category,
                latency_ms=elapsed_ms,
            )

        # 5. Success: Construct safe operational metadata (NO raw prompt/completion stored)
        safe_metadata: Dict[str, Any] = {
            "llm_provider": self.config.provider,
            "llm_model": self.config.model_name,
            "schema_version": self.config.schema_version,
            "fallback_used": False,
            "latency_ms": elapsed_ms,
            "confidence_signal": "HIGH",
        }

        return PaymentContext(
            payment_id=event.payment_id,
            customer_id=event.customer_id,
            customer_tier=event.customer_tier,
            payment_method=event.payment_method,
            raw_error_code=event.raw_error_code,
            raw_error_message=event.raw_error_message,
            failure_category=validation.failure_category,  # validated enum
            failure_severity=validation.failure_severity,  # validated enum
            attempt_count=event.attempt_count,
            last_attempt_timestamp=event.timestamp,
            diagnostic_explanation=validation.diagnostic_explanation,
            extra_metadata=safe_metadata,
        )

    def _execute_fallback(
        self,
        event: PaymentFailedEvent,
        fallback_category: str,
        latency_ms: float,
    ) -> PaymentContext:
        """Trigger deterministic fallback to rule-based context extractor."""
        base_context = self._fallback_extractor.extract_context(event)

        safe_metadata: Dict[str, Any] = {
            "llm_provider": self.config.provider,
            "llm_model": self.config.model_name,
            "schema_version": self.config.schema_version,
            "fallback_used": True,
            "fallback_category": fallback_category,
            "latency_ms": latency_ms,
        }

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
            diagnostic_explanation=base_context.diagnostic_explanation,
            extra_metadata=safe_metadata,
        )
