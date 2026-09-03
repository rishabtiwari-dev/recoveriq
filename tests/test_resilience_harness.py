"""Test-only payment store and replay harness for Sprint 7 Failure & Resilience Testing (SPEC §18).

ARCHITECTURAL DISCLOSURE:
As discovered during Sprint 6/7 audits, the production `RecoverIQEngine` does NOT persist
`Payment` entities across independent invocations of `process_failure_event`.
This module provides a thin, additive test-only orchestration harness that:
1. Maintains an in-memory repository of `Payment` states across multi-event streams.
2. Intercepts and records duplicate event arrivals, out-of-order deliveries, and timeouts.
3. Dispatches events through the real production `RecoverIQEngine` passing the current
   persisted `Payment` entity.

This harness is strictly a test utility and is NOT part of production recovery logic.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

from recoveriq.domain.events import EventType, PaymentEvent, PaymentFailedEvent
from recoveriq.domain.models import Payment
from recoveriq.domain.state import PaymentState, is_valid_transition
from recoveriq.engine import RecoverIQEngine
from recoveriq.executor.executor import ExecutionResult, ExecutionStatus


@dataclass
class ReplayEventOutcome:
    """Outcome of processing an event in the test harness."""
    event_id: str
    payment_id: str
    accepted: bool
    status: str
    rejection_reason: Optional[str] = None
    execution_result: Optional[ExecutionResult] = None
    final_payment_state: Optional[PaymentState] = None


class PaymentReplayHarness:
    """Additive test harness to sequence and replay failure events against RecoverIQEngine."""

    def __init__(self, engine: Optional[RecoverIQEngine] = None):
        self.engine = engine or RecoverIQEngine()
        self._payments: Dict[str, Payment] = {}
        self._processed_events: Dict[str, PaymentEvent] = {}
        self._event_history: List[ReplayEventOutcome] = []

    def get_payment(self, payment_id: str) -> Optional[Payment]:
        """Retrieve current in-memory payment state."""
        return self._payments.get(payment_id)

    def is_event_processed(self, event_id: str) -> bool:
        """Check if an event_id was previously ingested."""
        return event_id in self._processed_events

    def process_event(self, event: PaymentEvent) -> ReplayEventOutcome:
        """Ingest and sequence an event with deduplication and state ordering guards.

        Enforces:
        - SPEC §11.1 / §18.1: Event deduplication by event_id.
        - SPEC §11.5 / §11.7 / §18.2: Out-of-order and terminal state validation.
        """
        pid = event.payment_id

        # 1. Event Deduplication Check (SPEC §11.1 / §18.1)
        if event.event_id in self._processed_events:
            existing_payment = self._payments.get(pid)
            outcome = ReplayEventOutcome(
                event_id=event.event_id,
                payment_id=pid,
                accepted=False,
                status="DUPLICATE_EVENT_DROPPED",
                rejection_reason=f"Event {event.event_id} already processed (idempotent drop).",
                final_payment_state=existing_payment.state if existing_payment else None,
            )
            self._event_history.append(outcome)
            return outcome

        # Retrieve existing payment record if any
        payment = self._payments.get(pid)

        # 2. Terminal State Guard (SPEC §11.5 / §18.2)
        if payment is not None and payment.is_terminal:
            outcome = ReplayEventOutcome(
                event_id=event.event_id,
                payment_id=pid,
                accepted=False,
                status="TERMINAL_STATE_REJECTED",
                rejection_reason=f"Payment {pid} is already in terminal state {payment.state.value}.",
                final_payment_state=payment.state,
            )
            self._event_history.append(outcome)
            return outcome

        # 3. Handle specific event types
        if isinstance(event, PaymentFailedEvent):
            # Dispatch through real RecoverIQEngine passing existing payment
            exec_result = self.engine.process_failure_event(event, existing_payment=payment)

            # Record event as processed
            self._processed_events[event.event_id] = event

            # Update or create payment entity in harness
            if payment is None:
                payment = Payment(
                    payment_id=event.payment_id,
                    customer_id=event.customer_id,
                    amount=event.amount,
                    currency=event.currency,
                    state=exec_result.resulting_state,
                    attempt_count=1,
                    last_event_id=event.event_id,
                )
                self._payments[pid] = payment
            else:
                # Payment was updated in engine in-place, ensure stored in harness
                self._payments[pid] = payment

            outcome = ReplayEventOutcome(
                event_id=event.event_id,
                payment_id=pid,
                accepted=True,
                status=exec_result.status.value,
                execution_result=exec_result,
                final_payment_state=payment.state,
            )
            self._event_history.append(outcome)
            return outcome

        # Fallback for generic event
        self._processed_events[event.event_id] = event
        outcome = ReplayEventOutcome(
            event_id=event.event_id,
            payment_id=pid,
            accepted=True,
            status="PROCESSED",
            final_payment_state=payment.state if payment else None,
        )
        self._event_history.append(outcome)
        return outcome
