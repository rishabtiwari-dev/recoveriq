"""Deterministic Policy Gate interface and invariant rule enforcement."""

from datetime import datetime, timezone
from typing import List, Protocol, runtime_checkable

from recoveriq.config.settings import PolicyConfig
from recoveriq.domain.actions import Action
from recoveriq.domain.decisions import (
    PolicyDecision,
    PolicyRuleResult,
    RecoveryDecision,
)
from recoveriq.domain.models import CustomerTier, Payment, PaymentContext


@runtime_checkable
class PolicyGate(Protocol):
    """Protocol for the deterministic policy authorization gate."""

    def authorize(
        self,
        payment: Payment,
        context: PaymentContext,
        decision: RecoveryDecision,
    ) -> PolicyDecision:
        """Validate proposed action against safety invariants and return authorization verdict."""
        ...


class InvariantPolicyGate:
    """Deterministic policy gate enforcing hard business, safety, and state machine invariants."""

    def __init__(self, config: PolicyConfig = None):
        self.config = config or PolicyConfig()

    def authorize(
        self,
        payment: Payment,
        context: PaymentContext,
        decision: RecoveryDecision,
    ) -> PolicyDecision:
        proposed = decision.proposed_action
        rule_results: List[PolicyRuleResult] = []

        # 1. State Invariant Check: Cannot execute actions on terminal payments
        if payment.is_terminal:
            rule_results.append(
                PolicyRuleResult(
                    rule_name="STATE_TERMINAL_CHECK",
                    passed=False,
                    message=f"Payment is already in terminal state {payment.state.value}.",
                )
            )
            return PolicyDecision.reject_and_clamp(
                payment_id=payment.payment_id,
                proposed_action=proposed,
                fallback_action=Action.STOP,
                reason=f"Payment already in terminal state {payment.state.value}.",
                rule_results=rule_results,
            )
        else:
            rule_results.append(
                PolicyRuleResult(
                    rule_name="STATE_TERMINAL_CHECK",
                    passed=True,
                    message=f"Payment state {payment.state.value} is non-terminal.",
                )
            )

        # 2. Hard Decline Rule: Strictly forbid retry actions on hard declines
        if (
            self.config.disallow_retries_on_hard_declines
            and context.failure_category in self.config.hard_decline_categories
            and proposed.is_retry
        ):
            rule_results.append(
                PolicyRuleResult(
                    rule_name="HARD_DECLINE_RETRY_CHECK",
                    passed=False,
                    message=f"Retry action {proposed.value} disallowed on hard decline category {context.failure_category.value}.",
                )
            )
            return PolicyDecision.reject_and_clamp(
                payment_id=payment.payment_id,
                proposed_action=proposed,
                fallback_action=self.config.fallback_on_hard_decline,
                reason=f"Hard decline {context.failure_category.value} cannot be retried.",
                rule_results=rule_results,
            )
        else:
            rule_results.append(
                PolicyRuleResult(
                    rule_name="HARD_DECLINE_RETRY_CHECK",
                    passed=True,
                    message="Hard decline retry check passed.",
                )
            )

        # 3. Retry Budget Check: Cannot exceed max attempts
        if payment.attempt_count >= self.config.max_attempts and proposed != Action.STOP:
            # Check if VIP customer eligible for escalation upon budget exhaustion
            fallback = (
                Action.ESCALATE
                if (self.config.vip_escalation_enabled and context.customer_tier == CustomerTier.VIP)
                else self.config.fallback_on_budget_exhausted
            )
            rule_results.append(
                PolicyRuleResult(
                    rule_name="RETRY_BUDGET_CHECK",
                    passed=False,
                    message=f"Attempt count ({payment.attempt_count}) reached or exceeded max budget ({self.config.max_attempts}).",
                )
            )
            return PolicyDecision.reject_and_clamp(
                payment_id=payment.payment_id,
                proposed_action=proposed,
                fallback_action=fallback,
                reason=f"Exhausted retry budget of {self.config.max_attempts} attempts.",
                rule_results=rule_results,
            )
        else:
            rule_results.append(
                PolicyRuleResult(
                    rule_name="RETRY_BUDGET_CHECK",
                    passed=True,
                    message=f"Attempt count {payment.attempt_count} within limit {self.config.max_attempts}.",
                )
            )

        # 4. Cooldown Check: Enforce minimum time interval for non-immediate retries/nudges
        if (
            proposed in (Action.RETRY_LATER, Action.NUDGE)
            and context.last_attempt_timestamp is not None
        ):
            now = datetime.now(timezone.utc)
            elapsed_seconds = (now - context.last_attempt_timestamp).total_seconds()
            if elapsed_seconds < self.config.cooldown_seconds:
                rule_results.append(
                    PolicyRuleResult(
                        rule_name="COOLDOWN_WINDOW_CHECK",
                        passed=False,
                        message=f"Elapsed time ({elapsed_seconds:.1f}s) < required cooldown ({self.config.cooldown_seconds}s).",
                    )
                )
                return PolicyDecision.reject_and_clamp(
                    payment_id=payment.payment_id,
                    proposed_action=proposed,
                    fallback_action=Action.STOP,
                    reason=f"Mandatory cooldown of {self.config.cooldown_seconds}s not satisfied (elapsed: {elapsed_seconds:.1f}s).",
                    rule_results=rule_results,
                )
            else:
                rule_results.append(
                    PolicyRuleResult(
                        rule_name="COOLDOWN_WINDOW_CHECK",
                        passed=True,
                        message=f"Cooldown window satisfied ({elapsed_seconds:.1f}s >= {self.config.cooldown_seconds}s).",
                    )
                )

        # All invariants passed
        return PolicyDecision.authorize(
            payment_id=payment.payment_id,
            action=proposed,
            rule_results=rule_results,
        )
