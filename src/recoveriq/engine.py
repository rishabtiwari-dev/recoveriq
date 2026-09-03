"""RecoverIQ Engine pipeline coordinating modular components under strict policy gating."""

from decimal import Decimal
from typing import Optional

from recoveriq.ai.context_layer import AIContextLayer, StubAIContextLayer
from recoveriq.audit.logger import (
    AuditEvent,
    AuditEventType,
    AuditLogger,
    InMemoryAuditLogger,
)
from recoveriq.config.settings import RecoverIQConfig
from recoveriq.context.extractor import ContextExtractor, RuleBasedContextExtractor
from recoveriq.domain.events import PaymentFailedEvent
from recoveriq.domain.models import Payment
from recoveriq.domain.state import (
    PaymentState,
    StateTransition,
    is_valid_transition,
)
from recoveriq.economics.engine import DefaultEconomicEngine, EconomicEngine
from recoveriq.executor.executor import (
    ActionExecutor,
    ExecutionResult,
    InMemoryActionExecutor,
)
from recoveriq.model.probability import (
    RecoveryProbabilityModel,
    StubProbabilityModel,
)
from recoveriq.policy.gate import InvariantPolicyGate, PolicyGate


class RecoverIQEngine:
    """Core coordinator executing the strict, decoupled payment recovery pipeline:

    Event -> Context (AI/Rules) -> Probability -> Economics (EV) -> Policy Gate -> Executor -> Audit
    """

    def __init__(
        self,
        config: Optional[RecoverIQConfig] = None,
        context_extractor: Optional[ContextExtractor] = None,
        ai_layer: Optional[AIContextLayer] = None,
        probability_model: Optional[RecoveryProbabilityModel] = None,
        economic_engine: Optional[EconomicEngine] = None,
        policy_gate: Optional[PolicyGate] = None,
        executor: Optional[ActionExecutor] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.config = config or RecoverIQConfig.default()
        self.context_extractor = context_extractor or RuleBasedContextExtractor()
        self.ai_layer = ai_layer or StubAIContextLayer(fallback_extractor=self.context_extractor)
        self.probability_model = probability_model or StubProbabilityModel()
        self.economic_engine = economic_engine or DefaultEconomicEngine(self.config.economics)
        self.policy_gate = policy_gate or InvariantPolicyGate(self.config.policy)
        self.executor = executor or InMemoryActionExecutor()
        self.audit_logger = audit_logger or InMemoryAuditLogger()

    def process_failure_event(
        self,
        event: PaymentFailedEvent,
        existing_payment: Optional[Payment] = None,
    ) -> ExecutionResult:
        """Process an incoming failed payment event through the end-to-end recovery pipeline."""
        # 1. Ingestion / Payment entity initialization
        payment = existing_payment or Payment(
            payment_id=event.payment_id,
            customer_id=event.customer_id,
            amount=event.amount if isinstance(event.amount, Decimal) else Decimal(str(event.amount)),
            currency=event.currency,
            state=PaymentState.FAILED_INITIAL,
            attempt_count=event.attempt_count - 1,
            last_event_id=event.event_id,
        )

        self.audit_logger.log(
            AuditEvent.create(
                event_type=AuditEventType.INGESTION,
                payment_id=payment.payment_id,
                details={
                    "event_id": event.event_id,
                    "amount": str(payment.amount),
                    "currency": payment.currency,
                    "initial_state": payment.state.value,
                },
            )
        )

        # 2. Context Extraction (AI layer interprets; strictly NO execution authority)
        context = self.ai_layer.interpret_failure(event)

        self.audit_logger.log(
            AuditEvent.create(
                event_type=AuditEventType.CONTEXT_EXTRACTION,
                payment_id=payment.payment_id,
                details={
                    "category": context.failure_category.value,
                    "severity": context.failure_severity.value,
                    "diagnostic": context.diagnostic_explanation,
                },
            )
        )

        # 3. Probability Estimation
        probabilities = self.probability_model.estimate_probabilities(context)

        self.audit_logger.log(
            AuditEvent.create(
                event_type=AuditEventType.PROBABILITY_ESTIMATION,
                payment_id=payment.payment_id,
                details={
                    action.value: float(est.probability)
                    for action, est in probabilities.items()
                },
            )
        )

        # 4. Economic Evaluation (Expected Value ranking)
        recovery_decision = self.economic_engine.evaluate_actions(
            context=context,
            payment_amount=payment.amount,
            probabilities=probabilities,
        )

        self.audit_logger.log(
            AuditEvent.create(
                event_type=AuditEventType.ECONOMIC_EVALUATION,
                payment_id=payment.payment_id,
                details={
                    "proposed_action": recovery_decision.proposed_action.value,
                    "rationale": recovery_decision.rationale,
                    "candidate_evs": [
                        {
                            "action": c.action.value,
                            "net_ev": str(c.net_expected_value),
                            "p": str(c.estimated_probability),
                            "cost": str(c.intervention_cost),
                            "penalty": str(c.friction_penalty),
                        }
                        for c in recovery_decision.candidate_evaluations
                    ],
                },
            )
        )

        # 5. Deterministic Policy Gate (MANDATORY: validates & authorizes or rejects)
        policy_decision = self.policy_gate.authorize(
            payment=payment,
            context=context,
            decision=recovery_decision,
        )

        self.audit_logger.log(
            AuditEvent.create(
                event_type=(
                    AuditEventType.POLICY_AUTHORIZATION
                    if policy_decision.is_authorized
                    else AuditEventType.POLICY_REJECTION
                ),
                payment_id=payment.payment_id,
                details={
                    "proposed_action": policy_decision.proposed_action.value,
                    "authorized_action": policy_decision.authorized_action.value,
                    "is_authorized": policy_decision.is_authorized,
                    "rejection_reason": policy_decision.rejection_reason,
                    "rule_results": [
                        {"rule": r.rule_name, "passed": r.passed, "msg": r.message}
                        for r in policy_decision.rule_results
                    ],
                },
            )
        )

        # 6. Action Execution (strictly dispatches authorized action)
        execution_result = self.executor.execute(
            payment=payment,
            policy_decision=policy_decision,
            event_id=event.event_id,
        )

        self.audit_logger.log(
            AuditEvent.create(
                event_type=AuditEventType.ACTION_EXECUTION,
                payment_id=payment.payment_id,
                details={
                    "action": execution_result.action.value,
                    "status": execution_result.status.value,
                    "idempotency_key": execution_result.idempotency_key,
                    "resulting_state": execution_result.resulting_state.value,
                },
            )
        )

        # 7. State Transition Recording
        if is_valid_transition(payment.state, execution_result.resulting_state):
            transition = StateTransition.create(
                from_state=payment.state,
                to_state=execution_result.resulting_state,
                trigger_event_id=event.event_id,
                reason=f"Action {execution_result.action.value} executed with status {execution_result.status.value}",
            )
            payment.state = execution_result.resulting_state
            payment.attempt_count += 1
            payment.last_event_id = event.event_id

            self.audit_logger.log(
                AuditEvent.create(
                    event_type=AuditEventType.STATE_TRANSITION,
                    payment_id=payment.payment_id,
                    details={
                        "from_state": transition.from_state.value,
                        "to_state": transition.to_state.value,
                        "attempt_count": payment.attempt_count,
                    },
                )
            )

        return execution_result
