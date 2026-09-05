"""Sprint 11 — Controlled Sequential Policy & Escalation Valuation Ablation.

This module provides additive abstractions for investigating sequential policy dynamics:
1. TieredRecoverIQStrategy: A sequential decision strategy that preserves the cheap retry option
   value by restricting Action.ESCALATE during early attempts (Attempts 1 and 2), and allowing
   it to become available again on the final attempt (attempt == max_attempts).
2. HumanOpsValuation: Valuation accounting functions to estimate downstream human-operations
   recovery value for escalated payments without corrupting automated trajectory semantics.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional
import numpy as np

from recoveriq.config.settings import EconomicConfig
from recoveriq.domain.actions import Action
from recoveriq.domain.decisions import CandidateActionEV, RecoveryDecision
from recoveriq.domain.models import PaymentContext, PaymentState
from recoveriq.domain.state import PaymentState
from recoveriq.economics.engine import DefaultEconomicEngine, EconomicEngine
from recoveriq.evaluation.trajectory import TrajectoryEpisode, TrajectoryStrategyMetrics
from recoveriq.model.probability import ProbabilityEstimate, RecoveryProbabilityModel
from recoveriq.simulation.schema import GroundTruthRecord, SyntheticPaymentRecord


class TieredRecoverIQStrategy:
    """Tiered Sequential RecoverIQ Strategy (Sprint 11 Factor A - Condition B).

    Restricts Action.ESCALATE on early attempts (attempt < max_attempts) to preserve
    automated retry option value, permitting ESCALATE only on the final attempt
    (attempt >= max_attempts).

    Uses the existing trained probability model and EV optimization logic over the
    filtered candidate action space.
    """

    name: str = "RecoverIQ-Tiered"

    def __init__(
        self,
        probability_model: RecoveryProbabilityModel,
        economic_engine: Optional[EconomicEngine] = None,
        max_attempts: int = 3,
    ):
        self.probability_model = probability_model
        self.economic_engine = economic_engine or DefaultEconomicEngine()
        self.max_attempts = max_attempts

    def propose_action(
        self,
        record: SyntheticPaymentRecord,
        context: PaymentContext,
    ) -> Action:
        """Propose candidate action with tiered escalation availability.

        Attempts 1 to max_attempts - 1:
            ESCALATE is excluded. Candidate set: {RETRY_NOW, RETRY_LATER, SEND_LINK, NUDGE, STOP}.
        Attempt max_attempts (Final):
            Full action set available including ESCALATE.
        """
        probabilities = self.probability_model.estimate_probabilities(context)
        amount_decimal = record.amount if isinstance(record.amount, Decimal) else Decimal(str(record.amount))

        # Check if early attempt
        current_attempt = context.attempt_count
        if current_attempt < self.max_attempts:
            # Filter out ESCALATE from candidate probabilities prior to EV selection
            filtered_probs = {
                action: prob
                for action, prob in probabilities.items()
                if action != Action.ESCALATE
            }
        else:
            filtered_probs = probabilities

        decision = self.economic_engine.evaluate_actions(
            context=context,
            payment_amount=amount_decimal,
            probabilities=filtered_probs,
        )
        return decision.proposed_action


@dataclass(frozen=True)
class EscalationValuationRecord:
    """Valuation comparison record for a single strategy run under a specific valuation regime."""

    strategy_name: str
    valuation_regime: str  # "Automated-Only" or "Human-Ops"
    recovery_rate: float
    automated_nrv: Decimal
    expected_human_ops_value: Decimal
    full_system_expected_nrv: Decimal
    n_escalated: int
    escalation_rate: float


@dataclass
class MultiSeedValuationReport:
    """Aggregated valuation comparison across seeds with Mean +/- Std."""

    strategy_name: str
    valuation_regime: str
    mean_recovery_rate: float
    std_recovery_rate: float
    mean_automated_nrv: float
    std_automated_nrv: float
    mean_human_ops_value: float
    std_human_ops_value: float
    mean_full_system_nrv: float
    std_full_system_nrv: float
    mean_escalation_rate: float
    std_escalation_rate: float


def calculate_human_ops_valuation(
    episodes: List[TrajectoryEpisode],
    ground_truth_records: List[GroundTruthRecord],
    automated_metrics: TrajectoryStrategyMetrics,
) -> EscalationValuationRecord:
    """Calculate expected downstream human-operations recovery value for escalated payments.

    Accounting Rule (SPEC / Sprint 11):
    - For each payment ending in terminal state ESCALATED:
        P_human = ground_truth[Action.ESCALATE]
        Expected Gross Value = P_human * payment_amount
    - Downstream human ops cost is not defined by repository/SPEC -> INR 0.00 / not modeled.
    - Full-System Expected NRV = Automated NRV + Expected Human Ops Value.
    """
    gt_map = {gt.payment_id: gt for gt in ground_truth_records}

    expected_human_value = Decimal("0.00")
    escalated_count = 0

    for ep in episodes:
        if ep.terminal_state == PaymentState.ESCALATED:
            escalated_count += 1
            gt = gt_map.get(ep.payment_id)
            if gt is not None:
                p_human = Decimal(str(gt.action_base_probabilities.get(Action.ESCALATE, 0.0)))
            else:
                p_human = Decimal("0.00")
            expected_human_value += p_human * ep.payment_amount

    automated_nrv = automated_metrics.total_nrv
    full_system_nrv = automated_nrv + expected_human_value
    escalation_rate = escalated_count / len(episodes) if episodes else 0.0

    return EscalationValuationRecord(
        strategy_name=automated_metrics.strategy_name,
        valuation_regime="Human-Ops",
        recovery_rate=automated_metrics.recovery_rate,
        automated_nrv=automated_nrv,
        expected_human_ops_value=expected_human_value,
        full_system_expected_nrv=full_system_nrv,
        n_escalated=escalated_count,
        escalation_rate=escalation_rate,
    )
