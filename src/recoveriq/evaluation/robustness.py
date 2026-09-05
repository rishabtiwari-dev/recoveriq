"""Sprint 12 — Robustness, Sensitivity & Statistical Validation of RecoverIQ.

This module provides additive research utilities for:
1. Human-Ops Valuation Sensitivity (Experiment A): Sweeping P_human across [0.0, 1.0].
2. Payment-Value Stratification (Experiment B): Stratifying test payments into Lower, Middle, Higher tiers.
3. Expanded Multi-Seed Robustness (Experiment C): 20 deterministic evaluation seeds.
4. Paired CRN Statistical Comparisons (Experiment D): Paired differences & bootstrap confidence intervals.
5. Analytical Break-Even Threshold (Experiment E): V* calculation and empirical agreement.
6. Robustness of Tiered Policy (Experiment F): Comparative assessment across all conditions.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import numpy as np

from recoveriq.config.settings import ActionCostConfig, PenaltyConfig
from recoveriq.domain.actions import Action
from recoveriq.domain.models import CustomerTier, PaymentContext
from recoveriq.domain.state import PaymentState
from recoveriq.evaluation.sequential_policy import TieredRecoverIQStrategy
from recoveriq.evaluation.strategies import (
    FixedRetryStrategy,
    RecoverIQStrategy,
    RecoveryStrategy,
    RuleBasedStrategy,
)
from recoveriq.evaluation.trajectory import (
    AlwaysStopStrategy,
    TrajectoryEpisode,
    TrajectoryEvaluationRunner,
    TrajectoryStrategyMetrics,
)
from recoveriq.model.probability import RecoveryProbabilityModel
from recoveriq.simulation.schema import GroundTruthRecord, SyntheticPaymentRecord

# Canonical seeds expanded from 5 to 20 deterministic seeds
SPRINT12_EXPANDED_SEEDS: List[int] = [
    42, 100, 777, 999, 2024,  # Original 5 canonical seeds
    10, 20, 30, 40, 50,
    60, 70, 80, 90, 111,
    222, 333, 444, 555, 888,
]


@dataclass(frozen=True)
class ValueStratumResult:
    """Evaluation metrics for a specific strategy within a payment-value tier."""

    stratum_name: str
    min_amount: Decimal
    max_amount: Decimal
    n_payments: int
    strategy_name: str
    recovery_rate: float
    escalation_rate: float
    average_attempts: float
    automated_nrv_per_payment: float
    full_system_nrv_per_payment: float
    initial_escalate_pct: float


@dataclass(frozen=True)
class PairedComparisonResult:
    """Paired difference metrics between two strategies under Common Random Numbers."""

    strategy_a: str
    strategy_b: str
    n_observations: int
    mean_recovery_lift: float
    mean_nrv_diff_per_payment: float
    median_nrv_diff_per_payment: float
    std_nrv_diff_per_payment: float
    bootstrap_ci_nrv_diff_95: Tuple[float, float]
    mean_full_system_diff_per_payment: float
    median_full_system_diff_per_payment: float
    bootstrap_ci_full_system_diff_95: Tuple[float, float]


@dataclass(frozen=True)
class BreakEvenDiagnostic:
    """Theoretical EV break-even calculation for Action.ESCALATE vs strongest automated action."""

    payment_id: str
    payment_amount: Decimal
    customer_tier: CustomerTier
    best_automated_action: Action
    p_escalate: float
    p_best_automated: float
    cost_escalate: Decimal
    cost_best_automated: Decimal
    penalty_escalate: Decimal
    penalty_best_automated: Decimal
    theoretical_v_star: Optional[float]
    actual_proposed_action: Action
    agrees_with_prediction: bool


def calculate_human_ops_valuation_sweep(
    episodes: List[TrajectoryEpisode],
    p_human: float,
    automated_metrics: TrajectoryStrategyMetrics,
) -> Tuple[Decimal, Decimal]:
    """Calculate expected human-ops value and full-system NRV for a fixed p_human assumption.

    Formula:
    Expected Human-Ops Value = sum_{ep in ESCALATED} (p_human * ep.payment_amount)
    Full-System Expected NRV = Automated NRV + Expected Human-Ops Value
    """
    if not (0.0 <= p_human <= 1.0):
        raise ValueError(f"p_human must be in [0.0, 1.0], got {p_human}")

    p_human_dec = Decimal(f"{p_human:.4f}")
    expected_human_value = Decimal("0.00")

    for ep in episodes:
        if ep.terminal_state == PaymentState.ESCALATED:
            expected_human_value += p_human_dec * ep.payment_amount

    full_system_nrv = automated_metrics.total_nrv + expected_human_value
    return expected_human_value, full_system_nrv


def stratify_payments_by_value(
    observable_records: List[SyntheticPaymentRecord],
) -> Dict[str, Tuple[Decimal, Decimal, List[SyntheticPaymentRecord]]]:
    """Stratify payment records into 3 empirical value bins using tertiles (33.3% and 66.7% percentiles).

    Guarantees:
    - Every payment belongs to exactly one tier.
    - Non-overlapping, exhaustive partitioning.
    """
    amounts = sorted([float(r.amount) for r in observable_records])
    q33 = float(np.percentile(amounts, 33.333))
    q66 = float(np.percentile(amounts, 66.667))

    lower_records: List[SyntheticPaymentRecord] = []
    middle_records: List[SyntheticPaymentRecord] = []
    higher_records: List[SyntheticPaymentRecord] = []

    for r in observable_records:
        amt = float(r.amount)
        if amt <= q33:
            lower_records.append(r)
        elif amt <= q66:
            middle_records.append(r)
        else:
            higher_records.append(r)

    # Convert bounds to Decimal
    min_lower = min(r.amount for r in lower_records) if lower_records else Decimal("0.00")
    max_lower = max(r.amount for r in lower_records) if lower_records else Decimal("0.00")

    min_middle = min(r.amount for r in middle_records) if middle_records else Decimal("0.00")
    max_middle = max(r.amount for r in middle_records) if middle_records else Decimal("0.00")

    min_higher = min(r.amount for r in higher_records) if higher_records else Decimal("0.00")
    max_higher = max(r.amount for r in higher_records) if higher_records else Decimal("0.00")

    return {
        "Lower-Value": (min_lower, max_lower, lower_records),
        "Middle-Value": (min_middle, max_middle, middle_records),
        "Higher-Value": (min_higher, max_higher, higher_records),
    }


def compute_paired_crn_differences(
    episodes_a: List[TrajectoryEpisode],
    episodes_b: List[TrajectoryEpisode],
    ground_truth_records: List[GroundTruthRecord],
    strategy_a_name: str,
    strategy_b_name: str,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
) -> PairedComparisonResult:
    """Compute paired payment-level differences between two strategies evaluated under CRN.

    Because both strategies were evaluated against the exact same payments with identical
    CRN seeds, we can construct paired difference vectors:
        diff_nrv_i = NRV_A(i) - NRV_B(i)
        diff_rec_i = int(recovered_A(i)) - int(recovered_B(i))
    """
    gt_map = {gt.payment_id: gt for gt in ground_truth_records}

    # Map episodes by payment_id
    map_a = {ep.payment_id: ep for ep in episodes_a}
    map_b = {ep.payment_id: ep for ep in episodes_b}

    common_pids = [pid for pid in map_a if pid in map_b]
    n = len(common_pids)
    if n == 0:
        raise ValueError("No common payments between compared strategies.")

    nrv_diffs: List[float] = []
    rec_diffs: List[float] = []
    full_system_diffs: List[float] = []

    for pid in common_pids:
        ep_a = map_a[pid]
        ep_b = map_b[pid]
        gt = gt_map.get(pid)
        p_esc = float(gt.action_base_probabilities.get(Action.ESCALATE, 0.0)) if gt else 0.0

        # Automated NRV difference
        nrv_a = float(ep_a.net_recovered_value)
        nrv_b = float(ep_b.net_recovered_value)
        nrv_diffs.append(nrv_a - nrv_b)

        # Recovery difference
        rec_diffs.append(float(int(ep_a.final_recovered) - int(ep_b.final_recovered)))

        # Full-system NRV difference
        h_val_a = (p_esc * float(ep_a.payment_amount)) if ep_a.terminal_state == PaymentState.ESCALATED else 0.0
        h_val_b = (p_esc * float(ep_b.payment_amount)) if ep_b.terminal_state == PaymentState.ESCALATED else 0.0

        full_a = nrv_a + h_val_a
        full_b = nrv_b + h_val_b
        full_system_diffs.append(full_a - full_b)

    # Summary stats
    mean_rec_lift = float(np.mean(rec_diffs))
    mean_nrv_diff = float(np.mean(nrv_diffs))
    median_nrv_diff = float(np.median(nrv_diffs))
    std_nrv_diff = float(np.std(nrv_diffs))

    mean_full_diff = float(np.mean(full_system_diffs))
    median_full_diff = float(np.median(full_system_diffs))

    # Bootstrap 95% Confidence Intervals
    rng = np.random.RandomState(random_seed)
    boot_nrv_means = []
    boot_full_means = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_nrv_means.append(np.mean([nrv_diffs[i] for i in idx]))
        boot_full_means.append(np.mean([full_system_diffs[i] for i in idx]))

    ci_nrv = (float(np.percentile(boot_nrv_means, 2.5)), float(np.percentile(boot_nrv_means, 97.5)))
    ci_full = (float(np.percentile(boot_full_means, 2.5)), float(np.percentile(boot_full_means, 97.5)))

    return PairedComparisonResult(
        strategy_a=strategy_a_name,
        strategy_b=strategy_b_name,
        n_observations=n,
        mean_recovery_lift=mean_rec_lift,
        mean_nrv_diff_per_payment=mean_nrv_diff,
        median_nrv_diff_per_payment=median_nrv_diff,
        std_nrv_diff_per_payment=std_nrv_diff,
        bootstrap_ci_nrv_diff_95=ci_nrv,
        mean_full_system_diff_per_payment=mean_full_diff,
        median_full_system_diff_per_payment=median_full_diff,
        bootstrap_ci_full_system_diff_95=ci_full,
    )


def compute_break_even_diagnostic(
    record: SyntheticPaymentRecord,
    context: PaymentContext,
    probability_model: RecoveryProbabilityModel,
    cost_config: Optional[ActionCostConfig] = None,
    penalty_config: Optional[PenaltyConfig] = None,
) -> BreakEvenDiagnostic:
    """Compute theoretical V* break-even threshold between ESCALATE and the strongest automated action.

    Equation (SPEC / Sprint 12):
        EV(ESCALATE) > EV(best_automated)  <=>
        P(ESCALATE)*V - C_esc - Pen_esc > P(best_auto)*V - C_auto - Pen_auto <=>
        V * [P(ESCALATE) - P(best_auto)] > (C_esc + Pen_esc) - (C_auto + Pen_auto)

        V* = [(C_esc + Pen_esc) - (C_auto + Pen_auto)] / [P(ESCALATE) - P(best_auto)]
    """
    cost_cfg = cost_config or ActionCostConfig()
    penalty_cfg = penalty_config or PenaltyConfig()

    probabilities = probability_model.estimate_probabilities(context)

    # Cost and penalty for ESCALATE
    c_esc = cost_cfg.get_cost(Action.ESCALATE)
    pen_esc = penalty_cfg.get_penalty(Action.ESCALATE, record.customer_tier)
    p_esc = float(probabilities.get(Action.ESCALATE).probability)

    # Find strongest automated candidate action (excluding STOP and ESCALATE)
    automated_actions = [a for a in Action if a not in (Action.STOP, Action.ESCALATE)]

    # Best automated is argmax of EV among automated actions at current amount V
    v_amt = record.amount if isinstance(record.amount, Decimal) else Decimal(str(record.amount))
    best_auto_act = None
    best_auto_ev = Decimal("-Infinity")

    for a in automated_actions:
        p_a = probabilities.get(a).probability
        c_a = cost_cfg.get_cost(a)
        pen_a = penalty_cfg.get_penalty(a, record.customer_tier)
        ev_a = p_a * v_amt - c_a - pen_a
        if ev_a > best_auto_ev:
            best_auto_ev = ev_a
            best_auto_act = a

    p_auto = float(probabilities.get(best_auto_act).probability)
    c_auto = cost_cfg.get_cost(best_auto_act)
    pen_auto = penalty_cfg.get_penalty(best_auto_act, record.customer_tier)

    # Break-even threshold V*
    prob_gap = p_esc - p_auto
    cost_pen_gap = float((c_esc + pen_esc) - (c_auto + pen_auto))

    if prob_gap > 0:
        v_star = cost_pen_gap / prob_gap
    else:
        # If P(ESCALATE) <= P(best_auto), ESCALATE has both lower/equal prob and higher cost -> never preferred
        v_star = None

    # Compare actual RecoverIQ selection
    from recoveriq.economics.engine import DefaultEconomicEngine
    econ = DefaultEconomicEngine()
    dec = econ.evaluate_actions(context, v_amt, probabilities)
    actual_proposed = dec.proposed_action

    # Prediction: If V > V* and v_star is not None, ESCALATE should be preferred over best_auto
    v_float = float(v_amt)
    if v_star is not None and v_float > v_star:
        predicted_action = Action.ESCALATE
    else:
        predicted_action = best_auto_act

    agrees = (actual_proposed == Action.ESCALATE) if (v_star is not None and v_float > v_star) else (actual_proposed != Action.ESCALATE)

    return BreakEvenDiagnostic(
        payment_id=record.payment_id,
        payment_amount=v_amt,
        customer_tier=record.customer_tier,
        best_automated_action=best_auto_act,
        p_escalate=p_esc,
        p_best_automated=p_auto,
        cost_escalate=c_esc,
        cost_best_automated=c_auto,
        penalty_escalate=pen_esc,
        penalty_best_automated=pen_auto,
        theoretical_v_star=v_star,
        actual_proposed_action=actual_proposed,
        agrees_with_prediction=agrees,
    )
