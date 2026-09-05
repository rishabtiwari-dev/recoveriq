"""Sprint 15 — Uncertainty-Aware Hybrid Sequential Policy for RecoverIQ.

Implements RecoverIQ-Hybrid combining the model-based Bellman DP policy and the
model-free Fitted Q-Iteration policy through an uncertainty-aware action-value
arbitration mechanism:

    Q_hybrid(s, a) = w(s, a) * Q_bellman(s, a) + (1 - w(s, a)) * Q_modelfree(s, a)

Where:
- w(s, a) is an arbitration weight in [0, 1] derived from training/model confidence.
- ModelFree uncertainty is estimated strictly from disjoint training trajectories
  using state-action visitation counts and empirical sample variance.
- Bellman uncertainty is derived from the available probability-model/value estimates
  (e.g., dispersion of probability estimates / sharpness of immediate EV) without
  using evaluation outcomes or oracle ground truth.
- Three comparison regimes:
  1. EQUAL_WEIGHT: w(s, a) = 0.50
  2. FIXED_WEIGHT: w(s, a) = fixed_weight (e.g., 0.70 or configured constant)
  3. UNCERTAINTY_AWARE: w(s, a) = conf_bellman / (conf_bellman + conf_modelfree)

STRICT ANTI-LEAKAGE INVARIANTS:
- No oracle ground truth access at decision time.
- Training trajectories come strictly from disjoint training seeds/partitions.
- Evaluation occurs on completely held-out test partitions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple

from recoveriq.domain.actions import Action
from recoveriq.domain.models import CustomerTier, FailureCategory, PaymentContext
from recoveriq.evaluation.bellman_policy import (
    BellmanActionEvaluation,
    BellmanRecoverIQStrategy,
)
from recoveriq.evaluation.model_free_policy import (
    FittedQIterationPolicy,
    ModelFreeRecoverIQStrategy,
    QState,
    _make_state,
)
from recoveriq.model.probability import RecoveryProbabilityModel
from recoveriq.simulation.schema import SyntheticPaymentRecord


class HybridRegime(str, Enum):
    """Comparison regimes for the hybrid sequential policy."""

    EQUAL_WEIGHT = "EQUAL_WEIGHT"
    FIXED_WEIGHT = "FIXED_WEIGHT"
    UNCERTAINTY_AWARE = "UNCERTAINTY_AWARE"


@dataclass(frozen=True)
class HybridActionEvaluation:
    """Action evaluation decomposition for RecoverIQ-Hybrid."""

    action: Action
    q_bellman: Decimal
    q_modelfree: Decimal
    weight_bellman: Decimal
    weight_modelfree: Decimal
    q_hybrid: Decimal
    bellman_confidence: Decimal
    modelfree_confidence: Decimal


@dataclass(frozen=True)
class HybridDecision:
    """Decision output produced by HybridRecoverIQStrategy."""

    payment_id: str
    attempt_count: int
    proposed_action: Action
    regime: HybridRegime
    evaluations: List[HybridActionEvaluation]
    rationale: str

    @property
    def selected_evaluation(self) -> HybridActionEvaluation:
        for ev in self.evaluations:
            if ev.action == self.proposed_action:
                return ev
        return self.evaluations[0]


class UncertaintyEstimator:
    """Calculates uncertainty and confidence for ModelFree and Bellman components.

    Anti-leakage guarantee:
    - ModelFree uncertainty uses ONLY training-derived state-action counts and sample variance.
    - Bellman uncertainty uses ONLY inference-time model estimates and context features.
    - No evaluation ground truth or counterfactual rewards are ever accessed.
    """

    def __init__(
        self,
        fitted_mf_policy: FittedQIterationPolicy,
        n_min_samples: int = 5,
        default_mf_confidence: float = 0.1,
    ) -> None:
        self.fitted_policy = fitted_mf_policy
        self.n_min_samples = n_min_samples
        self.default_mf_confidence = default_mf_confidence

    def get_modelfree_confidence(self, state: QState, action: Action) -> float:
        """Estimate ModelFree confidence in [0, 1] based on training visitation and variance.

        Confidence increases with sample size n and decreases with return dispersion:
            conf_mf = (n / (n + n_min)) * (1 / (1 + CV))
        where CV = std / (abs(mean) + 1.0).
        """
        samples = self.fitted_policy._q_table.get((state, action), [])
        n = len(samples)
        if n == 0:
            return self.default_mf_confidence

        mean_val = sum(samples) / n
        if n > 1:
            variance = sum((x - mean_val) ** 2 for x in samples) / (n - 1)
            std_val = math.sqrt(max(0.0, variance))
        else:
            std_val = abs(mean_val) * 0.5

        # Sample size factor saturates to 1 as n grows
        size_factor = float(n) / (float(n) + float(self.n_min_samples))

        # Dispersion factor (normalized between 0 and 1)
        cv = std_val / (abs(mean_val) + 1.0)
        dispersion_factor = 1.0 / (1.0 + min(cv, 10.0))

        confidence = size_factor * dispersion_factor
        return float(max(0.01, min(0.99, confidence)))

    def get_bellman_confidence(
        self,
        context: PaymentContext,
        action: Action,
        bellman_eval: BellmanActionEvaluation,
        probability_model: RecoveryProbabilityModel,
    ) -> float:
        """Estimate Bellman confidence in [0, 1] based on model probability dispersion.

        Confidence is higher for confident (near 0 or near 1) probability estimates,
        and penalizes high uncertainty around 0.5:
            conf_p = 1 - 4 * (p - 0.5)^2   (entropy-like penalty)
            conf_bellman = 1.0 - conf_p * 0.5
        Also incorporates attempt attenuation (later attempts have longer lookahead uncertainty).
        """
        p = float(bellman_eval.probability)
        # Distance from maximal entropy (p=0.5)
        dist_from_half = abs(p - 0.5)  # in [0, 0.5]
        certainty = dist_from_half * 2.0  # in [0, 1]

        # Base confidence from model probability decisiveness
        base_conf = 0.5 + 0.4 * certainty

        # Lookahead discount: attempt 1 has 2 steps lookahead, attempt 3 has 0 steps
        attempt = max(1, context.attempt_count)
        horizon_factor = 1.0 - (0.05 * (attempt - 1))

        conf = base_conf * horizon_factor
        return float(max(0.01, min(0.99, conf)))


class HybridRecoverIQStrategy:
    """RecoverIQ-Hybrid Strategy combining Bellman DP and ModelFree Q-values.

    Formula:
        Q_hybrid(s, a) = w(s, a) * Q_bellman(s, a) + (1 - w(s, a)) * Q_modelfree(s, a)

    Arbitration Regimes:
    - EQUAL_WEIGHT: w = 0.5
    - FIXED_WEIGHT: w = fixed_bellman_weight (default 0.70)
    - UNCERTAINTY_AWARE: w = conf_bellman / (conf_bellman + conf_modelfree)
    """

    name: str = "RecoverIQ-Hybrid"

    def __init__(
        self,
        bellman_strategy: BellmanRecoverIQStrategy,
        modelfree_strategy: ModelFreeRecoverIQStrategy,
        regime: HybridRegime = HybridRegime.UNCERTAINTY_AWARE,
        fixed_bellman_weight: float = 0.70,
        uncertainty_estimator: Optional[UncertaintyEstimator] = None,
    ) -> None:
        self.bellman_strategy = bellman_strategy
        self.modelfree_strategy = modelfree_strategy
        self.regime = regime
        self.fixed_bellman_weight = Decimal(str(fixed_bellman_weight))
        self.uncertainty_estimator = uncertainty_estimator or UncertaintyEstimator(
            fitted_mf_policy=modelfree_strategy.policy
        )
        self.last_decision: Optional[HybridDecision] = None

    def evaluate_hybrid_actions(
        self,
        record: SyntheticPaymentRecord,
        context: PaymentContext,
    ) -> List[HybridActionEvaluation]:
        """Compute hybrid Q-values for all candidate actions."""
        current_attempt = context.attempt_count if context.attempt_count > 0 else 1

        # 1. Obtain Bellman evaluations
        bellman_evals = self.bellman_strategy.evaluate_q_values(
            record=record,
            context=context,
            current_attempt=current_attempt,
            effective_horizon=self.bellman_strategy.planning_horizon,
        )
        bellman_map = {ev.action: ev for ev in bellman_evals}

        # 2. State for ModelFree Q lookup
        state = _make_state(
            failure_category=context.failure_category,
            customer_tier=context.customer_tier,
            attempt_number=current_attempt,
        )

        hybrid_evals: List[HybridActionEvaluation] = []

        for action in Action:
            b_eval = bellman_map[action]
            q_bellman = b_eval.total_q_value

            # ModelFree Q-value (float -> Decimal)
            q_mf_float = self.modelfree_strategy.policy.get_q_value(state, action)
            q_mf = Decimal(str(round(q_mf_float, 4)))

            # Determine arbitration weights
            if self.regime == HybridRegime.EQUAL_WEIGHT:
                w_b = Decimal("0.50")
                w_mf = Decimal("0.50")
                conf_b = Decimal("0.50")
                conf_mf = Decimal("0.50")
            elif self.regime == HybridRegime.FIXED_WEIGHT:
                w_b = self.fixed_bellman_weight
                w_mf = Decimal("1.0") - w_b
                conf_b = self.fixed_bellman_weight
                conf_mf = w_mf
            elif self.regime == HybridRegime.UNCERTAINTY_AWARE:
                c_b_float = self.uncertainty_estimator.get_bellman_confidence(
                    context=context,
                    action=action,
                    bellman_eval=b_eval,
                    probability_model=self.bellman_strategy.probability_model,
                )
                c_mf_float = self.uncertainty_estimator.get_modelfree_confidence(
                    state=state,
                    action=action,
                )
                total_conf = c_b_float + c_mf_float
                w_b_float = c_b_float / total_conf if total_conf > 0 else 0.5
                w_b = Decimal(str(round(w_b_float, 4)))
                w_mf = Decimal("1.0") - w_b
                conf_b = Decimal(str(round(c_b_float, 4)))
                conf_mf = Decimal(str(round(c_mf_float, 4)))
            else:
                w_b = Decimal("0.50")
                w_mf = Decimal("0.50")
                conf_b = Decimal("0.50")
                conf_mf = Decimal("0.50")

            # Compute combined Q_hybrid
            q_hybrid = w_b * q_bellman + w_mf * q_mf

            hybrid_evals.append(
                HybridActionEvaluation(
                    action=action,
                    q_bellman=q_bellman,
                    q_modelfree=q_mf,
                    weight_bellman=w_b,
                    weight_modelfree=w_mf,
                    q_hybrid=q_hybrid,
                    bellman_confidence=conf_b,
                    modelfree_confidence=conf_mf,
                )
            )

        # Sort descending by q_hybrid
        hybrid_evals.sort(key=lambda x: x.q_hybrid, reverse=True)
        return hybrid_evals

    def propose_action(
        self,
        record: SyntheticPaymentRecord,
        context: PaymentContext,
    ) -> Action:
        """Select action maximizing Q_hybrid at the current decision step."""
        evals = self.evaluate_hybrid_actions(record, context)
        best = evals[0]

        current_attempt = context.attempt_count if context.attempt_count > 0 else 1

        # If best hybrid Q is below threshold, stop
        if best.q_hybrid <= self.bellman_strategy.config.min_ev_threshold:
            proposed_action = Action.STOP
            rationale = f"Max Q_hybrid ({best.q_hybrid:.2f}) <= threshold; defaulting to STOP."
        else:
            proposed_action = best.action
            rationale = (
                f"Selected {best.action.value} via {self.regime.value} arbitration: "
                f"Q_hybrid={best.q_hybrid:.2f} (Q_b={best.q_bellman:.2f} * {best.weight_bellman:.2f} + "
                f"Q_mf={best.q_modelfree:.2f} * {best.weight_modelfree:.2f})."
            )

        self.last_decision = HybridDecision(
            payment_id=context.payment_id,
            attempt_count=current_attempt,
            proposed_action=proposed_action,
            regime=self.regime,
            evaluations=evals,
            rationale=rationale,
        )

        return proposed_action
