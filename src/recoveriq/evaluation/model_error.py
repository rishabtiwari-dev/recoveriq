"""Sprint 14 — Model Error Perturbation & Distribution Shift Utilities.

Provides two orthogonal experiment axes:

1. MODEL MISSPECIFICATION (M0–M3):
   Wraps a RecoveryProbabilityModel with deterministic structured perturbations
   that simulate degraded probability estimates without modifying the trained model.

   Perturbation conditions:
   - M0 (CORRECT):    Identity — returns clean estimates unchanged.
   - M1 (MILD):       Calibration squeeze toward 0.5 by 10 percentage points.
   - M2 (MODERATE):   Calibration squeeze toward 0.5 by 20 percentage points.
   - M3 (SEVERE):     Calibration squeeze (30pp) PLUS systematic ESCALATE overestimation
                      (+0.20 additive bias, clamped to [0,1]).

   Mathematical specification:
   M1: p' = p + 0.10 * (0.5 - p)  = 0.90*p + 0.05
   M2: p' = p + 0.20 * (0.5 - p)  = 0.80*p + 0.10
   M3 (non-ESCALATE): p' = p + 0.30 * (0.5 - p) = 0.70*p + 0.15
   M3 (ESCALATE):     p' = clip(p_squeeze + 0.20, 0.0, 1.0)

   All perturbed probabilities are clamped to [0.0, 1.0].
   Perturbations are deterministic (no RNG).

2. DISTRIBUTION SHIFT (D0–D3):
   Returns a synthetically modified copy of observable payment records
   to simulate covariate shift between training and evaluation distributions.

   Shift conditions:
   - D0 (IN-DISTRIBUTION):   No transformation — identity.
   - D1 (VALUE SHIFT):       Payment amounts doubled (higher-value skew).
   - D2 (PROFILE SHIFT):     CustomerTier shifted up by one level
                              (NEW → STANDARD, STANDARD → PREMIUM, PREMIUM → VIP, VIP → VIP).
   - D3 (COMBINED SHIFT):    D1 + D2 simultaneously applied.

   All shifts are deterministic, documentable, and do not modify ground-truth records.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional

from recoveriq.domain.actions import Action
from recoveriq.domain.models import CustomerTier, PaymentContext
from recoveriq.model.probability import ProbabilityEstimate, RecoveryProbabilityModel
from recoveriq.simulation.schema import SyntheticPaymentRecord


# ---------------------------------------------------------------------------
# Model Error Conditions
# ---------------------------------------------------------------------------

class ModelErrorCondition(str, Enum):
    """Controlled model misspecification conditions for Sprint 14."""

    M0_CORRECT = "M0_CORRECT"
    M1_MILD = "M1_MILD"
    M2_MODERATE = "M2_MODERATE"
    M3_SEVERE = "M3_SEVERE"

    @property
    def squeeze_alpha(self) -> float:
        """Fraction of distance toward 0.5 to squeeze probabilities."""
        return {
            ModelErrorCondition.M0_CORRECT: 0.00,
            ModelErrorCondition.M1_MILD: 0.10,
            ModelErrorCondition.M2_MODERATE: 0.20,
            ModelErrorCondition.M3_SEVERE: 0.30,
        }[self]

    @property
    def escalate_bias(self) -> float:
        """Additive bias applied specifically to ESCALATE probability (M3 only)."""
        return 0.20 if self == ModelErrorCondition.M3_SEVERE else 0.00

    @property
    def label(self) -> str:
        return {
            ModelErrorCondition.M0_CORRECT: "M0 (Correct)",
            ModelErrorCondition.M1_MILD: "M1 (Mild ±10pp)",
            ModelErrorCondition.M2_MODERATE: "M2 (Moderate ±20pp)",
            ModelErrorCondition.M3_SEVERE: "M3 (Severe ±30pp + ESCALATE bias)",
        }[self]


ALL_MODEL_ERROR_CONDITIONS: List[ModelErrorCondition] = [
    ModelErrorCondition.M0_CORRECT,
    ModelErrorCondition.M1_MILD,
    ModelErrorCondition.M2_MODERATE,
    ModelErrorCondition.M3_SEVERE,
]


def _perturb_probability(
    p: Decimal,
    action: Action,
    condition: ModelErrorCondition,
) -> Decimal:
    """Apply deterministic perturbation to a single probability value.

    Mathematical specification:
      M0: p' = p
      M1: p' = p + 0.10*(0.5 - p) = 0.90*p + 0.05
      M2: p' = p + 0.20*(0.5 - p) = 0.80*p + 0.10
      M3 (non-ESCALATE): p' = p + 0.30*(0.5 - p) = 0.70*p + 0.15
      M3 (ESCALATE):     p' = clip(squeezed + 0.20, 0.0, 1.0)
    """
    alpha = Decimal(str(condition.squeeze_alpha))
    half = Decimal("0.5")

    # Calibration squeeze toward 0.5
    p_squeezed = p + alpha * (half - p)

    # Additional escalate bias for M3
    if action == Action.ESCALATE:
        p_biased = p_squeezed + Decimal(str(condition.escalate_bias))
    else:
        p_biased = p_squeezed

    # Clamp to [0, 1]
    p_final = max(Decimal("0.0"), min(Decimal("1.0"), p_biased))
    return p_final


class PerturbedProbabilityModel:
    """Wraps a RecoveryProbabilityModel and applies deterministic probability perturbations.

    This is a pure research wrapper — it does not modify the underlying trained model.
    The perturbation is applied only to the output probabilities at inference time.

    Anti-leakage guarantee:
        PerturbedProbabilityModel never accesses ground_truth_records.
        It only transforms the output of the base model's estimate_probabilities().

    Usage:
        perturbed_model = PerturbedProbabilityModel(base_model, ModelErrorCondition.M2_MODERATE)
        bellman_strategy = BellmanRecoverIQStrategy(probability_model=perturbed_model, ...)
    """

    def __init__(
        self,
        base_model: RecoveryProbabilityModel,
        condition: ModelErrorCondition,
    ) -> None:
        self._base_model = base_model
        self.condition = condition

    def estimate_probabilities(
        self,
        context: PaymentContext,
    ) -> Dict[Action, ProbabilityEstimate]:
        """Return perturbed probability estimates for all candidate actions."""
        base_estimates = self._base_model.estimate_probabilities(context)
        perturbed: Dict[Action, ProbabilityEstimate] = {}

        for action, est in base_estimates.items():
            p_perturbed = _perturb_probability(est.probability, action, self.condition)
            perturbed[action] = ProbabilityEstimate(
                action=action,
                probability=p_perturbed,
                model_version=f"{est.model_version}+{self.condition.value}",
            )

        return perturbed


# ---------------------------------------------------------------------------
# Distribution Shift Conditions
# ---------------------------------------------------------------------------

class DistributionShiftCondition(str, Enum):
    """Controlled evaluation distribution shift conditions for Sprint 14."""

    D0_IN_DISTRIBUTION = "D0_IN_DISTRIBUTION"
    D1_VALUE_SHIFT = "D1_VALUE_SHIFT"
    D2_PROFILE_SHIFT = "D2_PROFILE_SHIFT"
    D3_COMBINED_SHIFT = "D3_COMBINED_SHIFT"

    @property
    def label(self) -> str:
        return {
            DistributionShiftCondition.D0_IN_DISTRIBUTION: "D0 (In-Distribution)",
            DistributionShiftCondition.D1_VALUE_SHIFT: "D1 (Value Shift 2x)",
            DistributionShiftCondition.D2_PROFILE_SHIFT: "D2 (Profile Shift +1 Tier)",
            DistributionShiftCondition.D3_COMBINED_SHIFT: "D3 (Combined D1+D2)",
        }[self]


ALL_DISTRIBUTION_SHIFT_CONDITIONS: List[DistributionShiftCondition] = [
    DistributionShiftCondition.D0_IN_DISTRIBUTION,
    DistributionShiftCondition.D1_VALUE_SHIFT,
    DistributionShiftCondition.D2_PROFILE_SHIFT,
    DistributionShiftCondition.D3_COMBINED_SHIFT,
]

# Tier progression for D2/D3 profile shift
_TIER_UPGRADE: Dict[CustomerTier, CustomerTier] = {
    CustomerTier.NEW: CustomerTier.STANDARD,
    CustomerTier.STANDARD: CustomerTier.PREMIUM,
    CustomerTier.PREMIUM: CustomerTier.VIP,
    CustomerTier.VIP: CustomerTier.VIP,  # VIP stays VIP (ceiling)
}


def _shift_record(
    record: SyntheticPaymentRecord,
    shift: DistributionShiftCondition,
) -> SyntheticPaymentRecord:
    """Apply a deterministic distribution shift to a single observable payment record.

    Does NOT modify ground_truth_records. Uses dataclass field replacement.

    D0: identity (no change)
    D1: amount *= 2 (higher-value skew)
    D2: CustomerTier bumped up one level
    D3: D1 + D2
    """
    import dataclasses

    if shift == DistributionShiftCondition.D0_IN_DISTRIBUTION:
        return record

    # Compute shifted fields
    new_amount = record.amount
    new_tier = record.customer_tier

    if shift in (
        DistributionShiftCondition.D1_VALUE_SHIFT,
        DistributionShiftCondition.D3_COMBINED_SHIFT,
    ):
        new_amount = record.amount * Decimal("2")

    if shift in (
        DistributionShiftCondition.D2_PROFILE_SHIFT,
        DistributionShiftCondition.D3_COMBINED_SHIFT,
    ):
        new_tier = _TIER_UPGRADE[record.customer_tier]

    return dataclasses.replace(record, amount=new_amount, customer_tier=new_tier)


def apply_distribution_shift(
    records: List[SyntheticPaymentRecord],
    shift: DistributionShiftCondition,
) -> List[SyntheticPaymentRecord]:
    """Return a new list of observable records under the specified distribution shift.

    Guarantees:
    - The original records list is not mutated.
    - Ground truth records are NOT modified (caller must keep them aligned separately).
    - The transformation is deterministic and reproducible.
    - Record order is preserved.

    Args:
        records: Original observable payment records.
        shift: Distribution shift condition to apply.

    Returns:
        New list of (possibly modified) SyntheticPaymentRecord instances.
    """
    return [_shift_record(r, shift) for r in records]


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def get_perturbation_description(condition: ModelErrorCondition) -> str:
    """Return a human-readable mathematical description of the perturbation."""
    descriptions = {
        ModelErrorCondition.M0_CORRECT: "p' = p  (no perturbation)",
        ModelErrorCondition.M1_MILD: "p' = 0.90·p + 0.05  (10pp calibration squeeze toward 0.5)",
        ModelErrorCondition.M2_MODERATE: "p' = 0.80·p + 0.10  (20pp calibration squeeze toward 0.5)",
        ModelErrorCondition.M3_SEVERE: (
            "p' = 0.70·p + 0.15  (30pp squeeze) for all actions except ESCALATE; "
            "for ESCALATE: p' = clip(0.70·p + 0.15 + 0.20, 0, 1)  (+0.20 additive bias)"
        ),
    }
    return descriptions[condition]
