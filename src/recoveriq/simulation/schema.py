"""Observable payment record schema and hidden ground-truth record.

DESIGN INVARIANT:
- SyntheticPaymentRecord contains ONLY the fields visible at decision time.
- GroundTruthRecord contains the HIDDEN latent recoverability variables that the
  simulator uses to resolve action outcomes. These must NEVER be exposed as
  model input features.

These two objects are deliberately separate dataclasses so it is impossible
to accidentally pass ground-truth data into the model or policy.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, unique
from typing import Dict

from recoveriq.domain.actions import Action
from recoveriq.domain.models import CustomerTier, FailureCategory, FailureSeverity, PaymentMethod


@unique
class RecoverabilityProfile(str, Enum):
    """Latent recoverability class assigned by the world model.

    This field is HIDDEN from all model inputs. It represents the simulator's
    true probability tier for this payment, generated independently.
    """

    VERY_LOW = "VERY_LOW"      # ~5-10% base recovery probability
    LOW = "LOW"                # ~15-30% base recovery probability
    MEDIUM = "MEDIUM"          # ~35-55% base recovery probability
    HIGH = "HIGH"              # ~60-80% base recovery probability
    VERY_HIGH = "VERY_HIGH"    # ~80-95% base recovery probability


# Base recovery probabilities per profile, per action.
# These are the world model's true parameters — entirely independent of
# RecoverIQ's policy engine, heuristics, or probability estimator.
#
# Format: {RecoverabilityProfile: {Action: base_probability}}
GROUND_TRUTH_RECOVERY_TABLE: Dict[RecoverabilityProfile, Dict[Action, float]] = {
    RecoverabilityProfile.VERY_LOW: {
        Action.RETRY_NOW: 0.04,
        Action.RETRY_LATER: 0.07,
        Action.SEND_LINK: 0.05,
        Action.NUDGE: 0.03,
        Action.ESCALATE: 0.09,
        Action.STOP: 0.00,
    },
    RecoverabilityProfile.LOW: {
        Action.RETRY_NOW: 0.12,
        Action.RETRY_LATER: 0.20,
        Action.SEND_LINK: 0.18,
        Action.NUDGE: 0.10,
        Action.ESCALATE: 0.25,
        Action.STOP: 0.00,
    },
    RecoverabilityProfile.MEDIUM: {
        Action.RETRY_NOW: 0.30,
        Action.RETRY_LATER: 0.45,
        Action.SEND_LINK: 0.42,
        Action.NUDGE: 0.28,
        Action.ESCALATE: 0.52,
        Action.STOP: 0.00,
    },
    RecoverabilityProfile.HIGH: {
        Action.RETRY_NOW: 0.55,
        Action.RETRY_LATER: 0.68,
        Action.SEND_LINK: 0.65,
        Action.NUDGE: 0.50,
        Action.ESCALATE: 0.75,
        Action.STOP: 0.00,
    },
    RecoverabilityProfile.VERY_HIGH: {
        Action.RETRY_NOW: 0.78,
        Action.RETRY_LATER: 0.85,
        Action.SEND_LINK: 0.82,
        Action.NUDGE: 0.72,
        Action.ESCALATE: 0.90,
        Action.STOP: 0.00,
    },
}


@dataclass(frozen=True)
class SyntheticPaymentRecord:
    """Observable payment record — these are the ONLY fields visible at decision time.

    No hidden recoverability fields are present here. This is what gets passed
    to the context extractor, AI layer, model, and economic engine.
    """

    payment_id: str
    customer_id: str
    amount: Decimal
    currency: str

    # Observable failure signals
    failure_category: FailureCategory
    failure_severity: FailureSeverity
    customer_tier: CustomerTier
    payment_method: PaymentMethod
    raw_error_code: str
    raw_error_message: str

    # Temporal features (observable)
    failure_timestamp: datetime
    attempt_count: int


@dataclass(frozen=True)
class GroundTruthRecord:
    """HIDDEN ground-truth record used ONLY by the simulation environment.

    This object must NEVER be passed to any RecoverIQ module
    (context extractor, AI layer, probability model, economic engine, or policy gate).
    It exists solely to allow the SimulationEnvironment to resolve action outcomes.
    """

    payment_id: str
    latent_recoverability_profile: RecoverabilityProfile
    # Per-action base recovery probability from the world model (before noise)
    action_base_probabilities: Dict[Action, float] = field(default_factory=dict)
