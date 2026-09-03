"""Synthetic payment failure record generator.

Generates reproducible synthetic failed-payment datasets using configurable
distributions over failure categories, customer tiers, amounts, and methods.

DESIGN NOTE:
The generator produces two parallel lists that are kept strictly separated:
  - observable_records: List[SyntheticPaymentRecord]  — visible to RecoverIQ
  - ground_truth_records: List[GroundTruthRecord]      — hidden world-model data

The caller (simulation environment / evaluation harness) is responsible for
keeping these two lists aligned by index / payment_id and NEVER passing
ground_truth_records into any RecoverIQ module.
"""

import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from recoveriq.domain.models import CustomerTier, FailureCategory, FailureSeverity, PaymentMethod
from recoveriq.simulation.config import SimulationConfig
from recoveriq.simulation.ground_truth import assign_ground_truth
from recoveriq.simulation.schema import GroundTruthRecord, SyntheticPaymentRecord


# --- Raw error code / message templates per failure category --------------------

_ERROR_TEMPLATES: Dict[FailureCategory, List[Tuple[str, str]]] = {
    FailureCategory.INSUFFICIENT_FUNDS: [
        ("INSUFFICIENT_FUNDS", "Transaction declined: insufficient balance"),
        ("LOW_BALANCE", "Account balance too low for this transaction"),
        ("NSF_01", "Not sufficient funds — issuer declined"),
    ],
    FailureCategory.NETWORK_TIMEOUT: [
        ("504_GATEWAY_TIMEOUT", "Gateway timeout while contacting issuer switch"),
        ("NETWORK_ERROR_502", "502 Bad Gateway — upstream payment processor"),
        ("TIMEOUT_30S", "Request timed out after 30 seconds"),
        ("CONNECTION_RESET", "Connection reset by payment gateway"),
    ],
    FailureCategory.CARD_EXPIRED: [
        ("CARD_EXPIRED", "Card expiry date has passed"),
        ("EXPIRED_CARD_01", "Issuer declined: card expired"),
    ],
    FailureCategory.AUTHENTICATION_FAILED: [
        ("OTP_FAILED", "OTP verification failed — maximum attempts exceeded"),
        ("3DS_FAILED", "3D Secure authentication challenge not completed"),
        ("AUTH_FAILED_002", "Authentication step dropped by customer"),
    ],
    FailureCategory.AUTHENTICATION_REJECTED: [
        ("AUTH_REJECTED", "Issuer rejected authentication attempt"),
        ("3DS_REJECTED", "3D Secure rejected by card scheme"),
    ],
    FailureCategory.INVALID_DETAILS: [
        ("INVALID_CVV", "Invalid CVV/CVC provided"),
        ("INVALID_CARD_NUMBER", "Card number failed Luhn check"),
        ("INVALID_EXPIRY", "Mismatched expiry date provided"),
    ],
    FailureCategory.HARD_DECLINE: [
        ("DO_NOT_HONOR", "Issuer: do not honor this transaction"),
        ("STOLEN_CARD", "Card reported stolen — hard decline"),
        ("RESTRICTED_CARD", "Card restricted by issuer"),
        ("FRAUD_SUSPECTED", "Transaction flagged for suspected fraud"),
    ],
    FailureCategory.VELOCITY_EXCEEDED: [
        ("VELOCITY_LIMIT_EXCEEDED", "Daily transaction limit reached"),
        ("LIMIT_EXCEEDED_04", "Velocity limit exceeded on this card"),
    ],
    FailureCategory.UNKNOWN: [
        ("UNKNOWN_ERROR", "Unknown payment error — please retry"),
        ("GENERIC_DECLINE", "Generic decline — contact your bank"),
    ],
}

# Severity mapping per failure category (deterministic, not stochastic)
_CATEGORY_SEVERITY: Dict[FailureCategory, FailureSeverity] = {
    FailureCategory.INSUFFICIENT_FUNDS: FailureSeverity.RECOVERABLE,
    FailureCategory.NETWORK_TIMEOUT: FailureSeverity.TRANSIENT,
    FailureCategory.CARD_EXPIRED: FailureSeverity.STRUCTURAL,
    FailureCategory.AUTHENTICATION_FAILED: FailureSeverity.RECOVERABLE,
    FailureCategory.AUTHENTICATION_REJECTED: FailureSeverity.STRUCTURAL,
    FailureCategory.INVALID_DETAILS: FailureSeverity.FATAL,
    FailureCategory.HARD_DECLINE: FailureSeverity.FATAL,
    FailureCategory.VELOCITY_EXCEEDED: FailureSeverity.STRUCTURAL,
    FailureCategory.UNKNOWN: FailureSeverity.RECOVERABLE,
}


def _sample_from_weights(rng: random.Random, weights: Dict) -> object:
    """Sample one key from a dict of {value: weight} pairs."""
    keys = list(weights.keys())
    wts = [weights[k] for k in keys]
    return rng.choices(keys, weights=wts, k=1)[0]


def _sample_amount(
    rng: random.Random,
    tier: CustomerTier,
    config: SimulationConfig,
) -> Decimal:
    """Generate a random Decimal payment amount for the given customer tier."""
    lo, hi = config.amount_distribution.tier_amount_ranges[tier]
    raw = rng.uniform(float(lo), float(hi))
    # Round to 2 decimal places
    return Decimal(str(round(raw, 2)))


def _sample_failure_timestamp(rng: random.Random, base: datetime) -> datetime:
    """Generate a plausible failure timestamp within ±30 days of base."""
    offset_seconds = rng.randint(-30 * 86400, 0)
    return base + timedelta(seconds=offset_seconds)


@dataclass
class SyntheticDataset:
    """Container for a generated synthetic dataset.

    Attributes:
        observable_records: Payment features visible at decision time.
        ground_truth_records: Hidden world-model data — NEVER pass to RecoverIQ.
        seed: Random seed used for generation.
    """

    observable_records: List[SyntheticPaymentRecord] = field(default_factory=list)
    ground_truth_records: List[GroundTruthRecord] = field(default_factory=list)
    seed: int = 42

    def __post_init__(self) -> None:
        if len(self.observable_records) != len(self.ground_truth_records):
            raise ValueError(
                "observable_records and ground_truth_records must have the same length."
            )

    def __len__(self) -> int:
        return len(self.observable_records)

    def get_ground_truth(self, payment_id: str) -> Optional[GroundTruthRecord]:
        """Look up hidden ground truth by payment_id (O(n) — for small datasets)."""
        for gt in self.ground_truth_records:
            if gt.payment_id == payment_id:
                return gt
        return None


class SyntheticPaymentGenerator:
    """Generates reproducible synthetic failed-payment datasets.

    Usage:
        generator = SyntheticPaymentGenerator(SimulationConfig(n_payments=1000))
        dataset = generator.generate(seed=42)
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()

    def generate(self, seed: Optional[int] = None) -> SyntheticDataset:
        """Generate a fully reproducible synthetic dataset.

        Args:
            seed: Random seed. Overrides config.default_seed if provided.

        Returns:
            SyntheticDataset with observable records and hidden ground truth.
        """
        effective_seed = seed if seed is not None else self.config.default_seed
        rng = random.Random(effective_seed)

        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        # Pre-generate customer IDs so payments can be assigned to customers.
        customer_ids = [
            f"cust_{i:05d}" for i in range(self.config.n_customers)
        ]

        observable_records: List[SyntheticPaymentRecord] = []
        ground_truth_records: List[GroundTruthRecord] = []

        for i in range(self.config.n_payments):
            # --- Sample observable features ---
            payment_id = f"pay_{effective_seed}_{i:06d}"
            customer_id = rng.choice(customer_ids)

            tier = _sample_from_weights(rng, self.config.tier_distribution.weights)
            payment_method = _sample_from_weights(rng, self.config.method_distribution.weights)
            failure_category = _sample_from_weights(rng, self.config.failure_distribution.weights)
            failure_severity = _CATEGORY_SEVERITY[failure_category]
            amount = _sample_amount(rng, tier, self.config)
            timestamp = _sample_failure_timestamp(rng, base_time)
            attempt_count = rng.randint(1, 3)

            # Sample error code/message template
            templates = _ERROR_TEMPLATES.get(failure_category, [("GENERIC", "Generic error")])
            raw_error_code, raw_error_message = rng.choice(templates)

            observable = SyntheticPaymentRecord(
                payment_id=payment_id,
                customer_id=customer_id,
                amount=amount,
                currency=self.config.currency,
                failure_category=failure_category,
                failure_severity=failure_severity,
                customer_tier=tier,
                payment_method=payment_method,
                raw_error_code=raw_error_code,
                raw_error_message=raw_error_message,
                failure_timestamp=timestamp,
                attempt_count=attempt_count,
            )

            # --- Assign hidden ground truth (world model) ---
            gt = assign_ground_truth(
                payment_id=payment_id,
                failure_category=failure_category,
                customer_tier=tier,
                rng=rng,
            )

            observable_records.append(observable)
            ground_truth_records.append(gt)

        return SyntheticDataset(
            observable_records=observable_records,
            ground_truth_records=ground_truth_records,
            seed=effective_seed,
        )
