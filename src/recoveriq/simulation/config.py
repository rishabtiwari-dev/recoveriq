"""Simulation configuration: distributions, sizes, and seed management."""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Tuple

from recoveriq.domain.models import CustomerTier, FailureCategory, PaymentMethod


@dataclass(frozen=True)
class AmountDistributionConfig:
    """Configuration for synthetic payment amount generation."""

    # (min, max) pairs per customer tier in the payment currency (INR/USD etc.)
    tier_amount_ranges: Dict[CustomerTier, Tuple[Decimal, Decimal]] = field(
        default_factory=lambda: {
            CustomerTier.NEW: (Decimal("99"), Decimal("999")),
            CustomerTier.STANDARD: (Decimal("200"), Decimal("4999")),
            CustomerTier.PREMIUM: (Decimal("1000"), Decimal("24999")),
            CustomerTier.VIP: (Decimal("5000"), Decimal("99999")),
        }
    )


@dataclass(frozen=True)
class FailureCategoryDistributionConfig:
    """Probability weights for sampling failure categories."""

    # Weights do not need to sum to 1.0 — they are normalized internally.
    weights: Dict[FailureCategory, float] = field(
        default_factory=lambda: {
            FailureCategory.INSUFFICIENT_FUNDS: 0.28,
            FailureCategory.NETWORK_TIMEOUT: 0.22,
            FailureCategory.CARD_EXPIRED: 0.10,
            FailureCategory.AUTHENTICATION_FAILED: 0.15,
            FailureCategory.AUTHENTICATION_REJECTED: 0.06,
            FailureCategory.INVALID_DETAILS: 0.05,
            FailureCategory.HARD_DECLINE: 0.07,
            FailureCategory.VELOCITY_EXCEEDED: 0.05,
            FailureCategory.UNKNOWN: 0.02,
        }
    )


@dataclass(frozen=True)
class CustomerTierDistributionConfig:
    """Probability weights for sampling customer tiers."""

    weights: Dict[CustomerTier, float] = field(
        default_factory=lambda: {
            CustomerTier.NEW: 0.25,
            CustomerTier.STANDARD: 0.50,
            CustomerTier.PREMIUM: 0.15,
            CustomerTier.VIP: 0.10,
        }
    )


@dataclass(frozen=True)
class PaymentMethodDistributionConfig:
    """Probability weights for sampling payment methods."""

    weights: Dict[PaymentMethod, float] = field(
        default_factory=lambda: {
            PaymentMethod.UPI: 0.35,
            PaymentMethod.CREDIT_CARD: 0.25,
            PaymentMethod.DEBIT_CARD: 0.20,
            PaymentMethod.NET_BANKING: 0.12,
            PaymentMethod.WALLET: 0.08,
        }
    )


@dataclass(frozen=True)
class SimulationConfig:
    """Root configuration for synthetic payment simulation."""

    # Number of synthetic failed payments to generate
    n_payments: int = 2000
    # Default random seed
    default_seed: int = 42
    # Train fraction for payment-level split
    train_fraction: float = 0.75
    # Currency for all synthetic amounts
    currency: str = "INR"
    # Number of customers (payments randomly assigned to customers)
    n_customers: int = 500
    # Whether to hold leakage-sensitive ground-truth fields out of observable features
    hide_ground_truth_from_features: bool = True

    # Sub-configs
    amount_distribution: AmountDistributionConfig = field(
        default_factory=AmountDistributionConfig
    )
    failure_distribution: FailureCategoryDistributionConfig = field(
        default_factory=FailureCategoryDistributionConfig
    )
    tier_distribution: CustomerTierDistributionConfig = field(
        default_factory=CustomerTierDistributionConfig
    )
    method_distribution: PaymentMethodDistributionConfig = field(
        default_factory=PaymentMethodDistributionConfig
    )
