"""Typed configuration objects for RecoverIQ policy, economics, costs, and limits."""

import os
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, FrozenSet

from recoveriq.domain.actions import Action
from recoveriq.domain.models import CustomerTier, FailureCategory


@dataclass(frozen=True)
class ActionCostConfig:
    """Direct operational cost associated with executing each recovery action."""

    costs: Dict[Action, Decimal] = field(
        default_factory=lambda: {
            Action.RETRY_NOW: Decimal("0.15"),    # Gateway API retry fee
            Action.RETRY_LATER: Decimal("0.15"),  # Gateway API retry fee
            Action.SEND_LINK: Decimal("0.35"),    # SMS + Payment Link generation fee
            Action.NUDGE: Decimal("0.10"),        # In-app / Push notification cost
            Action.ESCALATE: Decimal("3.50"),     # Human support agent operations cost
            Action.STOP: Decimal("0.00"),         # No operational cost
        }
    )

    def get_cost(self, action: Action) -> Decimal:
        """Retrieve the configured direct cost for an action."""
        return self.costs.get(action, Decimal("0.00"))


@dataclass(frozen=True)
class PenaltyConfig:
    """Friction and customer dissatisfaction penalty configurations."""

    tier_multipliers: Dict[CustomerTier, Decimal] = field(
        default_factory=lambda: {
            CustomerTier.STANDARD: Decimal("1.0"),
            CustomerTier.PREMIUM: Decimal("1.5"),
            CustomerTier.VIP: Decimal("3.0"),      # High VIP friction sensitivity
            CustomerTier.NEW: Decimal("1.2"),
        }
    )

    base_action_penalties: Dict[Action, Decimal] = field(
        default_factory=lambda: {
            Action.RETRY_NOW: Decimal("0.05"),
            Action.RETRY_LATER: Decimal("0.02"),
            Action.SEND_LINK: Decimal("0.10"),
            Action.NUDGE: Decimal("0.25"),         # Repeated nudges cause annoyance
            Action.ESCALATE: Decimal("0.00"),      # VIP escalation is positive service
            Action.STOP: Decimal("0.00"),
        }
    )

    def get_penalty(self, action: Action, tier: CustomerTier) -> Decimal:
        """Calculate friction penalty scaled by customer tier."""
        base = self.base_action_penalties.get(action, Decimal("0.00"))
        multiplier = self.tier_multipliers.get(tier, Decimal("1.0"))
        return base * multiplier


@dataclass(frozen=True)
class PolicyConfig:
    """Deterministic policy rules, thresholds, and safety bounds."""

    max_attempts: int = 3
    cooldown_seconds: int = 900  # 15 minutes between non-immediate actions
    disallow_retries_on_hard_declines: bool = True
    hard_decline_categories: FrozenSet[FailureCategory] = field(
        default_factory=lambda: frozenset(
            {
                FailureCategory.HARD_DECLINE,
                FailureCategory.INVALID_DETAILS,
            }
        )
    )
    fallback_on_budget_exhausted: Action = Action.STOP
    fallback_on_hard_decline: Action = Action.STOP
    vip_escalation_enabled: bool = True


@dataclass(frozen=True)
class EconomicConfig:
    """Economic optimization parameters."""

    min_ev_threshold: Decimal = Decimal("0.00")
    cost_config: ActionCostConfig = field(default_factory=ActionCostConfig)
    penalty_config: PenaltyConfig = field(default_factory=PenaltyConfig)


@dataclass(frozen=True)
class LLMConfig:
    """Strongly typed LLM configuration for contextual interpretation."""

    provider: str = "gemini"
    model_name: str = field(
        default_factory=lambda: os.environ.get("LLM_MODEL", "gemini-3.8-flash")
    )
    timeout_seconds: float = 5.0
    temperature: float = 0.0
    max_retries: int = 1
    schema_version: str = "v1"


@dataclass(frozen=True)
class RecoverIQConfig:
    """Root configuration object for RecoverIQ engine."""

    policy: PolicyConfig = field(default_factory=PolicyConfig)
    economics: EconomicConfig = field(default_factory=EconomicConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    environment: str = "development"

    @classmethod
    def default(cls) -> "RecoverIQConfig":
        """Factory for default research configuration."""
        return cls()
