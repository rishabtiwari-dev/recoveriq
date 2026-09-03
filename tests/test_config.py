"""Tests for the RecoverIQ typed configuration system."""

from decimal import Decimal
from recoveriq.config.settings import (
    ActionCostConfig,
    EconomicConfig,
    PenaltyConfig,
    PolicyConfig,
    RecoverIQConfig,
)
from recoveriq.domain.actions import Action
from recoveriq.domain.models import CustomerTier, FailureCategory


def test_default_config_initialization():
    """Verify default configuration hierarchy."""
    config = RecoverIQConfig.default()
    assert config.policy.max_attempts == 3
    assert config.policy.cooldown_seconds == 900
    assert config.economics.min_ev_threshold == Decimal("0.00")
    assert config.environment == "development"


def test_action_cost_config():
    """Verify configured action costs."""
    cost_cfg = ActionCostConfig()
    assert cost_cfg.get_cost(Action.RETRY_NOW) == Decimal("0.15")
    assert cost_cfg.get_cost(Action.ESCALATE) == Decimal("3.50")
    assert cost_cfg.get_cost(Action.STOP) == Decimal("0.00")


def test_penalty_config_scaling():
    """Verify friction penalty calculation scaled by customer tier."""
    penalty_cfg = PenaltyConfig()
    # NUDGE base penalty is 0.25
    standard_nudge_penalty = penalty_cfg.get_penalty(Action.NUDGE, CustomerTier.STANDARD)
    vip_nudge_penalty = penalty_cfg.get_penalty(Action.NUDGE, CustomerTier.VIP)

    assert standard_nudge_penalty == Decimal("0.25")
    # VIP multiplier is 3.0 -> 0.25 * 3.0 = 0.75
    assert vip_nudge_penalty == Decimal("0.75")


def test_policy_config_hard_declines():
    """Verify hard decline rule configuration."""
    policy_cfg = PolicyConfig()
    assert FailureCategory.HARD_DECLINE in policy_cfg.hard_decline_categories
    assert FailureCategory.INVALID_DETAILS in policy_cfg.hard_decline_categories
    assert FailureCategory.NETWORK_TIMEOUT not in policy_cfg.hard_decline_categories
