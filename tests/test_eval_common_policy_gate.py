"""Sprint 5 Tests — Common Policy Gate enforcement on all strategies."""

import pytest

from recoveriq.domain.actions import Action
from recoveriq.evaluation.runner import EvaluationRunner
from recoveriq.evaluation.strategies import FixedRetryStrategy, RuleBasedStrategy
from recoveriq.simulation.config import SimulationConfig
from recoveriq.simulation.generator import SyntheticPaymentGenerator
from recoveriq.simulation.partitioner import partition_dataset


@pytest.fixture
def sim_partition():
    cfg = SimulationConfig(n_payments=100, n_customers=20, default_seed=42)
    gen = SyntheticPaymentGenerator(cfg)
    ds = gen.generate(seed=42)
    return partition_dataset(ds, train_fraction=0.75)


def test_fixed_retry_blocked_on_hard_declines(sim_partition):
    """Fixed-Retry proposes RETRY_NOW on hard decline, but the Policy Gate must clamp it to STOP."""
    runner = EvaluationRunner()
    strat = FixedRetryStrategy()

    metrics = runner.evaluate_strategy_on_partition(
        strategy=strat,
        test_observable=sim_partition.test_observable,
        test_ground_truth=sim_partition.test_ground_truth,
        seed=42,
    )

    hard_decline_pids = {
        r.payment_id for r in sim_partition.test_observable if r.failure_category.is_hard_decline
    }
    assert len(hard_decline_pids) > 0

    for rec in metrics.records:
        if rec.payment_id in hard_decline_pids:
            assert rec.proposed_action == Action.RETRY_NOW
            assert rec.authorized_action == Action.STOP
            assert rec.is_authorized is False
            assert "cannot be retried" in rec.rejection_reason


def test_all_strategies_have_zero_policy_violations(sim_partition):
    """Policy violation rate must be strictly 0.00% across all strategies."""
    runner = EvaluationRunner()
    fixed_retry = FixedRetryStrategy()
    rule_based = RuleBasedStrategy()

    res = runner.evaluate_all_strategies(
        strategies=[fixed_retry, rule_based],
        test_observable=sim_partition.test_observable,
        test_ground_truth=sim_partition.test_ground_truth,
        seed=42,
    )

    for name, metrics in res.items():
        assert metrics.policy_violation_rate == 0.0
        # Check every record has a valid authorized action
        for r in metrics.records:
            assert isinstance(r.authorized_action, Action)
