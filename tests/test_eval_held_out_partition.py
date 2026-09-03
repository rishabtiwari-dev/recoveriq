"""Sprint 5 Tests — Held-out test partition guarantees and training separation."""

import pytest

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


def test_evaluated_payments_are_strictly_from_test_partition(sim_partition):
    """Every payment ID evaluated must belong to the test partition with zero training leakage."""
    runner = EvaluationRunner()
    strat = FixedRetryStrategy()

    metrics = runner.evaluate_strategy_on_partition(
        strategy=strat,
        test_observable=sim_partition.test_observable,
        test_ground_truth=sim_partition.test_ground_truth,
        seed=42,
    )

    eval_pids = {r.payment_id for r in metrics.records}
    test_pids = set(sim_partition.test_payment_ids())
    train_pids = set(sim_partition.train_payment_ids())

    assert eval_pids == test_pids
    assert len(eval_pids & train_pids) == 0


def test_empty_test_observable_raises_error(sim_partition):
    """Empty test set must fail fast with ValueError."""
    runner = EvaluationRunner()
    strat = FixedRetryStrategy()

    with pytest.raises(ValueError) as exc_info:
        runner.evaluate_strategy_on_partition(
            strategy=strat,
            test_observable=[],
            test_ground_truth=[],
            seed=42,
        )

    assert "cannot be empty" in str(exc_info.value)
