"""Sprint 5 Tests — Common Random Numbers (CRN) invariance and variance reduction."""

import pytest

from recoveriq.domain.actions import Action
from recoveriq.evaluation.runner import EvaluationRunner
from recoveriq.evaluation.strategies import FixedRetryStrategy, RuleBasedStrategy
from recoveriq.simulation.config import SimulationConfig
from recoveriq.simulation.environment import SimulationEnvironment
from recoveriq.simulation.generator import SyntheticPaymentGenerator
from recoveriq.simulation.partitioner import partition_dataset


@pytest.fixture
def sim_partition():
    cfg = SimulationConfig(n_payments=80, n_customers=15, default_seed=42)
    gen = SyntheticPaymentGenerator(cfg)
    ds = gen.generate(seed=42)
    return partition_dataset(ds, train_fraction=0.75)


def test_crn_same_action_yields_identical_stochastic_outcomes(sim_partition):
    """Under CRN with the same seed, whenever two strategies authorize the same action, outcome is 100% identical."""
    runner = EvaluationRunner()

    strat1 = FixedRetryStrategy()
    # Strategy 2 is another instance of Fixed-Retry to check pure CRN determinism
    strat2 = FixedRetryStrategy()

    res1 = runner.evaluate_strategy_on_partition(
        strategy=strat1,
        test_observable=sim_partition.test_observable,
        test_ground_truth=sim_partition.test_ground_truth,
        seed=42,
    )
    res2 = runner.evaluate_strategy_on_partition(
        strategy=strat2,
        test_observable=sim_partition.test_observable,
        test_ground_truth=sim_partition.test_ground_truth,
        seed=42,
    )

    for r1, r2 in zip(res1.records, res2.records):
        assert r1.payment_id == r2.payment_id
        assert r1.authorized_action == r2.authorized_action
        assert r1.recovered == r2.recovered, (
            f"CRN violation for payment {r1.payment_id}: outcomes differ!"
        )


def test_crn_environments_are_independent_instances(sim_partition):
    """Evaluating strategy A must NOT mutate or advance strategy B's simulation environment."""
    env1 = SimulationEnvironment(sim_partition.test_ground_truth, seed=123)
    env2 = SimulationEnvironment(sim_partition.test_ground_truth, seed=123)

    # env1 is used for some queries
    pid = sim_partition.test_observable[0].payment_id
    env1.apply_action(pid, Action.RETRY_NOW)

    # env2 is completely independent and untouched
    # Its first call must receive the identical outcome that env1 received on its first call
    o1_fresh = SimulationEnvironment(sim_partition.test_ground_truth, seed=123).apply_action(pid, Action.RETRY_NOW)
    o2 = env2.apply_action(pid, Action.RETRY_NOW)

    assert o1_fresh.recovered == o2.recovered


def test_crn_aligns_pairwise_action_matches_across_distinct_strategies(sim_partition):
    """When Fixed-Retry and Rule-Based both choose RETRY_NOW on the same payment, outcomes must match under CRN."""
    runner = EvaluationRunner()
    fixed_retry = FixedRetryStrategy()
    rule_based = RuleBasedStrategy()

    res_fr = runner.evaluate_strategy_on_partition(
        strategy=fixed_retry,
        test_observable=sim_partition.test_observable,
        test_ground_truth=sim_partition.test_ground_truth,
        seed=999,
    )
    res_rb = runner.evaluate_strategy_on_partition(
        strategy=rule_based,
        test_observable=sim_partition.test_observable,
        test_ground_truth=sim_partition.test_ground_truth,
        seed=999,
    )

    matches = 0
    for r_fr, r_rb in zip(res_fr.records, res_rb.records):
        if r_fr.authorized_action == r_rb.authorized_action:
            assert r_fr.recovered == r_rb.recovered
            matches += 1

    assert matches > 0, "Expected at least one payment where actions match"
