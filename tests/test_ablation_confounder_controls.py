"""Sprint 6 Tests — Confounder controls and scientific validity across ablation variants."""

import pytest

from recoveriq.domain.actions import Action
from recoveriq.evaluation.ablation_strategies import (
    RecoverIQCtxAblationStrategy,
    RecoverIQNoEconStrategy,
)
from recoveriq.evaluation.runner import EvaluationRunner
from recoveriq.evaluation.strategies import FixedRetryStrategy, RecoverIQStrategy, RuleBasedStrategy
from recoveriq.model.trainer import ModelTrainer
from recoveriq.simulation.config import SimulationConfig
from recoveriq.simulation.environment import SimulationEnvironment
from recoveriq.simulation.generator import SyntheticPaymentGenerator
from recoveriq.simulation.partitioner import partition_dataset


@pytest.fixture
def sim_partition():
    cfg = SimulationConfig(n_payments=100, n_customers=20, default_seed=42)
    gen = SyntheticPaymentGenerator(cfg)
    ds = gen.generate(seed=42)
    return partition_dataset(ds, train_fraction=0.75)


@pytest.fixture
def trained_model(sim_partition):
    train_env = SimulationEnvironment(sim_partition.train_ground_truth, seed=42)
    trainer = ModelTrainer(c_regularization=1.0, random_state=42)
    return trainer.train(sim_partition.train_observable, train_env)


def test_crn_invariance_across_all_five_variants(sim_partition, trained_model):
    """Under CRN, any two variants that authorize the same action on payment k receive the identical outcome."""
    runner = EvaluationRunner()

    fixed_retry = FixedRetryStrategy()
    rule_based = RuleBasedStrategy()
    recoveriq_full = RecoverIQStrategy(probability_model=trained_model)
    ctx_ablation = RecoverIQCtxAblationStrategy(probability_model=trained_model)
    no_econ = RecoverIQNoEconStrategy(probability_model=trained_model)

    results = runner.evaluate_all_strategies(
        strategies=[fixed_retry, rule_based, recoveriq_full, ctx_ablation, no_econ],
        test_observable=sim_partition.test_observable,
        test_ground_truth=sim_partition.test_ground_truth,
        seed=42,
    )

    records_by_strat = {name: {r.payment_id: r for r in res.records} for name, res in results.items()}

    strat_names = list(results.keys())
    crn_checks = 0

    for i in range(len(strat_names)):
        for j in range(i + 1, len(strat_names)):
            s1, s2 = strat_names[i], strat_names[j]
            r1_map = records_by_strat[s1]
            r2_map = records_by_strat[s2]

            for pid, r1 in r1_map.items():
                r2 = r2_map[pid]
                if r1.authorized_action == r2.authorized_action:
                    assert r1.recovered == r2.recovered, (
                        f"CRN violation between {s1} and {s2} on payment {pid}: "
                        f"action={r1.authorized_action}, r1.recovered={r1.recovered}, r2.recovered={r2.recovered}"
                    )
                    crn_checks += 1

    assert crn_checks > 0, "Expected at least one action match across strategies under CRN"


def test_common_policy_gate_all_five_strategies_zero_violations(sim_partition, trained_model):
    """All five strategies must undergo InvariantPolicyGate with 0.00% violation rate."""
    runner = EvaluationRunner()

    strats = [
        FixedRetryStrategy(),
        RuleBasedStrategy(),
        RecoverIQStrategy(probability_model=trained_model),
        RecoverIQCtxAblationStrategy(probability_model=trained_model),
        RecoverIQNoEconStrategy(probability_model=trained_model),
    ]

    results = runner.evaluate_all_strategies(
        strategies=strats,
        test_observable=sim_partition.test_observable,
        test_ground_truth=sim_partition.test_ground_truth,
        seed=42,
    )

    for name, metrics in results.items():
        assert metrics.policy_violation_rate == 0.0
        assert len(metrics.records) == len(sim_partition.test_observable)
        for r in metrics.records:
            assert isinstance(r.authorized_action, Action)


def test_ablation_evaluates_strictly_test_partition(sim_partition, trained_model):
    """Ablation evaluations must only touch held-out test payments."""
    runner = EvaluationRunner()
    ctx_ablation = RecoverIQCtxAblationStrategy(probability_model=trained_model)
    no_econ = RecoverIQNoEconStrategy(probability_model=trained_model)

    res = runner.evaluate_all_strategies(
        strategies=[ctx_ablation, no_econ],
        test_observable=sim_partition.test_observable,
        test_ground_truth=sim_partition.test_ground_truth,
        seed=42,
    )

    test_pids = set(sim_partition.test_payment_ids())
    train_pids = set(sim_partition.train_payment_ids())

    for name, metrics in res.items():
        eval_pids = {r.payment_id for r in metrics.records}
        assert eval_pids == test_pids
        assert len(eval_pids & train_pids) == 0
