"""Sprint 15 — Contract Tests: Uncertainty-Aware Hybrid Sequential Policy for RecoverIQ.

Covers:
1. Deterministic hybrid decisions.
2. No oracle/future-outcome access.
3. Training/evaluation partition isolation.
4. Uncertainty calculation correctness.
5. Equal-weight behavior.
6. Fixed-weight behavior.
7. Uncertainty-aware weighting behavior.
8. Bellman/ModelFree action-value integration.
9. Terminal-action handling.
10. Multi-step trajectory continuation.
11. Original strategy immutability.
12. Regression compatibility.
"""

from decimal import Decimal
import math
import pytest

from recoveriq.domain.actions import Action
from recoveriq.domain.models import CustomerTier, FailureCategory, PaymentContext
from recoveriq.domain.state import PaymentState
from recoveriq.evaluation.bellman_policy import BellmanRecoverIQStrategy
from recoveriq.evaluation.hybrid_policy import (
    HybridActionEvaluation,
    HybridDecision,
    HybridRecoverIQStrategy,
    HybridRegime,
    UncertaintyEstimator,
)
from recoveriq.evaluation.model_free_policy import (
    FittedQIterationPolicy,
    ModelFreeRecoverIQStrategy,
    _make_state,
    train_model_free_policy,
)
from recoveriq.evaluation.sequential_policy import TieredRecoverIQStrategy
from recoveriq.evaluation.strategies import RecoverIQStrategy, RuleBasedStrategy
from recoveriq.evaluation.trajectory import (
    AlwaysStopStrategy,
    TrajectoryEvaluationRunner,
)
from recoveriq.model.trainer import ModelTrainer
from recoveriq.simulation.config import SimulationConfig
from recoveriq.simulation.environment import SimulationEnvironment
from recoveriq.simulation.generator import SyntheticPaymentGenerator
from recoveriq.simulation.partitioner import partition_dataset


@pytest.fixture(scope="module")
def small_dataset():
    cfg = SimulationConfig(n_payments=200, n_customers=50, train_fraction=0.75)
    gen = SyntheticPaymentGenerator(cfg)
    dataset = gen.generate(seed=42)
    return partition_dataset(dataset, train_fraction=cfg.train_fraction)


@pytest.fixture(scope="module")
def trained_model(small_dataset):
    train_env = SimulationEnvironment(small_dataset.train_ground_truth, seed=42)
    trainer = ModelTrainer(c_regularization=1.0, random_state=42)
    return trainer.train(small_dataset.train_observable, train_env)


@pytest.fixture(scope="module")
def sample_context(small_dataset):
    rec = small_dataset.test_observable[0]
    return PaymentContext(
        payment_id=rec.payment_id,
        customer_id=rec.customer_id,
        customer_tier=rec.customer_tier,
        payment_method=rec.payment_method,
        raw_error_code=rec.raw_error_code,
        raw_error_message=rec.raw_error_message,
        failure_category=rec.failure_category,
        failure_severity=rec.failure_severity,
        attempt_count=1,
        extra_metadata={"amount": float(rec.amount)},
    )


@pytest.fixture(scope="module")
def bellman_strategy(trained_model):
    return BellmanRecoverIQStrategy(
        probability_model=trained_model,
        max_attempts=3,
        planning_horizon=3,
    )


@pytest.fixture(scope="module")
def modelfree_strategy(small_dataset, trained_model):
    runner = TrajectoryEvaluationRunner(max_attempts=3, scheduled_cooldown_seconds=900)
    strat = RecoverIQStrategy(probability_model=trained_model)
    strat.name = "RecoverIQ"

    train_episodes = []
    train_env = SimulationEnvironment(small_dataset.train_ground_truth, seed=42)
    for rec in small_dataset.train_observable:
        ep = runner.evaluate_episode(rec, strat, train_env)
        train_episodes.append(ep)

    policy = train_model_free_policy(
        training_episodes_by_strategy={"RecoverIQ": train_episodes},
        training_records=small_dataset.train_observable,
    )
    return ModelFreeRecoverIQStrategy(fitted_policy=policy)


@pytest.fixture(scope="module")
def hybrid_strategy(bellman_strategy, modelfree_strategy):
    return HybridRecoverIQStrategy(
        bellman_strategy=bellman_strategy,
        modelfree_strategy=modelfree_strategy,
        regime=HybridRegime.UNCERTAINTY_AWARE,
    )


# 1. Deterministic hybrid decisions
def test_hybrid_decision_is_deterministic(hybrid_strategy, small_dataset, sample_context):
    rec = small_dataset.test_observable[0]
    action1 = hybrid_strategy.propose_action(rec, sample_context)
    action2 = hybrid_strategy.propose_action(rec, sample_context)
    assert action1 == action2
    assert action1 in set(Action)


# 2. No oracle/future-outcome access
def test_hybrid_no_oracle_access(hybrid_strategy):
    assert not hasattr(hybrid_strategy, "ground_truth")
    assert not hasattr(hybrid_strategy, "test_ground_truth")
    # Verify ModelFree policy also has no oracle
    assert not hasattr(hybrid_strategy.modelfree_strategy, "probability_model")


# 3. Training/evaluation partition isolation
def test_training_evaluation_partition_isolation(small_dataset):
    train_pids = {r.payment_id for r in small_dataset.train_observable}
    test_pids = {r.payment_id for r in small_dataset.test_observable}
    assert train_pids.isdisjoint(test_pids)


# 4. Uncertainty calculation correctness
def test_uncertainty_calculation_correctness(hybrid_strategy, sample_context, small_dataset):
    rec = small_dataset.test_observable[0]
    estimator = hybrid_strategy.uncertainty_estimator
    state = _make_state(sample_context.failure_category, sample_context.customer_tier, 1)

    # Check ModelFree confidence is strictly in (0, 1]
    for a in Action:
        c_mf = estimator.get_modelfree_confidence(state, a)
        assert 0.0 < c_mf <= 1.0
        assert math.isfinite(c_mf)

    # Check Bellman confidence is in (0, 1]
    bellman_evals = hybrid_strategy.bellman_strategy.evaluate_q_values(rec, sample_context, 1, 3)
    for ev in bellman_evals:
        c_b = estimator.get_bellman_confidence(
            sample_context, ev.action, ev, hybrid_strategy.bellman_strategy.probability_model
        )
        assert 0.0 < c_b <= 1.0
        assert math.isfinite(c_b)


# 5. Equal-weight behavior
def test_equal_weight_hybrid_behavior(bellman_strategy, modelfree_strategy, small_dataset, sample_context):
    rec = small_dataset.test_observable[0]
    strat_eq = HybridRecoverIQStrategy(
        bellman_strategy=bellman_strategy,
        modelfree_strategy=modelfree_strategy,
        regime=HybridRegime.EQUAL_WEIGHT,
    )
    evals = strat_eq.evaluate_hybrid_actions(rec, sample_context)
    for ev in evals:
        assert ev.weight_bellman == Decimal("0.50")
        assert ev.weight_modelfree == Decimal("0.50")
        expected_q = Decimal("0.50") * ev.q_bellman + Decimal("0.50") * ev.q_modelfree
        assert abs(ev.q_hybrid - expected_q) < Decimal("0.001")


# 6. Fixed-weight behavior
def test_fixed_weight_hybrid_behavior(bellman_strategy, modelfree_strategy, small_dataset, sample_context):
    rec = small_dataset.test_observable[0]
    strat_fx = HybridRecoverIQStrategy(
        bellman_strategy=bellman_strategy,
        modelfree_strategy=modelfree_strategy,
        regime=HybridRegime.FIXED_WEIGHT,
        fixed_bellman_weight=0.75,
    )
    evals = strat_fx.evaluate_hybrid_actions(rec, sample_context)
    for ev in evals:
        assert ev.weight_bellman == Decimal("0.75")
        assert ev.weight_modelfree == Decimal("0.25")
        expected_q = Decimal("0.75") * ev.q_bellman + Decimal("0.25") * ev.q_modelfree
        assert abs(ev.q_hybrid - expected_q) < Decimal("0.001")


# 7. Uncertainty-aware weighting behavior
def test_uncertainty_aware_weighting_behavior(hybrid_strategy, small_dataset, sample_context):
    rec = small_dataset.test_observable[0]
    evals = hybrid_strategy.evaluate_hybrid_actions(rec, sample_context)
    for ev in evals:
        assert ev.weight_bellman + ev.weight_modelfree == Decimal("1.0")
        assert ev.weight_bellman > Decimal("0.0")
        assert ev.weight_modelfree > Decimal("0.0")
        expected_q = ev.weight_bellman * ev.q_bellman + ev.weight_modelfree * ev.q_modelfree
        assert abs(ev.q_hybrid - expected_q) < Decimal("0.001")


# 8. Bellman/ModelFree action-value integration
def test_action_value_integration(hybrid_strategy, small_dataset, sample_context):
    rec = small_dataset.test_observable[0]
    evals = hybrid_strategy.evaluate_hybrid_actions(rec, sample_context)
    actions_evaluated = {ev.action for ev in evals}
    assert actions_evaluated == set(Action)
    # Sorted descending by q_hybrid
    for i in range(len(evals) - 1):
        assert evals[i].q_hybrid >= evals[i + 1].q_hybrid


# 9. Terminal-action handling
def test_terminal_action_handling(hybrid_strategy, small_dataset):
    runner = TrajectoryEvaluationRunner(max_attempts=3, scheduled_cooldown_seconds=900)
    env = SimulationEnvironment(small_dataset.test_ground_truth, seed=42)
    for rec in small_dataset.test_observable[:15]:
        ep = runner.evaluate_episode(rec, hybrid_strategy, env)
        if ep.terminal_state in (PaymentState.ESCALATED, PaymentState.FAILED_TERMINAL):
            last_step = ep.steps[-1]
            assert last_step.authorized_action in (Action.ESCALATE, Action.STOP) or len(ep.steps) == 3


# 10. Multi-step trajectory continuation
def test_multistep_trajectory_continuation(hybrid_strategy, small_dataset):
    runner = TrajectoryEvaluationRunner(max_attempts=3, scheduled_cooldown_seconds=900)
    env = SimulationEnvironment(small_dataset.test_ground_truth, seed=42)
    episodes = [runner.evaluate_episode(r, hybrid_strategy, env) for r in small_dataset.test_observable[:25]]
    attempts = [len(ep.steps) for ep in episodes]
    assert max(attempts) >= 2, "Hybrid strategy must execute multi-step trajectories when needed."


# 11. Original strategy immutability
def test_original_strategies_immutable(bellman_strategy, modelfree_strategy, small_dataset, sample_context):
    rec = small_dataset.test_observable[0]
    b_action = bellman_strategy.propose_action(rec, sample_context)
    mf_action = modelfree_strategy.propose_action(rec, sample_context)
    assert b_action in set(Action)
    assert mf_action in set(Action)
    assert bellman_strategy.name == "RecoverIQ-Bellman"
    assert modelfree_strategy.name == "RecoverIQ-ModelFree"


# 12. Regression compatibility with existing baselines
def test_regression_compatibility(trained_model, small_dataset, sample_context):
    rec = small_dataset.test_observable[0]
    rule_strat = RuleBasedStrategy()
    tiered_strat = TieredRecoverIQStrategy(probability_model=trained_model)
    stop_strat = AlwaysStopStrategy()

    assert rule_strat.propose_action(rec, sample_context) in set(Action)
    assert tiered_strat.propose_action(rec, sample_context) in set(Action)
    assert stop_strat.propose_action(rec, sample_context) == Action.STOP
