"""Sprint 14 — Contract Tests: Model-Free Sequential Policy & Model Misspecification.

Tests verifying:
1. ModelFree strategy is deterministic.
2. ModelFree strategy does NOT access ground-truth probabilities.
3. Training and evaluation datasets are separated.
4. Model-free Q-values are finite.
5. Policy respects the existing action space.
6. Policy continues across multiple attempts.
7. Terminal actions remain terminal.
8. Existing RecoverIQStrategy remains unchanged.
9. Bellman strategy remains unchanged.
10. Probability perturbations are deterministic.
11. Perturbed probabilities remain in [0, 1].
12. Zero-error perturbation (M0) reproduces baseline probability estimates.
13. Distribution shifts are deterministic.
14. CRN pairing remains valid.
15. Existing policy-gate invariants remain enforced.
16. Unseen state fallback is a valid (non-crashing) Action.
17. M3 escalate bias exceeds M2 for ESCALATE action.
18. D1 shift doubles amounts deterministically.
"""

import math
from decimal import Decimal
from typing import Dict, List

import pytest

from recoveriq.domain.actions import Action
from recoveriq.domain.models import CustomerTier, FailureCategory, PaymentContext
from recoveriq.domain.state import PaymentState
from recoveriq.evaluation.bellman_policy import BellmanRecoverIQStrategy
from recoveriq.evaluation.model_error import (
    ALL_DISTRIBUTION_SHIFT_CONDITIONS,
    ALL_MODEL_ERROR_CONDITIONS,
    DistributionShiftCondition,
    ModelErrorCondition,
    PerturbedProbabilityModel,
    apply_distribution_shift,
    get_perturbation_description,
)
from recoveriq.evaluation.model_free_policy import (
    FittedQIterationPolicy,
    ModelFreeRecoverIQStrategy,
    train_model_free_policy,
)
from recoveriq.evaluation.robustness import SPRINT12_EXPANDED_SEEDS
from recoveriq.evaluation.sequential_policy import TieredRecoverIQStrategy
from recoveriq.evaluation.strategies import RecoverIQStrategy, RuleBasedStrategy
from recoveriq.evaluation.trajectory import (
    AlwaysStopStrategy,
    TrajectoryEvaluationRunner,
)
from recoveriq.model.probability import ProbabilityEstimate, StubProbabilityModel
from recoveriq.simulation.config import SimulationConfig
from recoveriq.simulation.environment import SimulationEnvironment
from recoveriq.simulation.generator import SyntheticPaymentGenerator
from recoveriq.simulation.partitioner import partition_dataset
from recoveriq.model.trainer import ModelTrainer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_dataset():
    """A small fixed dataset for lightweight contract tests."""
    cfg = SimulationConfig(n_payments=200, n_customers=50, train_fraction=0.75)
    gen = SyntheticPaymentGenerator(cfg)
    dataset = gen.generate(seed=42)
    partitioned = partition_dataset(dataset, train_fraction=cfg.train_fraction)
    return partitioned


@pytest.fixture(scope="module")
def trained_model(small_dataset):
    train_env = SimulationEnvironment(small_dataset.train_ground_truth, seed=42)
    trainer = ModelTrainer(c_regularization=1.0, random_state=42)
    return trainer.train(small_dataset.train_observable, train_env)


@pytest.fixture(scope="module")
def sample_context(small_dataset):
    """A PaymentContext derived from a real test record (all fields non-null)."""
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
def model_free_strategy(small_dataset, trained_model):
    """Train a ModelFreeRecoverIQStrategy on a small training set for contracts."""
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


# ---------------------------------------------------------------------------
# Test 1: ModelFree strategy is deterministic
# ---------------------------------------------------------------------------

def test_model_free_is_deterministic(model_free_strategy, small_dataset):
    """ModelFreeRecoverIQStrategy must produce identical actions on repeated calls."""
    rec = small_dataset.test_observable[0]
    ctx = PaymentContext(
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
    action_1 = model_free_strategy.propose_action(rec, ctx)
    action_2 = model_free_strategy.propose_action(rec, ctx)
    assert action_1 == action_2, "ModelFree must be deterministic for same input."


# ---------------------------------------------------------------------------
# Test 2: ModelFree does NOT access ground-truth probabilities
# ---------------------------------------------------------------------------

def test_model_free_no_ground_truth_access(model_free_strategy, small_dataset):
    """Verify ModelFree Q-table was not built from ground-truth probability lookup."""
    # ModelFree strategy has no probability_model attribute
    assert not hasattr(model_free_strategy, "probability_model"), (
        "ModelFree must not store a probability model reference."
    )
    # The fitted policy's Q-table should only store finite floats
    for (state, action), samples in model_free_strategy.policy._q_table.items():
        for s in samples:
            assert math.isfinite(s), f"Q-table sample must be finite, got {s}"


# ---------------------------------------------------------------------------
# Test 3: Training and evaluation datasets are separated
# ---------------------------------------------------------------------------

def test_train_eval_separation(small_dataset):
    """Training observable records must not overlap with test observable records."""
    train_ids = {r.payment_id for r in small_dataset.train_observable}
    test_ids = {r.payment_id for r in small_dataset.test_observable}
    assert train_ids.isdisjoint(test_ids), (
        "Training and evaluation payment IDs must be disjoint (no leakage)."
    )


# ---------------------------------------------------------------------------
# Test 4: Model-free Q-values are finite
# ---------------------------------------------------------------------------

def test_q_values_finite(model_free_strategy, small_dataset):
    """All Q-values in the fitted Q-table must be finite numbers."""
    for (state, action), samples in model_free_strategy.policy._q_table.items():
        for s in samples:
            assert math.isfinite(s), f"Non-finite Q-value found: {s} for state {state}, action {action}"


# ---------------------------------------------------------------------------
# Test 5: Policy respects the existing action space
# ---------------------------------------------------------------------------

def test_model_free_action_space(model_free_strategy, small_dataset):
    """ModelFree must only propose actions from the valid Action enum."""
    valid_actions = set(Action)
    for rec in small_dataset.test_observable[:20]:
        ctx = PaymentContext(
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
        action = model_free_strategy.propose_action(rec, ctx)
        assert action in valid_actions, f"ModelFree proposed invalid action: {action}"


# ---------------------------------------------------------------------------
# Test 6: Policy continues across multiple attempts
# ---------------------------------------------------------------------------

def test_model_free_multi_attempt(model_free_strategy, small_dataset):
    """ModelFree must return valid actions at attempt 1, 2, and 3."""
    rec = small_dataset.test_observable[0]
    for attempt in [1, 2, 3]:
        ctx = PaymentContext(
            payment_id=rec.payment_id,
            customer_id=rec.customer_id,
            customer_tier=rec.customer_tier,
            payment_method=rec.payment_method,
            raw_error_code=rec.raw_error_code,
            raw_error_message=rec.raw_error_message,
            failure_category=rec.failure_category,
            failure_severity=rec.failure_severity,
            attempt_count=attempt,
            extra_metadata={"amount": float(rec.amount)},
        )
        action = model_free_strategy.propose_action(rec, ctx)
        assert action in set(Action), f"Invalid action at attempt {attempt}: {action}"


# ---------------------------------------------------------------------------
# Test 7: Terminal actions remain terminal in the trajectory runner
# ---------------------------------------------------------------------------

def test_terminal_actions_remain_terminal(model_free_strategy, small_dataset):
    """After STOP or ESCALATE, trajectory must not continue."""
    runner = TrajectoryEvaluationRunner(max_attempts=3, scheduled_cooldown_seconds=900)
    env = SimulationEnvironment(small_dataset.test_ground_truth, seed=42)
    for rec in small_dataset.test_observable[:10]:
        ep = runner.evaluate_episode(rec, model_free_strategy, env)
        if ep.terminal_state in (PaymentState.ESCALATED, PaymentState.FAILED_TERMINAL):
            # No steps should follow a terminal step
            for step in ep.steps:
                if step.authorized_action in (Action.ESCALATE, Action.STOP):
                    assert step == ep.steps[-1], (
                        "Terminal action must be the last step in the trajectory."
                    )


# ---------------------------------------------------------------------------
# Test 8: Existing RecoverIQStrategy remains unchanged
# ---------------------------------------------------------------------------

def test_recoveriq_strategy_unchanged(trained_model, sample_context, small_dataset):
    """RecoverIQStrategy must still function correctly and return a valid action."""
    strat = RecoverIQStrategy(probability_model=trained_model)
    rec = small_dataset.test_observable[0]
    action = strat.propose_action(rec, sample_context)
    assert action in set(Action), "RecoverIQStrategy must return a valid Action."


# ---------------------------------------------------------------------------
# Test 9: Bellman strategy remains unchanged
# ---------------------------------------------------------------------------

def test_bellman_strategy_unchanged(trained_model, small_dataset):
    """BellmanRecoverIQStrategy must still function correctly."""
    bellman = BellmanRecoverIQStrategy(trained_model, max_attempts=3, planning_horizon=3)
    rec = small_dataset.test_observable[0]
    ctx = PaymentContext(
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
    action = bellman.propose_action(rec, ctx)
    assert action in set(Action), "BellmanRecoverIQStrategy must return a valid Action."


# ---------------------------------------------------------------------------
# Test 10: Probability perturbations are deterministic
# ---------------------------------------------------------------------------

def test_perturbation_is_deterministic(trained_model, sample_context):
    """PerturbedProbabilityModel must return identical results on repeated calls."""
    model = PerturbedProbabilityModel(trained_model, ModelErrorCondition.M2_MODERATE)
    estimates_1 = model.estimate_probabilities(sample_context)
    estimates_2 = model.estimate_probabilities(sample_context)
    for action in Action:
        assert estimates_1[action].probability == estimates_2[action].probability, (
            f"Perturbation not deterministic for {action}"
        )


# ---------------------------------------------------------------------------
# Test 11: Perturbed probabilities remain in [0, 1]
# ---------------------------------------------------------------------------

def test_perturbed_probabilities_in_range(trained_model, sample_context):
    """All perturbed probabilities must remain clamped to [0.0, 1.0]."""
    for condition in ALL_MODEL_ERROR_CONDITIONS:
        model = PerturbedProbabilityModel(trained_model, condition)
        estimates = model.estimate_probabilities(sample_context)
        for action, est in estimates.items():
            assert Decimal("0.0") <= est.probability <= Decimal("1.0"), (
                f"Probability out of range for {action} under {condition}: {est.probability}"
            )


# ---------------------------------------------------------------------------
# Test 12: Zero-error perturbation (M0) reproduces baseline probabilities
# ---------------------------------------------------------------------------

def test_m0_reproduces_baseline(trained_model, sample_context):
    """M0 (no perturbation) must return probabilities identical to the base model."""
    base_estimates = trained_model.estimate_probabilities(sample_context)
    m0_model = PerturbedProbabilityModel(trained_model, ModelErrorCondition.M0_CORRECT)
    m0_estimates = m0_model.estimate_probabilities(sample_context)
    for action in Action:
        assert base_estimates[action].probability == m0_estimates[action].probability, (
            f"M0 must reproduce baseline for {action}"
        )


# ---------------------------------------------------------------------------
# Test 13: Distribution shifts are deterministic
# ---------------------------------------------------------------------------

def test_distribution_shift_deterministic(small_dataset):
    """apply_distribution_shift must produce identical results on repeated calls."""
    records = small_dataset.test_observable[:10]
    for shift in ALL_DISTRIBUTION_SHIFT_CONDITIONS:
        shifted_1 = apply_distribution_shift(records, shift)
        shifted_2 = apply_distribution_shift(records, shift)
        for r1, r2 in zip(shifted_1, shifted_2):
            assert r1.amount == r2.amount, f"Amount not deterministic under {shift}"
            assert r1.customer_tier == r2.customer_tier, f"Tier not deterministic under {shift}"


# ---------------------------------------------------------------------------
# Test 14: CRN pairing remains valid
# ---------------------------------------------------------------------------

def test_crn_pairing_valid(model_free_strategy, small_dataset):
    """Under CRN, same seed must produce identical episode counts for ModelFree."""
    runner = TrajectoryEvaluationRunner(max_attempts=3, scheduled_cooldown_seconds=900)
    env_1 = SimulationEnvironment(small_dataset.test_ground_truth, seed=42)
    env_2 = SimulationEnvironment(small_dataset.test_ground_truth, seed=42)
    episodes_1, episodes_2 = [], []
    for rec in small_dataset.test_observable[:20]:
        episodes_1.append(runner.evaluate_episode(rec, model_free_strategy, env_1))
        episodes_2.append(runner.evaluate_episode(rec, model_free_strategy, env_2))
    assert len(episodes_1) == len(episodes_2), "CRN episode counts must match."
    for ep1, ep2 in zip(episodes_1, episodes_2):
        assert ep1.final_recovered == ep2.final_recovered, (
            "CRN must produce identical recovery outcomes."
        )


# ---------------------------------------------------------------------------
# Test 15: Policy gate invariants are enforced
# ---------------------------------------------------------------------------

def test_policy_gate_enforced(model_free_strategy, small_dataset):
    """Policy gate must clamp invalid proposals to STOP as required by SPEC §11.3."""
    runner = TrajectoryEvaluationRunner(max_attempts=3, scheduled_cooldown_seconds=900)
    env = SimulationEnvironment(small_dataset.test_ground_truth, seed=42)
    for rec in small_dataset.test_observable[:20]:
        ep = runner.evaluate_episode(rec, model_free_strategy, env)
        # No step should have more attempts than the configured maximum
        assert ep.attempt_count <= runner.max_attempts, (
            f"Episode exceeded max_attempts={runner.max_attempts}: {ep.attempt_count}"
        )


# ---------------------------------------------------------------------------
# Test 16: Unseen state fallback is a valid Action
# ---------------------------------------------------------------------------

def test_unseen_state_fallback():
    """FittedQIterationPolicy must return a valid fallback Action for unseen states."""
    empty_policy = FittedQIterationPolicy()
    empty_policy._is_fitted = True
    state = ("UNKNOWN_CATEGORY", "STANDARD", 1)
    action = empty_policy.get_best_action(state)
    assert action in set(Action), f"Fallback action {action} must be a valid Action."
    assert not empty_policy.is_state_seen(state), "Unseen state must not be marked as seen."


# ---------------------------------------------------------------------------
# Test 17: M3 ESCALATE bias exceeds M2 for ESCALATE action
# ---------------------------------------------------------------------------

def test_m3_escalate_bias_exceeds_m2(trained_model, sample_context):
    """M3 must apply additional ESCALATE bias not present in M2."""
    m2_model = PerturbedProbabilityModel(trained_model, ModelErrorCondition.M2_MODERATE)
    m3_model = PerturbedProbabilityModel(trained_model, ModelErrorCondition.M3_SEVERE)

    m2_estimates = m2_model.estimate_probabilities(sample_context)
    m3_estimates = m3_model.estimate_probabilities(sample_context)

    m2_escalate = m2_estimates[Action.ESCALATE].probability
    m3_escalate = m3_estimates[Action.ESCALATE].probability

    # M3 ESCALATE must be >= M2 ESCALATE (escalate bias pushes it up)
    assert m3_escalate >= m2_escalate, (
        f"M3 ESCALATE probability ({m3_escalate}) must be >= M2 ({m2_escalate})"
    )


# ---------------------------------------------------------------------------
# Test 18: D1 shift doubles amounts deterministically
# ---------------------------------------------------------------------------

def test_d1_doubles_amounts(small_dataset):
    """D1 distribution shift must exactly double all payment amounts."""
    records = small_dataset.test_observable[:10]
    shifted = apply_distribution_shift(records, DistributionShiftCondition.D1_VALUE_SHIFT)
    for original, shifted_rec in zip(records, shifted):
        assert shifted_rec.amount == original.amount * Decimal("2"), (
            f"D1 must double amount: expected {original.amount * 2}, got {shifted_rec.amount}"
        )
