"""Sprint 2 Tests — Stochastic outcomes and action-conditioned behavior."""

import pytest
from collections import defaultdict
from recoveriq.domain.actions import Action
from recoveriq.domain.models import FailureCategory
from recoveriq.simulation.config import SimulationConfig
from recoveriq.simulation.environment import SimulationEnvironment
from recoveriq.simulation.generator import SyntheticPaymentGenerator
from recoveriq.simulation.schema import RecoverabilityProfile


@pytest.fixture(scope="module")
def env_and_dataset():
    cfg = SimulationConfig(n_payments=1000, n_customers=200)
    gen = SyntheticPaymentGenerator(cfg)
    ds = gen.generate(seed=42)
    env = SimulationEnvironment(ds.ground_truth_records, seed=42)
    return env, ds


def test_environment_resolves_outcome_for_all_records(env_and_dataset):
    """Environment must resolve an outcome for every payment_id."""
    env, ds = env_and_dataset
    for record in ds.observable_records:
        outcome = env.apply_action(record.payment_id, Action.RETRY_NOW)
        assert isinstance(outcome.recovered, bool)
        assert outcome.payment_id == record.payment_id
        assert outcome.action == Action.RETRY_NOW


def test_environment_unknown_payment_raises(env_and_dataset):
    """Querying an unregistered payment_id must raise KeyError."""
    env, _ = env_and_dataset
    with pytest.raises(KeyError):
        env.apply_action("pay_does_not_exist", Action.RETRY_NOW)


def test_action_conditioned_outcomes_differ_across_actions(env_and_dataset):
    """Different actions should produce different recovery rates over many payments."""
    env, ds = env_and_dataset
    # Use a fresh env with fixed seed for this measurement
    fresh_env = SimulationEnvironment(ds.ground_truth_records, seed=99)

    rates = {}
    for action in [Action.RETRY_NOW, Action.ESCALATE, Action.STOP]:
        outcomes = fresh_env.batch_apply_action(
            [r.payment_id for r in ds.observable_records], action
        )
        rates[action] = sum(o.recovered for o in outcomes) / len(outcomes)

    # STOP should always be 0.0 (no recovery possible)
    assert rates[Action.STOP] == 0.0, f"STOP action recovered at rate {rates[Action.STOP]}"
    # ESCALATE should have higher recovery rate than RETRY_NOW on average
    # (based on GROUND_TRUTH_RECOVERY_TABLE where ESCALATE has higher probs)
    assert rates[Action.ESCALATE] > rates[Action.RETRY_NOW], (
        f"ESCALATE rate {rates[Action.ESCALATE]:.4f} should exceed RETRY_NOW {rates[Action.RETRY_NOW]:.4f}"
    )


def test_outcomes_are_stochastic_not_deterministic(env_and_dataset):
    """Two independent runs with different seeds should produce different outcomes."""
    env, ds = env_and_dataset
    env_a = SimulationEnvironment(ds.ground_truth_records, seed=111)
    env_b = SimulationEnvironment(ds.ground_truth_records, seed=222)

    outcomes_a = [env_a.apply_action(r.payment_id, Action.RETRY_LATER).recovered
                  for r in ds.observable_records[:100]]
    outcomes_b = [env_b.apply_action(r.payment_id, Action.RETRY_LATER).recovered
                  for r in ds.observable_records[:100]]

    # With different seeds, the outcome sequences should differ for at least one payment
    assert outcomes_a != outcomes_b, "Same outcomes with different seeds — environment is not stochastic!"


def test_outcome_reproducibility_with_same_seed(env_and_dataset):
    """Same environment seed must produce identical outcome sequences."""
    _, ds = env_and_dataset
    env1 = SimulationEnvironment(ds.ground_truth_records, seed=42)
    env2 = SimulationEnvironment(ds.ground_truth_records, seed=42)

    outcomes1 = [env1.apply_action(r.payment_id, Action.SEND_LINK).recovered
                 for r in ds.observable_records[:200]]
    outcomes2 = [env2.apply_action(r.payment_id, Action.SEND_LINK).recovered
                 for r in ds.observable_records[:200]]

    assert outcomes1 == outcomes2


def test_hard_decline_payments_rarely_recover(env_and_dataset):
    """Hard decline payments should have very low recovery rates across 1000 resolutions."""
    env, ds = env_and_dataset
    hard_decline_ids = [
        r.payment_id for r in ds.observable_records
        if r.failure_category == FailureCategory.HARD_DECLINE
    ]
    if len(hard_decline_ids) < 5:
        pytest.skip("Not enough hard decline payments for this test")

    fresh_env = SimulationEnvironment(ds.ground_truth_records, seed=42)
    outcomes = fresh_env.batch_apply_action(hard_decline_ids, Action.RETRY_NOW)
    recovery_rate = sum(o.recovered for o in outcomes) / len(outcomes)

    # Hard declines are mostly VERY_LOW / LOW profiles — expect < 35% recovery
    assert recovery_rate < 0.35, (
        f"Hard decline recovery rate {recovery_rate:.3f} is suspiciously high"
    )


def test_very_high_profile_payments_recover_more_than_very_low(env_and_dataset):
    """VERY_HIGH profile payments should recover at higher rates than VERY_LOW."""
    _, ds = env_and_dataset
    very_high_ids = [
        obs.payment_id for obs, gt in zip(ds.observable_records, ds.ground_truth_records)
        if gt.latent_recoverability_profile == RecoverabilityProfile.VERY_HIGH
    ]
    very_low_ids = [
        obs.payment_id for obs, gt in zip(ds.observable_records, ds.ground_truth_records)
        if gt.latent_recoverability_profile == RecoverabilityProfile.VERY_LOW
    ]
    if not very_high_ids or not very_low_ids:
        pytest.skip("Insufficient records in extreme profiles")

    fresh_env = SimulationEnvironment(ds.ground_truth_records, seed=42)

    high_outcomes = fresh_env.batch_apply_action(very_high_ids, Action.RETRY_LATER)
    low_outcomes = fresh_env.batch_apply_action(very_low_ids, Action.RETRY_LATER)

    high_rate = sum(o.recovered for o in high_outcomes) / len(high_outcomes)
    low_rate = sum(o.recovered for o in low_outcomes) / len(low_outcomes)

    assert high_rate > low_rate, (
        f"VERY_HIGH profile rate {high_rate:.3f} should exceed VERY_LOW {low_rate:.3f}"
    )
