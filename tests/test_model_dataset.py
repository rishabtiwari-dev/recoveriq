"""Sprint 4 Tests — Counterfactual dataset generation and observable feature extraction."""

from decimal import Decimal
import pytest

from recoveriq.domain.actions import Action
from recoveriq.domain.models import CustomerTier, FailureCategory, FailureSeverity, PaymentMethod
from recoveriq.model.dataset import CounterfactualDatasetBuilder, extract_observable_features
from recoveriq.simulation.environment import SimulationEnvironment
from recoveriq.simulation.generator import SyntheticPaymentGenerator
from recoveriq.simulation.config import SimulationConfig


@pytest.fixture
def sim_dataset():
    cfg = SimulationConfig(n_payments=50, n_customers=10, default_seed=42)
    gen = SyntheticPaymentGenerator(cfg)
    return gen.generate(seed=42)


def test_extract_observable_features_strictly_decision_time(sim_dataset):
    """Observable features must only contain fields known at decision time."""
    record = sim_dataset.observable_records[0]
    features = extract_observable_features(record)

    expected_keys = {
        "failure_category",
        "failure_severity",
        "customer_tier",
        "payment_method",
        "amount",
        "attempt_count",
        "failure_hour",
        "failure_day_of_week",
    }
    assert set(features.keys()) == expected_keys

    # Confirm hidden ground truth fields are NOT present
    forbidden_keys = {"true_probability", "latent_recoverability_profile", "action_base_probabilities"}
    for f in forbidden_keys:
        assert f not in features


def test_counterfactual_dataset_builder_full_factorial(sim_dataset):
    """Dataset builder must construct exactly N x 6 counterfactual observations."""
    env = SimulationEnvironment(sim_dataset.ground_truth_records, seed=42)
    builder = CounterfactualDatasetBuilder()

    n_payments = len(sim_dataset.observable_records)
    dataset = builder.build_dataset(sim_dataset.observable_records, env, seed=42)

    assert len(dataset) == n_payments * len(Action)

    # Every action must have exactly n_payments examples
    for action in Action:
        action_exs = dataset.get_action_examples(action)
        assert len(action_exs) == n_payments


def test_counterfactual_observations_share_identical_features(sim_dataset):
    """For any given payment, counterfactual rows across all 6 actions must share identical observable features."""
    env = SimulationEnvironment(sim_dataset.ground_truth_records, seed=42)
    builder = CounterfactualDatasetBuilder()
    dataset = builder.build_dataset(sim_dataset.observable_records, env, seed=42)

    target_pid = sim_dataset.observable_records[0].payment_id
    exs = [ex for ex in dataset.examples if ex.payment_id == target_pid]

    assert len(exs) == 6
    first_features = exs[0].features
    for other in exs[1:]:
        assert other.features == first_features


def test_labels_are_strictly_binary(sim_dataset):
    """Labels in the counterfactual dataset must strictly be integers in {0, 1}."""
    env = SimulationEnvironment(sim_dataset.ground_truth_records, seed=42)
    builder = CounterfactualDatasetBuilder()
    dataset = builder.build_dataset(sim_dataset.observable_records, env, seed=42)

    _, labels = dataset.get_feature_matrix_and_labels()
    for y in labels:
        assert y in (0, 1)
        assert isinstance(y, int)


def test_stop_action_labels_always_zero(sim_dataset):
    """Because STOP cannot recover payments under the world model, its outcomes are all 0."""
    env = SimulationEnvironment(sim_dataset.ground_truth_records, seed=42)
    builder = CounterfactualDatasetBuilder()
    dataset = builder.build_dataset(sim_dataset.observable_records, env, seed=42)

    _, stop_labels = dataset.get_feature_matrix_and_labels(Action.STOP)
    assert all(y == 0 for y in stop_labels)
