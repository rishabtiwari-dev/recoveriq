"""Sprint 4 Tests — Model trainer, 6 action models, STOP invariant, and reproducibility."""

from decimal import Decimal
import pytest
import numpy as np

from recoveriq.domain.actions import Action
from recoveriq.domain.models import CustomerTier, FailureCategory, FailureSeverity, PaymentContext, PaymentMethod
from recoveriq.model.probability import ProbabilityEstimate, RecoveryProbabilityModel
from recoveriq.model.trainer import ModelTrainer
from recoveriq.simulation.config import SimulationConfig
from recoveriq.simulation.environment import SimulationEnvironment
from recoveriq.simulation.generator import SyntheticPaymentGenerator
from recoveriq.simulation.partitioner import partition_dataset


@pytest.fixture
def train_data():
    cfg = SimulationConfig(n_payments=200, n_customers=50, default_seed=42)
    gen = SyntheticPaymentGenerator(cfg)
    ds = gen.generate(seed=42)
    partitioned = partition_dataset(ds, train_fraction=0.75)
    return partitioned.train_observable, partitioned.train_ground_truth


def test_trainer_produces_six_action_models(train_data):
    """ModelTrainer must produce a model containing all 6 distinct action models."""
    train_obs, train_gt = train_data
    env = SimulationEnvironment(train_gt, seed=42)
    trainer = ModelTrainer(c_regularization=1.0, random_state=42)
    model = trainer.train(train_obs, env)

    assert isinstance(model, RecoveryProbabilityModel)
    assert len(model.action_models) == 6
    for action in Action:
        assert action in model.action_models
        assert model.action_models[action].is_fitted is True


def test_stop_action_is_strictly_deterministic_zero(train_data):
    """Action.STOP must return exactly Decimal('0.00') without statistical variation."""
    train_obs, train_gt = train_data
    env = SimulationEnvironment(train_gt, seed=42)
    trainer = ModelTrainer(c_regularization=1.0, random_state=42)
    model = trainer.train(train_obs, env)

    # Test across multiple distinct contexts
    for tier in CustomerTier:
        for cat in FailureCategory:
            ctx = PaymentContext(
                payment_id="pay_test",
                customer_id="cust_test",
                customer_tier=tier,
                payment_method=PaymentMethod.CREDIT_CARD,
                raw_error_code="CODE",
                raw_error_message="MSG",
                failure_category=cat,
                failure_severity=FailureSeverity.RECOVERABLE,
                attempt_count=1,
            )
            estimates = model.estimate_probabilities(ctx)
            assert estimates[Action.STOP].probability == Decimal("0.00")


def test_model_training_is_reproducible(train_data):
    """Identical training records and random_state must produce identical weights and estimates."""
    train_obs, train_gt = train_data

    env1 = SimulationEnvironment(train_gt, seed=42)
    trainer1 = ModelTrainer(c_regularization=1.0, random_state=42)
    model1 = trainer1.train(train_obs, env1)

    env2 = SimulationEnvironment(train_gt, seed=42)
    trainer2 = ModelTrainer(c_regularization=1.0, random_state=42)
    model2 = trainer2.train(train_obs, env2)

    ctx = PaymentContext(
        payment_id="pay_comp",
        customer_id="cust_comp",
        customer_tier=CustomerTier.VIP,
        payment_method=PaymentMethod.UPI,
        raw_error_code="NSF",
        raw_error_message="Insufficient funds",
        failure_category=FailureCategory.INSUFFICIENT_FUNDS,
        failure_severity=FailureSeverity.RECOVERABLE,
        attempt_count=1,
    )

    est1 = model1.estimate_probabilities(ctx)
    est2 = model2.estimate_probabilities(ctx)

    for action in Action:
        assert est1[action].probability == est2[action].probability

    # Compare raw weight arrays
    for action in Action:
        w1 = model1.action_models[action].weights
        w2 = model2.action_models[action].weights
        np.testing.assert_array_almost_equal(w1, w2)


def test_model_coefficients_are_inspectable(train_data):
    """ActionLogisticRegression must provide feature_name -> weight inspectability."""
    train_obs, train_gt = train_data
    env = SimulationEnvironment(train_gt, seed=42)
    trainer = ModelTrainer(c_regularization=1.0, random_state=42)
    model = trainer.train(train_obs, env)

    coefs = model.action_models[Action.RETRY_NOW].coefficients_dict
    assert isinstance(coefs, dict)
    assert len(coefs) == 26
    assert "cat_INSUFFICIENT_FUNDS" in coefs
    assert "num_amount" in coefs
