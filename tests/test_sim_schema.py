"""Sprint 2 Tests — Schema validity and data integrity."""

from decimal import Decimal
import pytest
from recoveriq.domain.actions import Action
from recoveriq.domain.models import CustomerTier, FailureCategory, FailureSeverity, PaymentMethod
from recoveriq.simulation.config import SimulationConfig
from recoveriq.simulation.generator import SyntheticPaymentGenerator
from recoveriq.simulation.schema import (
    GroundTruthRecord,
    RecoverabilityProfile,
    SyntheticPaymentRecord,
    GROUND_TRUTH_RECOVERY_TABLE,
)


@pytest.fixture(scope="module")
def dataset():
    cfg = SimulationConfig(n_payments=500, n_customers=100)
    gen = SyntheticPaymentGenerator(cfg)
    return gen.generate(seed=42)


def test_dataset_length(dataset):
    """Dataset must contain exactly n_payments records."""
    assert len(dataset) == 500
    assert len(dataset.observable_records) == 500
    assert len(dataset.ground_truth_records) == 500


def test_observable_record_fields_present(dataset):
    """Every observable record must have all required fields populated."""
    for record in dataset.observable_records:
        assert isinstance(record.payment_id, str) and record.payment_id
        assert isinstance(record.customer_id, str) and record.customer_id
        assert isinstance(record.amount, Decimal)
        assert record.amount > 0
        assert isinstance(record.failure_category, FailureCategory)
        assert isinstance(record.failure_severity, FailureSeverity)
        assert isinstance(record.customer_tier, CustomerTier)
        assert isinstance(record.payment_method, PaymentMethod)
        assert isinstance(record.raw_error_code, str) and record.raw_error_code
        assert isinstance(record.raw_error_message, str) and record.raw_error_message
        assert record.attempt_count >= 1


def test_ground_truth_record_fields_present(dataset):
    """Every ground truth record must have profile and action probabilities."""
    for gt in dataset.ground_truth_records:
        assert isinstance(gt.payment_id, str) and gt.payment_id
        assert isinstance(gt.latent_recoverability_profile, RecoverabilityProfile)
        assert len(gt.action_base_probabilities) == len(Action)
        for action, prob in gt.action_base_probabilities.items():
            assert isinstance(action, Action)
            assert 0.0 <= prob <= 1.0, f"Probability {prob} out of range for action {action}"


def test_observable_and_ground_truth_payment_ids_match(dataset):
    """observable_records and ground_truth_records must be aligned by payment_id."""
    for obs, gt in zip(dataset.observable_records, dataset.ground_truth_records):
        assert obs.payment_id == gt.payment_id


def test_payment_ids_are_unique(dataset):
    """All payment_ids must be unique."""
    ids = [r.payment_id for r in dataset.observable_records]
    assert len(ids) == len(set(ids))


def test_amounts_are_decimal_and_positive(dataset):
    """All amounts must be Decimal and strictly positive."""
    for record in dataset.observable_records:
        assert isinstance(record.amount, Decimal)
        assert record.amount > Decimal("0")


def test_ground_truth_table_completeness():
    """GROUND_TRUTH_RECOVERY_TABLE must have entries for every profile × action."""
    for profile in RecoverabilityProfile:
        assert profile in GROUND_TRUTH_RECOVERY_TABLE
        for action in Action:
            assert action in GROUND_TRUTH_RECOVERY_TABLE[profile], (
                f"Action {action} missing from profile {profile}"
            )
            prob = GROUND_TRUTH_RECOVERY_TABLE[profile][action]
            assert 0.0 <= prob <= 1.0
